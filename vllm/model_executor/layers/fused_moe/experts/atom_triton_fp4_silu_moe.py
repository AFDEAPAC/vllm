# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# DEQUANT-FALLBACK MXFP4 MoE for DeepSeek V4-Pro on hardware lacking native
# MXFP4 matmul_ogs support (e.g. AMD MI300X / MI308X = gfx942 = CDNA3).
#
# Why this exists:
#   triton_kernels.tensor_details.layout.make_default_matmul_mxfp4_w_scale_layout
#   only emits a hardware-specific scale layout for:
#     - SM100+ (BlackwellMXScaleLayout)
#     - SM90+ (HopperMXScaleLayout)
#     - gfx950 / CDNA4 (CDNA4MXScaleLayout)
#   On gfx942 / CDNA3 it falls back to StridedLayout, which the matmul_ogs
#   MXFP4 kernels do NOT compute correctly with — silent-garbage output.
#
#   This means *any* TRITON / TRITON_UNFUSED MXFP4 path on gfx942 produces
#   wrong math, regardless of the apply()-flow tweaks. Verified empirically:
#   even with all of PR #41136 / #40931 applied + C kernel rebuild, V4-Pro
#   produces "raeli\n", "the magnetic | the magnetic" loops, GSM8K 0/8.
#
# Solution: bypass matmul_ogs entirely. Dequantize MXFP4 weights to BF16
# once in process_weights_after_loading, then run a standard PyTorch MoE
# forward (gather-by-expert, bmm, silu+clamp, bmm, scatter-sum). Slow but
# correct.
#
# Trade-off: ~5-10x slower than matmul_ogs (no fused kernel), but it's the
# only known-correct path until either (a) triton_kernels gains a CDNA3
# scale layout or (b) AMD lands a native gfx942 mxfp4 MoE in AITER.

from __future__ import annotations

import logging
import json
import os
import time
from math import prod

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)



@triton.jit
def _dsv4_mxfp4_build_active_metadata_kernel(
    topk_ids_ptr,
    active_count_ptr,
    active_ids_ptr,
    expert_counts_ptr,
    positions_ptr,
    overflow_ptr,
    TOTAL: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    MAX_ACTIVE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    write_count = tl.full((), 0, dtype=tl.int32)
    total_count = tl.full((), 0, dtype=tl.int32)

    for start in range(0, TOTAL, BLOCK):
        pos = start + offs
        mask = pos < TOTAL
        ids = tl.load(topk_ids_ptr + pos, mask=mask, other=-999999)
        match = (ids == expert_id) & mask
        match_i = match.to(tl.int32)
        local_rank = tl.cumsum(match_i, 0) - 1
        dst = write_count + local_rank
        store_mask = match & (dst < CAPACITY)
        tl.store(
            positions_ptr + expert_id * CAPACITY + dst,
            pos,
            mask=store_mask,
        )
        block_count = tl.sum(match_i, axis=0)
        write_count += tl.minimum(block_count, CAPACITY - write_count)
        total_count += block_count

    tl.store(expert_counts_ptr + expert_id, total_count)
    if total_count > CAPACITY:
        tl.store(overflow_ptr + expert_id, total_count - CAPACITY)
    else:
        tl.store(overflow_ptr + expert_id, 0)

    if total_count > 0:
        slot = tl.atomic_add(active_count_ptr, 1, sem="relaxed")
        if slot < MAX_ACTIVE:
            tl.store(active_ids_ptr + slot, expert_id)


@triton.jit
def _dsv4_mxfp4_build_active_metadata_page_kernel(
    topk_ids_ptr,
    active_count_ptr,
    active_ids_ptr,
    expert_counts_ptr,
    positions_ptr,
    overflow_ptr,
    TOTAL: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    MAX_ACTIVE: tl.constexpr,
    CAPACITY: tl.constexpr,
    PAGE_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    page_count = tl.full((), 0, dtype=tl.int32)
    total_count = tl.full((), 0, dtype=tl.int32)

    for start in range(0, TOTAL, BLOCK):
        pos = start + offs
        mask = pos < TOTAL
        ids = tl.load(topk_ids_ptr + pos, mask=mask, other=-999999)
        match = (ids == expert_id) & mask
        match_i = match.to(tl.int32)
        local_rank = tl.cumsum(match_i, 0) - 1
        global_rank = total_count + local_rank
        page_rank = global_rank - PAGE_OFFSET
        store_mask = match & (page_rank >= 0) & (page_rank < CAPACITY)
        tl.store(
            positions_ptr + expert_id * CAPACITY + page_rank,
            pos,
            mask=store_mask,
        )
        page_count += tl.sum(store_mask.to(tl.int32), axis=0)
        total_count += tl.sum(match_i, axis=0)

    tl.store(expert_counts_ptr + expert_id, page_count)
    remaining = total_count - PAGE_OFFSET - CAPACITY
    tl.store(overflow_ptr + expert_id, tl.maximum(remaining, 0))

    if page_count > 0:
        slot = tl.atomic_add(active_count_ptr, 1, sem="relaxed")
        if slot < MAX_ACTIVE:
            tl.store(active_ids_ptr + slot, expert_id)


@triton.jit
def _dsv4_mxfp4_reset_active_metadata_kernel(
    active_count_ptr,
    active_ids_ptr,
    expert_counts_ptr,
    positions_ptr,
    overflow_ptr,
    LOCAL_EXPERTS: tl.constexpr,
    MAX_ACTIVE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    max_pos = LOCAL_EXPERTS * CAPACITY
    tl.store(active_count_ptr, 0, mask=pid == 0)
    tl.store(active_ids_ptr + offs, -1, mask=offs < MAX_ACTIVE)
    tl.store(expert_counts_ptr + offs, 0, mask=offs < LOCAL_EXPERTS)
    tl.store(overflow_ptr + offs, 0, mask=offs < LOCAL_EXPERTS)
    tl.store(positions_ptr + offs, 0, mask=offs < max_pos)


@triton.jit
def _dsv4_mxfp4_ragged_reset_kernel(
    counts_ptr,
    cursor_ptr,
    offsets_ptr,
    tile_offsets_ptr,
    block_map_ptr,
    LOCAL_EXPERTS: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(counts_ptr + offs, 0, mask=offs < LOCAL_EXPERTS)
    tl.store(cursor_ptr + offs, 0, mask=offs < LOCAL_EXPERTS)
    tl.store(offsets_ptr + offs, 0, mask=offs < LOCAL_EXPERTS + 1)
    tl.store(tile_offsets_ptr + offs, 0, mask=offs < LOCAL_EXPERTS + 1)
    tl.store(block_map_ptr + offs, -1, mask=offs < MAX_TILES)


@triton.jit
def _dsv4_mxfp4_ragged_count_kernel(
    topk_ids_ptr,
    counts_ptr,
    TOTAL: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (ids >= 0) & (ids < LOCAL_EXPERTS)
    tl.atomic_add(counts_ptr + ids, 1, sem="relaxed", mask=valid)


@triton.jit
def _dsv4_mxfp4_ragged_prefix_kernel(
    counts_ptr,
    offsets_ptr,
    tile_offsets_ptr,
    LOCAL_EXPERTS: tl.constexpr,
    ROUTE_BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < LOCAL_EXPERTS
    counts = tl.load(counts_ptr + offs, mask=mask, other=0)
    route_prefix = tl.cumsum(counts, 0)
    tile_counts = (counts + ROUTE_BLOCK_M - 1) // ROUTE_BLOCK_M
    tile_prefix = tl.cumsum(tile_counts, 0)
    tl.store(offsets_ptr + offs, route_prefix - counts, mask=mask)
    tl.store(tile_offsets_ptr + offs, tile_prefix - tile_counts, mask=mask)
    tl.store(offsets_ptr + offs + 1, route_prefix, mask=mask)
    tl.store(tile_offsets_ptr + offs + 1, tile_prefix, mask=mask)


@triton.jit
def _dsv4_mxfp4_ragged_fill_positions_kernel(
    topk_ids_ptr,
    cursor_ptr,
    offsets_ptr,
    sorted_positions_ptr,
    TOTAL: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (ids >= 0) & (ids < LOCAL_EXPERTS)
    ranks = tl.atomic_add(cursor_ptr + ids, 1, sem="relaxed", mask=valid)
    bases = tl.load(offsets_ptr + ids, mask=valid, other=0)
    tl.store(sorted_positions_ptr + bases + ranks, offs, mask=valid)


@triton.jit
def _dsv4_mxfp4_ragged_fill_positions_by_expert_kernel(
    topk_ids_ptr,
    offsets_ptr,
    sorted_positions_ptr,
    TOTAL: tl.constexpr,
    LOCAL_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    write_count = tl.full((), 0, dtype=tl.int32)
    base = tl.load(offsets_ptr + expert_id)

    for start in range(0, TOTAL, BLOCK):
        pos = start + offs
        mask = pos < TOTAL
        ids = tl.load(topk_ids_ptr + pos, mask=mask, other=-999999)
        match = (ids == expert_id) & mask
        match_i = match.to(tl.int32)
        local_rank = tl.cumsum(match_i, 0) - 1
        tl.store(
            sorted_positions_ptr + base + write_count + local_rank,
            pos,
            mask=match,
        )
        write_count += tl.sum(match_i, axis=0)


@triton.jit
def _dsv4_mxfp4_ragged_block_map_kernel(
    counts_ptr,
    tile_offsets_ptr,
    block_map_ptr,
    LOCAL_EXPERTS: tl.constexpr,
    ROUTE_BLOCK_M: tl.constexpr,
    MAX_TILES_PER_EXPERT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    expert = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    count = tl.load(counts_ptr + expert)
    tile_count = (count + ROUTE_BLOCK_M - 1) // ROUTE_BLOCK_M
    tile_base = tl.load(tile_offsets_ptr + expert)
    packed = (t << 16) + expert
    mask = (expert < LOCAL_EXPERTS) & (t < tile_count) & (t < MAX_TILES_PER_EXPERT)
    tl.store(block_map_ptr + tile_base + t, packed, mask=mask)


def _get_fixed_metadata_config() -> tuple[int, int]:
    try:
        max_active = int(os.getenv("VLLM_DSV4_MXFP4_METADATA_MAX_ACTIVE", "64"))
    except ValueError:
        max_active = 64
    try:
        capacity = int(os.getenv("VLLM_DSV4_MXFP4_METADATA_CAPACITY", "8192"))
    except ValueError:
        capacity = 8192
    return max(1, max_active), max(1, capacity)


def _dsv4_mxfp4_parse_int_filter(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            values.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            values.add(int(part))
    return values


def _dsv4_mxfp4_trace_tensor(
    name: str,
    tensor: torch.Tensor | None,
    sample_elems: int,
) -> dict[str, object]:
    if tensor is None:
        return {"name": name, "present": False}
    with torch.no_grad():
        t = tensor.detach()
        flat = t.reshape(-1)
        summary: dict[str, object] = {
            "name": name,
            "present": True,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": int(flat.numel()),
        }
        if flat.numel() == 0:
            return summary
        f = flat.float()
        finite = torch.isfinite(f)
        finite_count = int(finite.sum().item())
        summary["finite_count"] = finite_count
        if finite_count == 0:
            return summary
        finite_f = f[finite]
        sample_n = min(sample_elems, flat.numel())
        sample = f[:sample_n]
        weights = torch.arange(
            1, sample_n + 1, dtype=torch.float32, device=f.device
        )
        summary.update(
            {
                "sum": float(finite_f.sum().item()),
                "abs_sum": float(finite_f.abs().sum().item()),
                "mean": float(finite_f.mean().item()),
                "std": float(
                    finite_f.std(unbiased=False).item()
                    if finite_count > 1
                    else 0.0
                ),
                "min": float(finite_f.min().item()),
                "max": float(finite_f.max().item()),
                "sample_sum": float(sample.sum().item()),
                "sample_weighted_sum": float((sample * weights).sum().item()),
            }
        )
        return summary


def _dsv4_mxfp4_write_trace_event(event: dict[str, object]) -> None:
    trace_dir = os.getenv("VLLM_DSV4_MXFP4_TRACE_DIR", "/tmp/vllm_dsv4_mxfp4_trace")
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"mxfp4_trace_pid{os.getpid()}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _dsv4_mxfp4_write_snapshot(
    event: dict[str, object],
    tensors: dict[str, torch.Tensor | None],
) -> None:
    if os.getenv("VLLM_DSV4_MXFP4_SNAPSHOT", "0") != "1":
        return
    snapshot_dir = os.getenv(
        "VLLM_DSV4_MXFP4_SNAPSHOT_DIR",
        os.getenv("VLLM_DSV4_MXFP4_TRACE_DIR", "/tmp/vllm_dsv4_mxfp4_trace"),
    )
    os.makedirs(snapshot_dir, exist_ok=True)
    token_limit = max(1, int(os.getenv("VLLM_DSV4_MXFP4_SNAPSHOT_TOKENS", "4")))
    dim_limit = max(1, int(os.getenv("VLLM_DSV4_MXFP4_SNAPSHOT_DIMS", "64")))
    snapshot_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        with torch.no_grad():
            t = tensor.detach()
            if t.dim() >= 2:
                t = t[:token_limit, :dim_limit]
            else:
                t = t[: min(token_limit * dim_limit, t.numel())]
            snapshot_tensors[name] = t.float().cpu()
    path = os.path.join(
        snapshot_dir,
        "snapshot_pid{pid}_phase{phase}_inst{inst}_call{call}_{stage}.pt".format(
            pid=event.get("pid"),
            phase=event.get("phase") or "none",
            inst=event.get("instance_id"),
            call=event.get("phase_call"),
            stage=event.get("stage"),
        ),
    )
    torch.save({"event": event, "tensors": snapshot_tensors}, path)

# Keep this branch self-contained: the production fallback below uses quark
# dequantization plus torch matmul instead of an out-of-tree prototype kernel.
_gfx942_mxfp4_dot = None

try:
    from vllm.model_executor.layers.fused_moe.experts.routed_mxfp4_kernels import (
        reduce_route_out_expert_order_into as _routed_mxfp4_reduce_route_out_expert_order_into,
        reduce_route_out_topk_into as _routed_mxfp4_reduce_route_out_topk_into,
        routed_offsets_stage1_into as _routed_mxfp4_offsets_stage1_into,
        routed_offsets_stage2_scatter_into as _routed_mxfp4_offsets_stage2_scatter_into,
        routed_offsets_stage2_store_route_into as _routed_mxfp4_offsets_stage2_store_route_into,
        routed_ragged_stage1_into as _routed_mxfp4_ragged_stage1_into,
        routed_ragged_stage2_scatter_into as _routed_mxfp4_ragged_stage2_scatter_into,
        routed_stage1_into as _routed_mxfp4_stage1_into,
        routed_stage2_scatter_into as _routed_mxfp4_stage2_scatter_into,
    )
except Exception:
    _routed_mxfp4_stage1_into = None
    _routed_mxfp4_stage2_scatter_into = None
    _routed_mxfp4_ragged_stage1_into = None
    _routed_mxfp4_ragged_stage2_scatter_into = None
    _routed_mxfp4_offsets_stage1_into = None
    _routed_mxfp4_offsets_stage2_scatter_into = None
    _routed_mxfp4_offsets_stage2_store_route_into = None
    _routed_mxfp4_reduce_route_out_expert_order_into = None
    _routed_mxfp4_reduce_route_out_topk_into = None


def _dequant_mxfp4_to_bf16(
    quant: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    # Real DeepSeek-V4-Pro checkpoints may store UE8M0 scales as
    # torch.float8_e8m0fnu. For all dequant code below we need the raw
    # exponent byte, not the numeric float8 value.
    if hasattr(torch, "float8_e8m0fnu") and scale.dtype == torch.float8_e8m0fnu:
        scale = scale.view(torch.uint8)
    """Dequantize MXFP4 (uint8 packed E2M1) + e8m0 scale to bf16.

    Uses quark.torch.kernel.mx.dq_mxfp4 if available; otherwise falls
    back to a pure-PyTorch implementation using the FP4 E2M1 lookup table.

    Args:
        quant:  uint8 tensor with shape (*, K_packed) where K_packed = K // 2
                Each byte holds two FP4 E2M1 values: (low_nibble | high_nibble<<4)
        scale:  uint8 e8m0 tensor with shape (*, K_packed * 2 // block_size)
                Each value is a power-of-2 scale (E8M0 = 8-bit unsigned exponent,
                bias 127). One scale per 32-element block along the K axis.

    Returns:
        bf16 tensor with shape (*, K_packed * 2)
    """
    try:
        from quark.torch.kernel import mx
        return mx.dq_mxfp4(quant, scale, torch.bfloat16)
    except Exception:
        pass

    # Pure-PyTorch fallback. Slow (no JIT) but correctness-first.
    # FP4 E2M1 values, sign-magnitude:
    #   sign bit (3) | exp 2 bits (2-1) | mantissa 1 bit (0)
    # Lookup table for nibble (0..15) -> float
    _FP4_TABLE = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    table = _FP4_TABLE.to(quant.device)

    # Unpack uint8 to two FP4 nibbles
    low = quant & 0x0F
    high = (quant >> 4) & 0x0F
    # interleave: out[..., 2i] = low_i, out[..., 2i+1] = high_i
    fp4_low = table[low.long()]
    fp4_high = table[high.long()]
    out = torch.stack([fp4_low, fp4_high], dim=-1)
    out = out.view(*quant.shape[:-1], quant.shape[-1] * 2)

    # Apply e8m0 scale (1 scale per 32-elem block along last axis)
    block = 32
    K = out.shape[-1]
    assert K % block == 0
    n_blocks = K // block
    # scale: uint8 -> exp -> 2.0**(exp - 127)
    exp = scale.view(torch.uint8).long() - 127
    scale_fp = torch.pow(2.0, exp.float())
    # broadcast scale to per-element
    scale_fp = scale_fp.unsqueeze(-1).expand(*scale_fp.shape, block)
    scale_fp = scale_fp.reshape(*scale_fp.shape[:-2], n_blocks * block)
    return (out * scale_fp).to(torch.bfloat16)


class AtomTritonFP4SiluExperts(mk.FusedMoEExpertsModular):
    """Dequant-fallback MXFP4 MoE for DeepSeek V4 SILU+clamp activation.

    Lives in NON-TRITON_BACKENDS path so layer.w13_weight stays as raw uint8
    (not swizzled into triton_kernels.Tensor). Dequantizes to bf16 lazily on
    the first apply() call (cached), then runs standard PyTorch MoE compute.
    """

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # SILU only; uses gemm1_clamp_limit from quant_config for the clamp.
        return activation == MoEActivation.SILU

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # Anywhere we have CUDA-alike: this is a pure-pytorch fallback.
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Topk gating + reduce is applied inside apply() (index_add_).
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Standard convention used elsewhere in vLLM:
        #   workspace1 (-> apply's `workspace13`) = activation result buffer
        #   workspace2 (-> apply's `workspace2`)  = matmul intermediate
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M * topk, activation_out_dim)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)


    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,  # raw uint8 mxfp4 (E, 2N, K_packed) where K_packed = K // 2
        w2: torch.Tensor,  # raw uint8 mxfp4 (E, K, N_packed) where N_packed = N // 2
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """Per-call lazy dequant: dequant w1[e]/w2[e] only for experts with
        routed tokens this call, free immediately after use.

        Memory cost: at most 1 expert's bf16 weights resident at any time.
        For V4-Pro per-rank shape this is ~10-20 MiB / expert. Slow but
        memory-safe for 1.6T model on TP=8.
        """
        quant_config = self.quant_config or FUSED_MOE_UNQUANTIZED_CONFIG

        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.shape[0]
        topk = topk_ids.size(1)
        M, K = hidden_states.shape
        trace_enabled = os.getenv("VLLM_DSV4_MXFP4_TRACE", "0") == "1"
        trace_instance_id = -1
        trace_instance_call = -1
        if trace_enabled:
            trace_instance_call = getattr(self, "_dsv4_trace_apply_calls", 0) + 1

        routed_chunk_m = 0
        if os.getenv("VLLM_DSV4_MXFP4_ROUTED_KERNEL", "0") == "1":
            try:
                routed_chunk_m = int(os.getenv("VLLM_DSV4_MXFP4_ROUTED_CHUNK_M", "0"))
            except ValueError:
                routed_chunk_m = 0
        if routed_chunk_m > 0 and M > routed_chunk_m:
            if (
                os.getenv("VLLM_DSV4_MXFP4_ROUTED_CHUNK_LOG", "0") == "1"
                and not hasattr(self, "_logged_routed_chunk_m")
            ):
                logger.warning(
                    "AtomTritonFP4SiluExperts routed chunking enabled: "
                    "M=%d chunk_m=%d topk=%d local_experts=%d",
                    M,
                    routed_chunk_m,
                    topk,
                    local_num_experts,
                )
                self._logged_routed_chunk_m = True
            for start in range(0, M, routed_chunk_m):
                end = min(start + routed_chunk_m, M)
                self.apply(
                    output[start:end],
                    hidden_states[start:end],
                    w1,
                    w2,
                    topk_weights[start:end],
                    topk_ids[start:end],
                    activation,
                    global_num_experts,
                    None,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            return

        w1_scale = quant_config.w1_scale
        w2_scale = quant_config.w2_scale
        assert w1_scale is not None, "w1_scale required for mxfp4 dequant"
        assert w2_scale is not None, "w2_scale required for mxfp4 dequant"
        raw_timing_interval = os.getenv("VLLM_DSV4_MXFP4_STAGE_TIMING_INTERVAL", "0")
        try:
            stage_timing_interval = max(0, int(raw_timing_interval))
        except ValueError:
            logger.warning(
                "Invalid VLLM_DSV4_MXFP4_STAGE_TIMING_INTERVAL=%r; disabling",
                raw_timing_interval,
            )
            stage_timing_interval = 0
        timing_enabled = stage_timing_interval > 0
        if not hasattr(self, "_logged_scale_dtype"):
            logger.warning(
                "AtomTritonFP4SiluExperts scale dtypes: w1_scale=%s "
                "w2_scale=%s weight dtypes: w1=%s w2=%s gfx_kernel=%s "
                "stage_timing_interval=%d",
                getattr(w1_scale, "dtype", None), getattr(w2_scale, "dtype", None),
                getattr(w1, "dtype", None), getattr(w2, "dtype", None),
                _gfx942_mxfp4_dot is not None,
                stage_timing_interval,
            )
            self._logged_scale_dtype = True
        if timing_enabled and not hasattr(self, "_logged_stage_timing_warning"):
            logger.warning(
                "VLLM_DSV4_MXFP4_STAGE_TIMING_INTERVAL is enabled. This is a "
                "synchronizing diagnostic hook intended for no-graph profiling; "
                "leave it disabled for FULL_AND_PIECEWISE validation."
            )
            self._logged_stage_timing_warning = True

        stage_times: dict[str, float] = {}
        stage_counts: dict[str, int] = {}

        def _sync_for_timing() -> None:
            if not timing_enabled:
                return
            try:
                torch.cuda.synchronize(hidden_states.device)
            except Exception:
                torch.cuda.synchronize()

        def _stage_start() -> float:
            if not timing_enabled:
                return 0.0
            _sync_for_timing()
            return time.perf_counter()

        def _stage_end(name: str, start: float) -> None:
            if not timing_enabled:
                return
            _sync_for_timing()
            stage_times[name] = stage_times.get(name, 0.0) + (
                time.perf_counter() - start
            ) * 1000.0
            stage_counts[name] = stage_counts.get(name, 0) + 1

        clamp_limit = quant_config.gemm1_clamp_limit
        if clamp_limit is None or clamp_limit <= 0:
            clamp_limit = float("inf")
        clamp_limit = float(clamp_limit)

        t_stage = _stage_start()
        # Group tokens by expert: flatten (M, topk) -> (M*topk,) then mask per expert.
        flat_topk_ids = topk_ids.reshape(-1).to(torch.int32)
        flat_topk_w = topk_weights.reshape(-1).to(torch.float32)
        flat_token_idx = (
            torch.arange(M, device=hidden_states.device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(M, topk)
            .reshape(-1)
            .contiguous()
        )

        # Output accumulator (fp32 for accumulation precision)
        output_fp32 = torch.zeros(M, K, dtype=torch.float32, device=output.device)
        active_local_experts = -1
        if timing_enabled:
            active_mask = (flat_topk_ids >= 0) & (flat_topk_ids < local_num_experts)
            active_local_experts = int(torch.unique(flat_topk_ids[active_mask]).numel())
        _stage_end("setup", t_stage)

        if os.getenv("VLLM_DSV4_MXFP4_METADATA_COMPARE", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            # Diagnostic hook only: validates fixed-shape Triton metadata against
            # torch reference without changing production output.
            if not torch.compiler.is_compiling() and not is_capturing:
                max_active, metadata_capacity = _get_fixed_metadata_config()
                meta_shape = (local_num_experts, metadata_capacity)
                if (
                    not hasattr(self, "_metadata_positions")
                    or self._metadata_positions.shape != meta_shape
                    or self._metadata_positions.device != hidden_states.device
                ):
                    self._metadata_active_count = torch.empty(
                        (1,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._metadata_active_ids = torch.empty(
                        (max_active,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._metadata_expert_counts = torch.empty(
                        (local_num_experts,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._metadata_overflow = torch.empty(
                        (local_num_experts,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._metadata_positions = torch.empty(
                        meta_shape, dtype=torch.int32, device=hidden_states.device
                    )
                reset_block = 1024
                reset_elems = max(
                    local_num_experts * metadata_capacity,
                    local_num_experts,
                    max_active,
                )
                _dsv4_mxfp4_reset_active_metadata_kernel[
                    (triton.cdiv(reset_elems, reset_block),)
                ](
                    self._metadata_active_count,
                    self._metadata_active_ids,
                    self._metadata_expert_counts,
                    self._metadata_positions,
                    self._metadata_overflow,
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_ACTIVE=max_active,
                    CAPACITY=metadata_capacity,
                    BLOCK=reset_block,
                )
                _dsv4_mxfp4_build_active_metadata_kernel[(local_num_experts,)](
                    flat_topk_ids,
                    self._metadata_active_count,
                    self._metadata_active_ids,
                    self._metadata_expert_counts,
                    self._metadata_positions,
                    self._metadata_overflow,
                    TOTAL=flat_topk_ids.numel(),
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_ACTIVE=max_active,
                    CAPACITY=metadata_capacity,
                    BLOCK=1024,
                )
                torch_counts = torch.bincount(
                    torch.clamp(flat_topk_ids, min=0, max=local_num_experts - 1).to(torch.int64),
                    minlength=local_num_experts,
                ).to(torch.int32)
                invalid = (flat_topk_ids < 0) | (flat_topk_ids >= local_num_experts)
                if invalid.any():
                    invalid_ids = torch.clamp(
                        flat_topk_ids[invalid], min=0, max=local_num_experts - 1
                    ).to(torch.int64)
                    torch_counts.index_add_(
                        0,
                        invalid_ids,
                        -torch.ones_like(invalid_ids, dtype=torch.int32),
                    )
                count_diff = (self._metadata_expert_counts - torch_counts).abs()
                active_ref = int((torch_counts > 0).sum().item())
                active_kernel = int(self._metadata_active_count.item())
                overflow_total = int(self._metadata_overflow.sum().item())
                max_count_diff = int(count_diff.max().item()) if count_diff.numel() else 0
                logger.warning(
                    "AtomTritonFP4SiluExperts fixed metadata compare: "
                    "M=%d topk=%d local_experts=%d max_active=%d capacity=%d "
                    "active_kernel=%d active_ref=%d max_count_diff=%d "
                    "overflow_total=%d first_active_ids=%s",
                    M,
                    topk,
                    local_num_experts,
                    max_active,
                    metadata_capacity,
                    active_kernel,
                    active_ref,
                    max_count_diff,
                    overflow_total,
                    self._metadata_active_ids[: min(max_active, 16)].detach().cpu().tolist(),
                )

        # Experimental routed MXFP4 kernel path. This is the first branch that
        # consumes fixed metadata directly in Triton kernels:
        #   metadata build -> routed stage1 -> routed stage2/scatter.
        # The launch shape is fixed by local_num_experts and metadata_capacity;
        # inactive experts return inside the kernels.
        if os.getenv("VLLM_DSV4_MXFP4_ROUTED_KERNEL", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            routed_apply_call = getattr(type(self), "_routed_apply_global_calls", 0) + 1
            type(self)._routed_apply_global_calls = routed_apply_call
            try:
                routed_compare_offset = int(os.getenv("VLLM_DSV4_MXFP4_ROUTED_COMPARE_OFFSET", "0"))
            except ValueError:
                routed_compare_offset = 0
            routed_compare_requested = (
                os.getenv("VLLM_DSV4_MXFP4_ROUTED_COMPARE", "0") == "1"
                and (
                    os.getenv("VLLM_DSV4_MXFP4_FORCE_COMPARE", "0") == "1"
                    or not torch.compiler.is_compiling()
                )
                and not is_capturing
            )
            routed_compare = routed_compare_requested and routed_apply_call > max(0, routed_compare_offset)
            routed_compare_return_baseline = (
                os.getenv("VLLM_DSV4_MXFP4_COMPARE_RETURN_BASELINE", "0") == "1"
            )
            routed_debug_sync = (
                os.getenv("VLLM_DSV4_MXFP4_ROUTED_DEBUG_SYNC", "0") == "1"
                and not torch.compiler.is_compiling()
                and not is_capturing
            )
            if _routed_mxfp4_stage1_into is None or _routed_mxfp4_stage2_scatter_into is None:
                raise RuntimeError(
                    "VLLM_DSV4_MXFP4_ROUTED_KERNEL=1 but routed_mxfp4_kernels "
                    "could not be imported"
                )

            def _routed_debug_checkpoint(name: str) -> None:
                if not routed_debug_sync:
                    return
                torch.cuda.synchronize(hidden_states.device)
                logger.warning(
                    "AtomTritonFP4SiluExperts routed-debug-sync: %s M=%d K=%d topk=%d",
                    name, M, K, topk,
                )

            trace_instances = None
            trace_calls = None
            trace_sample_elems = 64
            trace_phase = os.getenv("VLLM_DSV4_MXFP4_TRACE_PHASE", "")
            trace_control_file = os.getenv("VLLM_DSV4_MXFP4_TRACE_CONTROL_FILE", "")
            if trace_enabled:
                if trace_control_file:
                    try:
                        with open(trace_control_file, encoding="utf-8") as f:
                            trace_phase = f.read().strip()
                    except FileNotFoundError:
                        trace_phase = ""
                    except Exception as exc:
                        logger.warning(
                            "Failed to read MXFP4 trace control file %s: %s",
                            trace_control_file,
                            exc,
                        )
                        trace_phase = ""
                    if trace_phase.lower() in ("", "0", "off", "false", "none"):
                        trace_enabled = False
                try:
                    trace_instances = _dsv4_mxfp4_parse_int_filter(
                        os.getenv(
                            "VLLM_DSV4_MXFP4_TRACE_LAYERS",
                            os.getenv("VLLM_DSV4_MXFP4_TRACE_INSTANCES", ""),
                        )
                    )
                    trace_calls = _dsv4_mxfp4_parse_int_filter(
                        os.getenv("VLLM_DSV4_MXFP4_TRACE_CALLS", "")
                    )
                    trace_sample_elems = max(
                        1,
                        int(os.getenv("VLLM_DSV4_MXFP4_TRACE_SAMPLE_ELEMS", "64")),
                    )
                except ValueError as exc:
                    logger.warning("Invalid MXFP4 trace filter; disabling trace: %s", exc)
                    trace_enabled = False
            trace_phase_call = trace_instance_call
            if trace_enabled and trace_phase:
                phase_counts = getattr(self, "_dsv4_trace_phase_counts", {})
                trace_phase_call = int(phase_counts.get(trace_phase, 0)) + 1
                phase_counts[trace_phase] = trace_phase_call
            if routed_compare:
                try:
                    compare_instances = _dsv4_mxfp4_parse_int_filter(
                        os.getenv("VLLM_DSV4_MXFP4_ROUTED_COMPARE_INSTANCES", "")
                    )
                    compare_calls = _dsv4_mxfp4_parse_int_filter(
                        os.getenv("VLLM_DSV4_MXFP4_ROUTED_COMPARE_CALLS", "")
                    )
                except ValueError as exc:
                    logger.warning(
                        "Invalid MXFP4 routed compare filter; disabling compare: %s",
                        exc,
                    )
                    routed_compare = False
                    compare_instances = None
                    compare_calls = None
                if (
                    routed_compare
                    and os.getenv(
                        "VLLM_DSV4_MXFP4_ROUTED_COMPARE_REQUIRE_TRACE_PHASE", "0"
                    )
                    == "1"
                    and not trace_phase
                ):
                    routed_compare = False
                if (
                    routed_compare
                    and compare_instances is not None
                    and trace_instance_id not in compare_instances
                ):
                    routed_compare = False
                if (
                    routed_compare
                    and compare_calls is not None
                    and trace_phase_call not in compare_calls
                ):
                    routed_compare = False

            def _trace_record(stage: str, tensors: dict[str, torch.Tensor | None]) -> None:
                if not trace_enabled:
                    return
                if trace_instances is not None and trace_instance_id not in trace_instances:
                    return
                if trace_calls is not None and trace_phase_call not in trace_calls:
                    return
                try:
                    event: dict[str, object] = {
                        "stage": stage,
                        "pid": os.getpid(),
                        "phase": trace_phase,
                        "instance_id": trace_instance_id,
                        "instance_call": trace_instance_call,
                        "phase_call": trace_phase_call,
                        "routed_apply_call": routed_apply_call,
                        "M": M,
                        "K": K,
                        "topk": topk,
                        "route_total": route_total,
                        "local_num_experts": local_num_experts,
                        "metadata_capacity": metadata_capacity,
                        "is_capturing": bool(is_capturing),
                        "torch_compiling": bool(torch.compiler.is_compiling()),
                        "tensors": {
                            name: _dsv4_mxfp4_trace_tensor(
                                name, tensor, trace_sample_elems
                            )
                            for name, tensor in tensors.items()
                        },
                    }
                    if hasattr(self, "_ragged_counts"):
                        counts = self._ragged_counts.detach()
                        event["active_experts"] = int((counts > 0).sum().item())
                        event["max_expert_count"] = (
                            int(counts.max().item()) if counts.numel() else 0
                        )
                    if hasattr(self, "_metadata_overflow"):
                        event["metadata_overflow_total"] = int(
                            self._metadata_overflow.detach().sum().item()
                        )
                    _dsv4_mxfp4_write_trace_event(event)
                    _dsv4_mxfp4_write_snapshot(event, tensors)
                except Exception:
                    logger.exception("Failed to write MXFP4 trace event at stage %s", stage)

            max_active, metadata_capacity = _get_fixed_metadata_config()
            meta_shape = (local_num_experts, metadata_capacity)
            if (
                not hasattr(self, "_metadata_positions")
                or self._metadata_positions.shape != meta_shape
                or self._metadata_positions.device != hidden_states.device
            ):
                self._metadata_active_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_active_ids = torch.empty((max_active,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_expert_counts = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_overflow = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_positions = torch.empty(meta_shape, dtype=torch.int32, device=hidden_states.device)

            reset_block = 1024
            reset_elems = max(local_num_experts * metadata_capacity, local_num_experts, max_active)
            _dsv4_mxfp4_reset_active_metadata_kernel[(triton.cdiv(reset_elems, reset_block),)](
                self._metadata_active_count,
                self._metadata_active_ids,
                self._metadata_expert_counts,
                self._metadata_positions,
                self._metadata_overflow,
                LOCAL_EXPERTS=local_num_experts,
                MAX_ACTIVE=max_active,
                CAPACITY=metadata_capacity,
                BLOCK=reset_block,
            )
            _dsv4_mxfp4_build_active_metadata_kernel[(local_num_experts,)](
                flat_topk_ids,
                self._metadata_active_count,
                self._metadata_active_ids,
                self._metadata_expert_counts,
                self._metadata_positions,
                self._metadata_overflow,
                TOTAL=flat_topk_ids.numel(),
                LOCAL_EXPERTS=local_num_experts,
                MAX_ACTIVE=max_active,
                CAPACITY=metadata_capacity,
                BLOCK=1024,
            )
            _routed_debug_checkpoint("metadata")

            route_total = flat_topk_ids.numel()
            gate_up_shape = (route_total, w1.shape[1])
            gate_up_numel = route_total * w1.shape[1]
            use_workspace = (
                os.getenv("VLLM_DSV4_MXFP4_ROUTED_USE_WORKSPACE", "1") == "1"
            )
            if (
                use_workspace
                and workspace2.dtype == torch.bfloat16
                and workspace2.numel() >= gate_up_numel
            ):
                routed_gate_up = workspace2.view(-1)[:gate_up_numel].view(gate_up_shape)
                routed_gate_up_from_workspace = True
            else:
                gate_up_alloc_each_call = (
                    os.getenv("VLLM_DSV4_MXFP4_GATE_UP_ALLOC_EACH_CALL", "0") == "1"
                )
                if gate_up_alloc_each_call:
                    if (
                        os.getenv(
                            "VLLM_DSV4_MXFP4_GATE_UP_ALLOC_EACH_CALL_LOG", "0"
                        )
                        == "1"
                        and not hasattr(self, "_logged_gate_up_alloc_each_call")
                    ):
                        logger.warning(
                            "AtomTritonFP4SiluExperts gate_up alloc-each-call enabled: "
                            "M=%d K=%d topk=%d route_total=%d gate_up_shape=%s",
                            M, K, topk, route_total, gate_up_shape,
                        )
                        self._logged_gate_up_alloc_each_call = True
                    routed_gate_up = torch.empty(
                        gate_up_shape, dtype=torch.bfloat16, device=hidden_states.device
                    )
                elif (
                    not hasattr(self, "_routed_gate_up_2d")
                    or self._routed_gate_up_2d.shape != gate_up_shape
                    or self._routed_gate_up_2d.device != hidden_states.device
                ):
                    self._routed_gate_up_2d = torch.empty(
                        gate_up_shape, dtype=torch.bfloat16, device=hidden_states.device
                    )
                    routed_gate_up = self._routed_gate_up_2d
                else:
                    routed_gate_up = self._routed_gate_up_2d
                routed_gate_up_from_workspace = False
            if os.getenv("VLLM_DSV4_MXFP4_GATE_UP_ZERO_BEFORE", "0") == "1":
                if (
                    os.getenv("VLLM_DSV4_MXFP4_GATE_UP_ZERO_BEFORE_LOG", "0") == "1"
                    and not hasattr(self, "_logged_gate_up_zero_before")
                ):
                    logger.warning(
                        "AtomTritonFP4SiluExperts gate_up zero-before enabled: "
                        "M=%d K=%d topk=%d route_total=%d from_workspace=%s",
                        M, K, topk, route_total, routed_gate_up_from_workspace,
                    )
                    self._logged_gate_up_zero_before = True
                routed_gate_up.zero_()

            routed_output_fp32 = (
                torch.empty_like(output_fp32) if routed_compare else output_fp32
            )

            if os.getenv("VLLM_DSV4_MXFP4_ROUTED_RAGGED", "0") == "1":
                if (
                    _routed_mxfp4_ragged_stage1_into is None
                    or _routed_mxfp4_ragged_stage2_scatter_into is None
                ):
                    raise RuntimeError(
                        "VLLM_DSV4_MXFP4_ROUTED_RAGGED=1 but ragged routed "
                        "MXFP4 kernels could not be imported"
                    )
                try:
                    ragged_block_m = int(os.getenv("VLLM_DSV4_MXFP4_RAGGED_BLOCK_M", "16"))
                except ValueError:
                    ragged_block_m = 16
                ragged_block_m = max(1, ragged_block_m)
                try:
                    stage2_store_num_warps = int(
                        os.getenv("VLLM_DSV4_MXFP4_STAGE2_STORE_NUM_WARPS", "4")
                    )
                except ValueError:
                    stage2_store_num_warps = 4
                try:
                    stage2_store_num_stages = int(
                        os.getenv("VLLM_DSV4_MXFP4_STAGE2_STORE_NUM_STAGES", "3")
                    )
                except ValueError:
                    stage2_store_num_stages = 3
                try:
                    reduce_num_warps = int(
                        os.getenv("VLLM_DSV4_MXFP4_REDUCE_NUM_WARPS", "4")
                    )
                except ValueError:
                    reduce_num_warps = 4
                try:
                    reduce_num_stages = int(
                        os.getenv("VLLM_DSV4_MXFP4_REDUCE_NUM_STAGES", "3")
                    )
                except ValueError:
                    reduce_num_stages = 3
                max_tiles = local_num_experts + triton.cdiv(route_total, ragged_block_m)
                max_tiles_per_expert = triton.cdiv(M, ragged_block_m)
                ragged_shape_ok = (
                    hasattr(self, "_ragged_sorted_positions")
                    and self._ragged_sorted_positions.shape == (route_total,)
                    and self._ragged_block_map.shape == (max_tiles,)
                    and self._ragged_counts.shape == (local_num_experts,)
                    and self._ragged_sorted_positions.device == hidden_states.device
                )
                if not ragged_shape_ok:
                    self._ragged_counts = torch.empty(
                        (local_num_experts,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._ragged_cursor = torch.empty(
                        (local_num_experts,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._ragged_offsets = torch.empty(
                        (local_num_experts + 1,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._ragged_tile_offsets = torch.empty(
                        (local_num_experts + 1,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._ragged_sorted_positions = torch.empty(
                        (route_total,), dtype=torch.int32, device=hidden_states.device
                    )
                    self._ragged_block_map = torch.empty(
                        (max_tiles,), dtype=torch.int32, device=hidden_states.device
                    )
                ragged_reset_block = 1024
                ragged_reset_elems = max(local_num_experts + 1, max_tiles)
                _dsv4_mxfp4_ragged_reset_kernel[
                    (triton.cdiv(ragged_reset_elems, ragged_reset_block),)
                ](
                    self._ragged_counts,
                    self._ragged_cursor,
                    self._ragged_offsets,
                    self._ragged_tile_offsets,
                    self._ragged_block_map,
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_TILES=max_tiles,
                    BLOCK=ragged_reset_block,
                )
                _dsv4_mxfp4_ragged_count_kernel[
                    (triton.cdiv(route_total, ragged_reset_block),)
                ](
                    flat_topk_ids,
                    self._ragged_counts,
                    TOTAL=route_total,
                    LOCAL_EXPERTS=local_num_experts,
                    BLOCK=ragged_reset_block,
                )
                prefix_block = triton.next_power_of_2(local_num_experts)
                _dsv4_mxfp4_ragged_prefix_kernel[(1,)](
                    self._ragged_counts,
                    self._ragged_offsets,
                    self._ragged_tile_offsets,
                    LOCAL_EXPERTS=local_num_experts,
                    ROUTE_BLOCK_M=ragged_block_m,
                    BLOCK_E=prefix_block,
                )
                if os.getenv("VLLM_DSV4_MXFP4_RAGGED_ATOMIC_FILL", "0") == "1":
                    _dsv4_mxfp4_ragged_fill_positions_kernel[
                        (triton.cdiv(route_total, ragged_reset_block),)
                    ](
                        flat_topk_ids,
                        self._ragged_cursor,
                        self._ragged_offsets,
                        self._ragged_sorted_positions,
                        TOTAL=route_total,
                        LOCAL_EXPERTS=local_num_experts,
                        BLOCK=ragged_reset_block,
                    )
                else:
                    _dsv4_mxfp4_ragged_fill_positions_by_expert_kernel[
                        (local_num_experts,)
                    ](
                        flat_topk_ids,
                        self._ragged_offsets,
                        self._ragged_sorted_positions,
                        TOTAL=route_total,
                        LOCAL_EXPERTS=local_num_experts,
                        BLOCK=1024,
                    )
                tile_block = triton.next_power_of_2(max(1, max_tiles_per_expert))
                _dsv4_mxfp4_ragged_block_map_kernel[(local_num_experts,)](
                    self._ragged_counts,
                    self._ragged_tile_offsets,
                    self._ragged_block_map,
                    LOCAL_EXPERTS=local_num_experts,
                    ROUTE_BLOCK_M=ragged_block_m,
                    MAX_TILES_PER_EXPERT=max_tiles_per_expert,
                    BLOCK_T=tile_block,
                )
                if (
                    os.getenv("VLLM_DSV4_MXFP4_ROUTED_RAGGED_LOG", "0") == "1"
                    and not hasattr(self, "_logged_routed_ragged")
                ):
                    logger.warning(
                        "AtomTritonFP4SiluExperts routed ragged metadata enabled: "
                        "M=%d route_total=%d block_m=%d max_tiles=%d "
                        "local_experts=%d",
                        M,
                        route_total,
                        ragged_block_m,
                        max_tiles,
                        local_num_experts,
                    )
                    self._logged_routed_ragged = True
                if os.getenv("VLLM_DSV4_MXFP4_RAGGED_OFFSET_GRID", "0") == "1":
                    if (
                        _routed_mxfp4_offsets_stage1_into is None
                        or _routed_mxfp4_offsets_stage2_scatter_into is None
                    ):
                        raise RuntimeError(
                            "VLLM_DSV4_MXFP4_RAGGED_OFFSET_GRID=1 but offset-grid "
                            "MXFP4 kernels could not be imported"
                        )
                    _routed_mxfp4_offsets_stage1_into(
                        hidden_states,
                        w1,
                        w1_scale,
                        self._ragged_counts,
                        self._ragged_offsets,
                        self._ragged_sorted_positions,
                        flat_token_idx,
                        routed_gate_up,
                        max_tiles_per_expert=max_tiles_per_expert,
                        block_m=ragged_block_m,
                    )
                    _trace_record(
                        "after_stage1",
                        {
                            "hidden_states": hidden_states,
                            "routed_gate_up": routed_gate_up,
                        },
                    )
                    if os.getenv("VLLM_DSV4_MXFP4_DETERMINISTIC_REDUCE", "0") == "1":
                        if (
                            _routed_mxfp4_offsets_stage2_store_route_into is None
                            or _routed_mxfp4_reduce_route_out_topk_into is None
                            or _routed_mxfp4_reduce_route_out_expert_order_into is None
                        ):
                            raise RuntimeError(
                                "VLLM_DSV4_MXFP4_DETERMINISTIC_REDUCE=1 but "
                                "route-store/reduce kernels could not be imported"
                            )
                        route_out_shape = (route_total, K)
                        route_out_alloc_each_call = (
                            os.getenv("VLLM_DSV4_MXFP4_ROUTE_OUT_ALLOC_EACH_CALL", "0") == "1"
                        )
                        if (
                            route_out_alloc_each_call
                            or not hasattr(self, "_routed_route_out")
                            or self._routed_route_out.shape != route_out_shape
                            or self._routed_route_out.device != hidden_states.device
                        ):
                            if (
                                route_out_alloc_each_call
                                and os.getenv("VLLM_DSV4_MXFP4_ROUTE_OUT_ALLOC_EACH_CALL_LOG", "0") == "1"
                                and not hasattr(self, "_logged_route_out_alloc_each_call")
                            ):
                                logger.warning(
                                    "AtomTritonFP4SiluExperts route_out alloc-each-call enabled: "
                                    "M=%d K=%d topk=%d route_total=%d",
                                    M, K, topk, route_total,
                                )
                                self._logged_route_out_alloc_each_call = True
                            self._routed_route_out = torch.empty(
                                route_out_shape,
                                dtype=torch.float32,
                                device=hidden_states.device,
                            )
                        if os.getenv("VLLM_DSV4_MXFP4_ROUTE_OUT_ZERO_BEFORE", "0") == "1":
                            if (
                                os.getenv("VLLM_DSV4_MXFP4_ROUTE_OUT_ZERO_BEFORE_LOG", "0") == "1"
                                and not hasattr(self, "_logged_route_out_zero_before")
                            ):
                                logger.warning(
                                    "AtomTritonFP4SiluExperts route_out zero-before enabled: "
                                    "M=%d K=%d topk=%d route_total=%d",
                                    M, K, topk, route_total,
                                )
                                self._logged_route_out_zero_before = True
                            self._routed_route_out.zero_()
                        if os.getenv("VLLM_DSV4_MXFP4_STAGE2_TORCH_FALLBACK", "0") == "1":
                            if (
                                os.getenv("VLLM_DSV4_MXFP4_STAGE2_TORCH_FALLBACK_LOG", "0") == "1"
                                and not hasattr(self, "_logged_stage2_torch_fallback")
                            ):
                                logger.warning(
                                    "AtomTritonFP4SiluExperts stage2 torch fallback enabled: "
                                    "M=%d K=%d topk=%d route_total=%d local_experts=%d",
                                    M, K, topk, route_total, local_num_experts,
                                )
                                self._logged_stage2_torch_fallback = True
                            self._routed_route_out.zero_()
                            zero_gate_up = torch.zeros(
                                (), dtype=routed_gate_up.dtype, device=routed_gate_up.device
                            )
                            zero_w_route = torch.zeros(
                                (), dtype=torch.float32, device=flat_topk_w.device
                            )
                            for e_t in range(local_num_experts):
                                mask = flat_topk_ids == e_t
                                gate_up_e = torch.where(
                                    mask.unsqueeze(1), routed_gate_up, zero_gate_up
                                )
                                gate_w_e = torch.where(mask, flat_topk_w, zero_w_route)
                                w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                                two_n_e = gate_up_e.shape[1]
                                n_e = two_n_e // 2
                                gate = gate_up_e[:, :n_e]
                                up = gate_up_e[:, n_e:]
                                if clamp_limit < float("inf"):
                                    gate = torch.clamp(gate, max=clamp_limit)
                                    up = torch.clamp(
                                        up, min=-clamp_limit, max=clamp_limit
                                    )
                                act = (
                                    F.silu(gate.to(torch.float32)) * up.to(torch.float32)
                                ).to(torch.bfloat16)
                                y_e = act @ w2_e.t()
                                self._routed_route_out.add_(
                                    y_e.to(torch.float32) * gate_w_e.unsqueeze(1)
                                )
                                del w2_e
                        else:
                            _routed_mxfp4_offsets_stage2_store_route_into(
                                routed_gate_up,
                                w2,
                                w2_scale,
                                self._ragged_counts,
                                self._ragged_offsets,
                                self._ragged_sorted_positions,
                                flat_topk_w,
                                self._routed_route_out,
                                max_tiles_per_expert=max_tiles_per_expert,
                                clamp_limit=clamp_limit,
                                block_m=ragged_block_m,
                                num_warps=stage2_store_num_warps,
                                num_stages=stage2_store_num_stages,
                            )
                        _trace_record(
                            "after_stage2_store_route",
                            {
                                "routed_gate_up": routed_gate_up,
                                "route_out": self._routed_route_out,
                            },
                        )
                        reduce_order = os.getenv(
                            "VLLM_DSV4_MXFP4_DETERMINISTIC_REDUCE_ORDER",
                            "topk",
                        ).lower()
                        if os.getenv("VLLM_DSV4_MXFP4_REDUCE_TORCH_FALLBACK", "0") == "1":
                            if (
                                os.getenv("VLLM_DSV4_MXFP4_REDUCE_TORCH_FALLBACK_LOG", "0") == "1"
                                and not hasattr(self, "_logged_reduce_torch_fallback")
                            ):
                                logger.warning(
                                    "AtomTritonFP4SiluExperts reduce torch fallback enabled: "
                                    "M=%d K=%d topk=%d route_total=%d order=%s",
                                    M, K, topk, route_total, reduce_order,
                                )
                                self._logged_reduce_torch_fallback = True
                            routed_output_fp32.zero_()
                            routed_output_fp32.index_add_(
                                0,
                                flat_token_idx.to(torch.int64),
                                self._routed_route_out,
                            )
                        elif reduce_order == "expert":
                            _routed_mxfp4_reduce_route_out_expert_order_into(
                                self._routed_route_out,
                                flat_topk_ids,
                                routed_output_fp32,
                                M=M,
                                topk=topk,
                                num_warps=reduce_num_warps,
                                num_stages=reduce_num_stages,
                            )
                        else:
                            _routed_mxfp4_reduce_route_out_topk_into(
                                self._routed_route_out,
                                routed_output_fp32,
                                M=M,
                                topk=topk,
                                num_warps=reduce_num_warps,
                                num_stages=reduce_num_stages,
                            )
                        _trace_record(
                            "after_reduce",
                            {
                                "route_out": self._routed_route_out,
                                "routed_output_fp32": routed_output_fp32,
                            },
                        )
                    else:
                        _routed_mxfp4_offsets_stage2_scatter_into(
                            routed_gate_up,
                            w2,
                            w2_scale,
                            self._ragged_counts,
                            self._ragged_offsets,
                            self._ragged_sorted_positions,
                            flat_token_idx,
                            flat_topk_w,
                            routed_output_fp32,
                            max_tiles_per_expert=max_tiles_per_expert,
                            clamp_limit=clamp_limit,
                            block_m=ragged_block_m,
                        )
                        _trace_record(
                            "after_stage2_scatter",
                            {
                                "routed_gate_up": routed_gate_up,
                                "routed_output_fp32": routed_output_fp32,
                            },
                        )
                else:
                    _routed_mxfp4_ragged_stage1_into(
                        hidden_states,
                        w1,
                        w1_scale,
                        self._ragged_counts,
                        self._ragged_offsets,
                        self._ragged_sorted_positions,
                        self._ragged_block_map,
                        flat_token_idx,
                        routed_gate_up,
                        block_m=ragged_block_m,
                    )
                    _trace_record(
                        "after_stage1",
                        {
                            "hidden_states": hidden_states,
                            "routed_gate_up": routed_gate_up,
                        },
                    )
                    _routed_mxfp4_ragged_stage2_scatter_into(
                        routed_gate_up,
                        w2,
                        w2_scale,
                        self._ragged_counts,
                        self._ragged_offsets,
                        self._ragged_sorted_positions,
                        self._ragged_block_map,
                        flat_token_idx,
                        flat_topk_w,
                        routed_output_fp32,
                        clamp_limit=clamp_limit,
                        block_m=ragged_block_m,
                    )
                    _trace_record(
                        "after_stage2_scatter",
                        {
                            "routed_gate_up": routed_gate_up,
                            "routed_output_fp32": routed_output_fp32,
                        },
                    )
                if routed_compare:
                    output_fp32.zero_()
                    for e_t in range(local_num_experts):
                        count = int(self._ragged_counts[e_t].item())
                        if count <= 0:
                            continue
                        base = int(self._ragged_offsets[e_t].item())
                        routed_pos = self._ragged_sorted_positions[
                            base : base + count
                        ].to(torch.int64)
                        tokens_for_e = flat_token_idx[routed_pos].to(torch.int64)
                        gate_w_for_e = flat_topk_w[routed_pos]
                        x_e = hidden_states[tokens_for_e].to(torch.bfloat16)
                        w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                        w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                        two_N = w1_e.shape[0]
                        N = two_N // 2
                        gate_up = x_e @ w1_e.t()
                        gate = gate_up[:, :N]
                        up = gate_up[:, N:]
                        if clamp_limit < float("inf"):
                            gate = torch.clamp(gate, max=clamp_limit)
                            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                        act = (
                            F.silu(gate.to(torch.float32)) * up.to(torch.float32)
                        ).to(torch.bfloat16)
                        y_e = act @ w2_e.t()
                        output_fp32.index_add_(
                            0,
                            tokens_for_e,
                            y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1),
                        )
                        del w1_e, w2_e

                    compare_call = getattr(type(self), "_ragged_compare_global_calls", 0) + 1
                    type(self)._ragged_compare_global_calls = compare_call
                    try:
                        compare_limit = int(os.getenv("VLLM_DSV4_MXFP4_ROUTED_COMPARE_LIMIT", "16"))
                    except ValueError:
                        compare_limit = 16
                    if compare_call <= max(0, compare_limit):
                        diff = (routed_output_fp32 - output_fp32).abs()
                        denom = torch.maximum(
                            routed_output_fp32.abs(), output_fp32.abs()
                        ).clamp_min(1e-6)
                        routed_flat = routed_output_fp32.flatten().float()
                        base_flat = output_fp32.flatten().float()
                        cos = (
                            F.cosine_similarity(routed_flat, base_flat, dim=0).item()
                            if routed_flat.numel() > 0
                            else 1.0
                        )
                        logger.warning(
                            "AtomTritonFP4SiluExperts ragged-compare: "
                            "call=%d apply_call=%d M=%d K=%d topk=%d local_experts=%d "
                            "route_total=%d block_m=%d max_abs=%.6g "
                            "mean_abs=%.6g max_rel=%.6g mean_rel=%.6g cos=%.8f",
                            compare_call,
                            routed_apply_call,
                            M,
                            K,
                            topk,
                            local_num_experts,
                            route_total,
                            ragged_block_m,
                            diff.max().item() if diff.numel() else 0.0,
                            diff.mean().item() if diff.numel() else 0.0,
                            (diff / denom).max().item() if diff.numel() else 0.0,
                            (diff / denom).mean().item() if diff.numel() else 0.0,
                            cos,
                        )
                output.copy_(
                    output_fp32
                    if routed_compare and routed_compare_return_baseline
                    else routed_output_fp32
                )
                _trace_record(
                    "after_output_copy",
                    {
                        "routed_output_fp32": routed_output_fp32,
                        "output": output,
                    },
                )
                return

            if os.getenv("VLLM_DSV4_MXFP4_ROUTED_PAGED", "0") == "1":
                try:
                    routed_pages = int(os.getenv("VLLM_DSV4_MXFP4_ROUTED_PAGES", "0"))
                except ValueError:
                    routed_pages = 0
                if routed_pages <= 0:
                    # An expert can receive at most one route per token when
                    # top-k ids are unique, so ceil(M / CAPACITY) pages covers
                    # the worst case without inflating the static metadata table.
                    routed_pages = triton.cdiv(M, metadata_capacity)
                if (
                    os.getenv("VLLM_DSV4_MXFP4_ROUTED_PAGED_LOG", "0") == "1"
                    and not hasattr(self, "_logged_routed_paged")
                ):
                    logger.warning(
                        "AtomTritonFP4SiluExperts routed paged metadata enabled: "
                        "M=%d pages=%d capacity=%d topk=%d local_experts=%d",
                        M,
                        routed_pages,
                        metadata_capacity,
                        topk,
                        local_num_experts,
                    )
                    self._logged_routed_paged = True
                for page_idx in range(routed_pages):
                    page_offset = page_idx * metadata_capacity
                    _dsv4_mxfp4_reset_active_metadata_kernel[
                        (triton.cdiv(reset_elems, reset_block),)
                    ](
                        self._metadata_active_count,
                        self._metadata_active_ids,
                        self._metadata_expert_counts,
                        self._metadata_positions,
                        self._metadata_overflow,
                        LOCAL_EXPERTS=local_num_experts,
                        MAX_ACTIVE=max_active,
                        CAPACITY=metadata_capacity,
                        BLOCK=reset_block,
                    )
                    _dsv4_mxfp4_build_active_metadata_page_kernel[(local_num_experts,)](
                        flat_topk_ids,
                        self._metadata_active_count,
                        self._metadata_active_ids,
                        self._metadata_expert_counts,
                        self._metadata_positions,
                        self._metadata_overflow,
                        TOTAL=flat_topk_ids.numel(),
                        LOCAL_EXPERTS=local_num_experts,
                        MAX_ACTIVE=max_active,
                        CAPACITY=metadata_capacity,
                        PAGE_OFFSET=page_offset,
                        BLOCK=1024,
                    )
                    _routed_mxfp4_stage1_into(
                        hidden_states,
                        w1,
                        w1_scale,
                        self._metadata_expert_counts,
                        self._metadata_positions,
                        flat_token_idx,
                        routed_gate_up,
                    )
                    _routed_mxfp4_stage2_scatter_into(
                        routed_gate_up,
                        w2,
                        w2_scale,
                        self._metadata_expert_counts,
                        self._metadata_positions,
                        flat_token_idx,
                        flat_topk_w,
                        output_fp32,
                        clamp_limit=clamp_limit,
                    )
                output.copy_(output_fp32)
                return

            _routed_mxfp4_stage1_into(
                hidden_states,
                w1,
                w1_scale,
                self._metadata_expert_counts,
                self._metadata_positions,
                flat_token_idx,
                routed_gate_up,
            )
            _routed_debug_checkpoint("stage1")
            _routed_mxfp4_stage2_scatter_into(
                routed_gate_up,
                w2,
                w2_scale,
                self._metadata_expert_counts,
                self._metadata_positions,
                flat_token_idx,
                flat_topk_w,
                routed_output_fp32,
                clamp_limit=clamp_limit,
            )
            _routed_debug_checkpoint("stage2")

            if routed_compare:
                output_fp32.zero_()
                overflow_total = int(self._metadata_overflow.sum().item())
                max_count = int(self._metadata_expert_counts.max().item()) if self._metadata_expert_counts.numel() else 0
                for e_t in range(local_num_experts):
                    count = int(self._metadata_expert_counts[e_t].item())
                    if count <= 0:
                        continue
                    n_e = min(count, metadata_capacity)
                    routed_pos = self._metadata_positions[e_t, :n_e].to(torch.int64)
                    tokens_for_e = flat_token_idx[routed_pos].to(torch.int64)
                    gate_w_for_e = flat_topk_w[routed_pos]
                    x_e = hidden_states[tokens_for_e].to(torch.bfloat16)
                    w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                    w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                    two_N = w1_e.shape[0]
                    N = two_N // 2
                    gate_up = x_e @ w1_e.t()
                    gate = gate_up[:, :N]
                    up = gate_up[:, N:]
                    if clamp_limit < float("inf"):
                        gate = torch.clamp(gate, max=clamp_limit)
                        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                    act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
                    y_e = act @ w2_e.t()
                    output_fp32.index_add_(
                        0,
                        tokens_for_e,
                        y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1),
                    )
                    del w1_e, w2_e
                _routed_debug_checkpoint("baseline")

                compare_call = getattr(type(self), "_routed_compare_global_calls", 0) + 1
                type(self)._routed_compare_global_calls = compare_call
                try:
                    compare_limit = int(os.getenv("VLLM_DSV4_MXFP4_ROUTED_COMPARE_LIMIT", "16"))
                except ValueError:
                    compare_limit = 16
                if compare_call <= max(0, compare_limit):
                    diff = (routed_output_fp32 - output_fp32).abs()
                    denom = torch.maximum(
                        routed_output_fp32.abs(), output_fp32.abs()
                    ).clamp_min(1e-6)
                    routed_flat = routed_output_fp32.flatten().float()
                    base_flat = output_fp32.flatten().float()
                    finite_routed = bool(torch.isfinite(routed_output_fp32).all().item())
                    finite_base = bool(torch.isfinite(output_fp32).all().item())
                    cos = (
                        F.cosine_similarity(routed_flat, base_flat, dim=0).item()
                        if routed_flat.numel() > 0
                        else 1.0
                    )
                    logger.warning(
                        "AtomTritonFP4SiluExperts routed-compare: "
                        "call=%d apply_call=%d M=%d K=%d topk=%d local_experts=%d "
                        "capacity=%d active=%d overflow=%d max_count=%d "
                        "workspace=%s finite_routed=%s finite_base=%s "
                        "max_abs=%.6g mean_abs=%.6g max_rel=%.6g mean_rel=%.6g "
                        "cos=%.8f routed_std=%.6g base_std=%.6g",
                        compare_call,
                        routed_apply_call,
                        M,
                        K,
                        topk,
                        local_num_experts,
                        metadata_capacity,
                        int((self._metadata_expert_counts > 0).sum().item()),
                        overflow_total,
                        max_count,
                        routed_gate_up_from_workspace,
                        finite_routed,
                        finite_base,
                        diff.max().item() if diff.numel() else 0.0,
                        diff.mean().item() if diff.numel() else 0.0,
                        (diff / denom).max().item() if diff.numel() else 0.0,
                        (diff / denom).mean().item() if diff.numel() else 0.0,
                        cos,
                        routed_output_fp32.std().item() if routed_output_fp32.numel() else 0.0,
                        output_fp32.std().item() if output_fp32.numel() else 0.0,
                    )

            output.copy_(
                output_fp32
                if routed_compare and routed_compare_return_baseline
                else routed_output_fp32
            )

            if os.getenv("VLLM_DSV4_MXFP4_ROUTED_KERNEL_LOG", "0") == "1":
                try:
                    is_capturing = torch.cuda.is_current_stream_capturing()
                except Exception:
                    is_capturing = False
                if not torch.compiler.is_compiling() and not is_capturing:
                    logger.warning(
                        "AtomTritonFP4SiluExperts routed-kernel compute: "
                        "M=%d K=%d topk=%d local_experts=%d capacity=%d active=%d overflow=%d",
                        M, K, topk, local_num_experts, metadata_capacity,
                        int((self._metadata_expert_counts > 0).sum().item()),
                        int(self._metadata_overflow.sum().item()),
                    )
            return

        # Optional diagnostic: deterministic expert-id loop driven by fixed-shape
        # metadata counts. It avoids torch.unique and keeps padded GEMM shape for
        # active experts. Still no-graph only because the Python `if count <= 0`
        # and int(.item()) are not capture-safe; this tests the next step toward
        # graph-safe active-padded compute.
        # Optional graph-shape prototype: fixed-capacity compute driven by
        # metadata positions. It avoids Python count/item and dynamic slicing by
        # running every expert with a fixed CAPACITY rows. Invalid rows map to
        # token 0 with gate 0. This is potentially capture-safe, but still slow
        # if CAPACITY is large. Use for FULL_AND_PIECEWISE capture experiments.
        # Experimental graph-safe candidate: tensor-predicate torch.cond.
        # It keeps padded [M*topk, K] compute for active experts but avoids
        # Python .item()/branch. If torch.cond lowers cleanly, inactive expert
        # GEMMs may be skipped while remaining visible as static control flow.
        if os.getenv("VLLM_DSV4_MXFP4_TORCH_COND_PADDED", "0") == "1":
            max_active, metadata_capacity = _get_fixed_metadata_config()
            meta_shape = (local_num_experts, metadata_capacity)
            if (
                not hasattr(self, "_metadata_positions")
                or self._metadata_positions.shape != meta_shape
                or self._metadata_positions.device != hidden_states.device
            ):
                self._metadata_active_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_active_ids = torch.empty((max_active,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_expert_counts = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_overflow = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_positions = torch.empty(meta_shape, dtype=torch.int32, device=hidden_states.device)
            reset_block = 1024
            reset_elems = max(local_num_experts * metadata_capacity, local_num_experts, max_active)
            _dsv4_mxfp4_reset_active_metadata_kernel[(triton.cdiv(reset_elems, reset_block),)](
                self._metadata_active_count,
                self._metadata_active_ids,
                self._metadata_expert_counts,
                self._metadata_positions,
                self._metadata_overflow,
                LOCAL_EXPERTS=local_num_experts,
                MAX_ACTIVE=max_active,
                CAPACITY=metadata_capacity,
                BLOCK=reset_block,
            )
            _dsv4_mxfp4_build_active_metadata_kernel[(local_num_experts,)](
                flat_topk_ids,
                self._metadata_active_count,
                self._metadata_active_ids,
                self._metadata_expert_counts,
                self._metadata_positions,
                self._metadata_overflow,
                TOTAL=flat_topk_ids.numel(),
                LOCAL_EXPERTS=local_num_experts,
                MAX_ACTIVE=max_active,
                CAPACITY=metadata_capacity,
                BLOCK=1024,
            )
            zero_long_cond = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
            zero_w_cond = torch.zeros((), dtype=torch.float32, device=hidden_states.device)

            def _inactive_branch(x_e, w1_e, w2_e, gate_w_masked):
                return torch.zeros((x_e.shape[0], K), dtype=torch.float32, device=x_e.device)

            def _active_branch(x_e, w1_e, w2_e, gate_w_masked):
                two_N = w1_e.shape[0]
                N = two_N // 2
                gate_up = x_e @ w1_e.t()
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
                y_e = act @ w2_e.t()
                return y_e.to(torch.float32) * gate_w_masked.unsqueeze(1)

            for e_t in range(local_num_experts):
                mask = flat_topk_ids == e_t
                safe_token_idx = torch.where(mask, flat_token_idx.to(torch.int64), zero_long_cond)
                gate_w_masked = torch.where(mask, flat_topk_w, zero_w_cond)
                x_e = hidden_states[safe_token_idx].to(torch.bfloat16)
                w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                pred = self._metadata_expert_counts[e_t] > 0
                contrib = torch.cond(pred, _active_branch, _inactive_branch, (x_e, w1_e, w2_e, gate_w_masked))
                output_fp32.index_add_(0, safe_token_idx, contrib)
                del w1_e, w2_e
            output.copy_(output_fp32.to(output.dtype))
            if os.getenv("VLLM_DSV4_MXFP4_TORCH_COND_PADDED_LOG", "0") == "1":
                logger.warning(
                    "AtomTritonFP4SiluExperts torch-cond-padded compute: M=%d K=%d topk=%d local_experts=%d",
                    M, K, topk, local_num_experts,
                )
            return

        if os.getenv("VLLM_DSV4_MXFP4_FIXED_CAPACITY_PADDED", "0") == "1":
            max_active, metadata_capacity = _get_fixed_metadata_config()
            meta_shape = (local_num_experts, metadata_capacity)
            if (
                not hasattr(self, "_metadata_positions")
                or self._metadata_positions.shape != meta_shape
                or self._metadata_positions.device != hidden_states.device
            ):
                self._metadata_active_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_active_ids = torch.empty((max_active,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_expert_counts = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_overflow = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                self._metadata_positions = torch.empty(meta_shape, dtype=torch.int32, device=hidden_states.device)
            reset_block = 1024
            reset_elems = max(local_num_experts * metadata_capacity, local_num_experts, max_active)
            _dsv4_mxfp4_reset_active_metadata_kernel[(triton.cdiv(reset_elems, reset_block),)](
                self._metadata_active_count,
                self._metadata_active_ids,
                self._metadata_expert_counts,
                self._metadata_positions,
                self._metadata_overflow,
                LOCAL_EXPERTS=local_num_experts,
                MAX_ACTIVE=max_active,
                CAPACITY=metadata_capacity,
                BLOCK=reset_block,
            )
            _dsv4_mxfp4_build_active_metadata_kernel[(local_num_experts,)](
                flat_topk_ids,
                self._metadata_active_count,
                self._metadata_active_ids,
                self._metadata_expert_counts,
                self._metadata_positions,
                self._metadata_overflow,
                TOTAL=flat_topk_ids.numel(),
                LOCAL_EXPERTS=local_num_experts,
                MAX_ACTIVE=max_active,
                CAPACITY=metadata_capacity,
                BLOCK=1024,
            )
            cap_range = torch.arange(metadata_capacity, device=hidden_states.device, dtype=torch.int32)
            zero_long_c = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
            zero_w_c = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
            for e_t in range(local_num_experts):
                pos_i32 = self._metadata_positions[e_t]
                valid = cap_range < self._metadata_expert_counts[e_t]
                safe_pos = torch.where(valid, pos_i32, torch.zeros_like(pos_i32))
                tokens_for_e = flat_token_idx[safe_pos].to(torch.int64)
                gate_w_for_e = torch.where(valid, flat_topk_w[safe_pos], zero_w_c)
                tokens_for_e = torch.where(valid, tokens_for_e, zero_long_c)
                x_e = hidden_states[tokens_for_e].to(torch.bfloat16)
                w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                two_N = w1_e.shape[0]
                N = two_N // 2
                gate_up = x_e @ w1_e.t()
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
                y_e = act @ w2_e.t()
                output_fp32.index_add_(0, tokens_for_e, y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1))
                del w1_e, w2_e
            output.copy_(output_fp32.to(output.dtype))
            if os.getenv("VLLM_DSV4_MXFP4_FIXED_CAPACITY_PADDED_LOG", "0") == "1":
                logger.warning(
                    "AtomTritonFP4SiluExperts fixed-capacity-padded compute: M=%d K=%d topk=%d local_experts=%d capacity=%d",
                    M, K, topk, local_num_experts, metadata_capacity,
                )
            return

        if os.getenv("VLLM_DSV4_MXFP4_FIXED_ORDER_PADDED", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            allow_capture_probe = os.getenv("VLLM_DSV4_MXFP4_FIXED_ORDER_ALLOW_CAPTURE", "0") == "1"
            if allow_capture_probe or (not torch.compiler.is_compiling() and not is_capturing):
                max_active, metadata_capacity = _get_fixed_metadata_config()
                meta_shape = (local_num_experts, metadata_capacity)
                if (
                    not hasattr(self, "_metadata_positions")
                    or self._metadata_positions.shape != meta_shape
                    or self._metadata_positions.device != hidden_states.device
                ):
                    self._metadata_active_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_active_ids = torch.empty((max_active,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_expert_counts = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_overflow = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_positions = torch.empty(meta_shape, dtype=torch.int32, device=hidden_states.device)
                reset_block = 1024
                reset_elems = max(local_num_experts * metadata_capacity, local_num_experts, max_active)
                _dsv4_mxfp4_reset_active_metadata_kernel[(triton.cdiv(reset_elems, reset_block),)](
                    self._metadata_active_count,
                    self._metadata_active_ids,
                    self._metadata_expert_counts,
                    self._metadata_positions,
                    self._metadata_overflow,
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_ACTIVE=max_active,
                    CAPACITY=metadata_capacity,
                    BLOCK=reset_block,
                )
                _dsv4_mxfp4_build_active_metadata_kernel[(local_num_experts,)](
                    flat_topk_ids,
                    self._metadata_active_count,
                    self._metadata_active_ids,
                    self._metadata_expert_counts,
                    self._metadata_positions,
                    self._metadata_overflow,
                    TOTAL=flat_topk_ids.numel(),
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_ACTIVE=max_active,
                    CAPACITY=metadata_capacity,
                    BLOCK=1024,
                )
                active_count_py = 0
                overflow_total = int(self._metadata_overflow.sum().item())
                zero_long_p = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
                zero_w_p = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
                for e_t in range(local_num_experts):
                    count = int(self._metadata_expert_counts[e_t].item())
                    if count <= 0:
                        continue
                    active_count_py += 1
                    mask = flat_topk_ids == e_t
                    safe_token_idx = torch.where(mask, flat_token_idx.to(torch.int64), zero_long_p)
                    gate_w_masked = torch.where(mask, flat_topk_w, zero_w_p)
                    x_e = hidden_states[safe_token_idx].to(torch.bfloat16)
                    w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                    w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                    two_N = w1_e.shape[0]
                    N = two_N // 2
                    gate_up = x_e @ w1_e.t()
                    gate = gate_up[:, :N]
                    up = gate_up[:, N:]
                    if clamp_limit < float("inf"):
                        gate = torch.clamp(gate, max=clamp_limit)
                        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                    act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
                    y_e = act @ w2_e.t()
                    output_fp32.index_add_(0, safe_token_idx, y_e.to(torch.float32) * gate_w_masked.unsqueeze(1))
                    del w1_e, w2_e
                output.copy_(output_fp32.to(output.dtype))
                if os.getenv("VLLM_DSV4_MXFP4_FIXED_ORDER_PADDED_LOG", "0") == "1":
                    logger.warning(
                        "AtomTritonFP4SiluExperts fixed-order-padded compute: M=%d K=%d topk=%d local_experts=%d active=%d overflow=%d",
                        M, K, topk, local_num_experts, active_count_py, overflow_total,
                    )
                return

        # Optional diagnostic: skip inactive experts, but keep the exact
        # all-expert padded GEMM shape for active experts. This preserves the
        # large [M*topk, K] matmul numerics better than compact active-only
        # [n_e, K], while avoiding most inactive expert work. No-graph only.
        if os.getenv("VLLM_DSV4_MXFP4_ACTIVE_PADDED", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            if not torch.compiler.is_compiling() and not is_capturing:
                valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < local_num_experts)
                used_experts = torch.unique(flat_topk_ids[valid_mask])
                zero_long_p = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
                zero_w_p = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
                for e_t_tensor in used_experts:
                    e_t = int(e_t_tensor.item())
                    mask = flat_topk_ids == e_t
                    safe_token_idx = torch.where(mask, flat_token_idx.to(torch.int64), zero_long_p)
                    gate_w_masked = torch.where(mask, flat_topk_w, zero_w_p)
                    x_e = hidden_states[safe_token_idx].to(torch.bfloat16)
                    w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                    w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                    two_N = w1_e.shape[0]
                    N = two_N // 2
                    gate_up = x_e @ w1_e.t()
                    gate = gate_up[:, :N]
                    up = gate_up[:, N:]
                    if clamp_limit < float("inf"):
                        gate = torch.clamp(gate, max=clamp_limit)
                        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                    act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
                    y_e = act @ w2_e.t()
                    output_fp32.index_add_(0, safe_token_idx, y_e.to(torch.float32) * gate_w_masked.unsqueeze(1))
                    del w1_e, w2_e
                output.copy_(output_fp32.to(output.dtype))
                if os.getenv("VLLM_DSV4_MXFP4_ACTIVE_PADDED_LOG", "0") == "1":
                    logger.warning(
                        "AtomTritonFP4SiluExperts active-padded compute: M=%d K=%d topk=%d local_experts=%d active=%d",
                        M, K, topk, local_num_experts, int(used_experts.numel()),
                    )
                return

        # Optional diagnostic path using fixed-shape metadata but deterministic
        # expert-id order. It remains no-graph only because it slices by runtime
        # expert_counts; use it to test numerical/order behavior before writing
        # graph-safe fixed-capacity compute kernels.
        if os.getenv("VLLM_DSV4_MXFP4_METADATA_COMPUTE", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            if not torch.compiler.is_compiling() and not is_capturing:
                max_active, metadata_capacity = _get_fixed_metadata_config()
                meta_shape = (local_num_experts, metadata_capacity)
                if (
                    not hasattr(self, "_metadata_positions")
                    or self._metadata_positions.shape != meta_shape
                    or self._metadata_positions.device != hidden_states.device
                ):
                    self._metadata_active_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_active_ids = torch.empty((max_active,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_expert_counts = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_overflow = torch.empty((local_num_experts,), dtype=torch.int32, device=hidden_states.device)
                    self._metadata_positions = torch.empty(meta_shape, dtype=torch.int32, device=hidden_states.device)
                reset_block = 1024
                reset_elems = max(local_num_experts * metadata_capacity, local_num_experts, max_active)
                _dsv4_mxfp4_reset_active_metadata_kernel[(triton.cdiv(reset_elems, reset_block),)](
                    self._metadata_active_count,
                    self._metadata_active_ids,
                    self._metadata_expert_counts,
                    self._metadata_positions,
                    self._metadata_overflow,
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_ACTIVE=max_active,
                    CAPACITY=metadata_capacity,
                    BLOCK=reset_block,
                )
                _dsv4_mxfp4_build_active_metadata_kernel[(local_num_experts,)](
                    flat_topk_ids,
                    self._metadata_active_count,
                    self._metadata_active_ids,
                    self._metadata_expert_counts,
                    self._metadata_positions,
                    self._metadata_overflow,
                    TOTAL=flat_topk_ids.numel(),
                    LOCAL_EXPERTS=local_num_experts,
                    MAX_ACTIVE=max_active,
                    CAPACITY=metadata_capacity,
                    BLOCK=1024,
                )
                overflow_total = int(self._metadata_overflow.sum().item())
                if overflow_total != 0:
                    logger.warning(
                        "AtomTritonFP4SiluExperts metadata compute overflow: total=%d",
                        overflow_total,
                    )
                for e_t in range(local_num_experts):
                    count = int(self._metadata_expert_counts[e_t].item())
                    if count <= 0:
                        continue
                    n_e = min(count, metadata_capacity)
                    routed_pos = self._metadata_positions[e_t, :n_e].to(torch.int64)
                    tokens_for_e = flat_token_idx[routed_pos].to(torch.int64)
                    gate_w_for_e = flat_topk_w[routed_pos]
                    x_e = hidden_states[tokens_for_e].to(torch.bfloat16)
                    w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                    w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                    two_N = w1_e.shape[0]
                    N = two_N // 2
                    gate_up = x_e @ w1_e.t()
                    gate = gate_up[:, :N]
                    up = gate_up[:, N:]
                    if clamp_limit < float("inf"):
                        gate = torch.clamp(gate, max=clamp_limit)
                        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                    act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
                    y_e = act @ w2_e.t()
                    output_fp32.index_add_(0, tokens_for_e, y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1))
                    del w1_e, w2_e
                if os.getenv("VLLM_DSV4_MXFP4_METADATA_COMPUTE_COMPARE", "0") == "1":
                    compare_all = torch.zeros_like(output_fp32)
                    zero_long_c = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
                    zero_w_c = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
                    for all_e_t in range(local_num_experts):
                        all_mask = flat_topk_ids == all_e_t
                        all_safe_token_idx = torch.where(
                            all_mask, flat_token_idx.to(torch.int64), zero_long_c
                        )
                        all_gate_w = torch.where(all_mask, flat_topk_w, zero_w_c)
                        all_x = hidden_states[all_safe_token_idx].to(torch.bfloat16)
                        all_w1 = _dequant_mxfp4_to_bf16(w1[all_e_t], w1_scale[all_e_t])
                        all_w2 = _dequant_mxfp4_to_bf16(w2[all_e_t], w2_scale[all_e_t])
                        all_two_n = all_w1.shape[0]
                        all_n = all_two_n // 2
                        all_gate_up = all_x @ all_w1.t()
                        all_gate = all_gate_up[:, :all_n]
                        all_up = all_gate_up[:, all_n:]
                        if clamp_limit < float("inf"):
                            all_gate = torch.clamp(all_gate, max=clamp_limit)
                            all_up = torch.clamp(all_up, min=-clamp_limit, max=clamp_limit)
                        all_act = (F.silu(all_gate.to(torch.float32)) * all_up.to(torch.float32)).to(torch.bfloat16)
                        all_y = all_act @ all_w2.t()
                        compare_all.index_add_(
                            0,
                            all_safe_token_idx,
                            all_y.to(torch.float32) * all_gate_w.unsqueeze(1),
                        )
                        del all_w1, all_w2
                    diff = (output_fp32 - compare_all).abs()
                    denom = torch.clamp(compare_all.abs(), min=1e-8)
                    rel = diff / denom
                    logger.warning(
                        "AtomTritonFP4SiluExperts metadata-compute compare: "
                        "M=%d K=%d topk=%d local_experts=%d "
                        "max_abs=%.6g mean_abs=%.6g max_rel=%.6g mean_rel=%.6g "
                        "metadata_std=%.6g all_std=%.6g",
                        M, K, topk, local_num_experts,
                        diff.max().item() if diff.numel() else 0.0,
                        diff.mean().item() if diff.numel() else 0.0,
                        rel.max().item() if rel.numel() else 0.0,
                        rel.mean().item() if rel.numel() else 0.0,
                        output_fp32.std().item() if output_fp32.numel() else 0.0,
                        compare_all.std().item() if compare_all.numel() else 0.0,
                    )
                output.copy_(output_fp32.to(output.dtype))
                if os.getenv("VLLM_DSV4_MXFP4_METADATA_COMPUTE_LOG", "0") == "1":
                    logger.warning(
                        "AtomTritonFP4SiluExperts metadata compute: M=%d K=%d topk=%d local_experts=%d active_ref=%d overflow=%d",
                        M, K, topk, local_num_experts,
                        int((self._metadata_expert_counts > 0).sum().item()),
                        overflow_total,
                    )
                return

        # Optional diagnostic fast path: dynamic active-expert loop for no-graph
        # experiments only. This is intentionally disabled by default because
        # torch.unique/nonzero and dynamic shapes are not HIP graph capture safe.
        if os.getenv("VLLM_DSV4_MXFP4_ACTIVE_ONLY", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            if not torch.compiler.is_compiling() and not is_capturing:
                t_stage = _stage_start()
                valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < local_num_experts)
                used_experts = torch.unique(flat_topk_ids[valid_mask])
                _stage_end("active_grouping", t_stage)
                for e_t_tensor in used_experts:
                    e_t = int(e_t_tensor.item())
                    t_stage = _stage_start()
                    mask = flat_topk_ids == e_t
                    tokens_for_e = flat_token_idx[mask].to(torch.int64)
                    gate_w_for_e = flat_topk_w[mask]
                    x_e = hidden_states[tokens_for_e].to(torch.bfloat16)
                    _stage_end("active_expert_select", t_stage)

                    t_stage = _stage_start()
                    w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                    w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                    _stage_end("active_dequant", t_stage)
                    two_N = w1_e.shape[0]
                    N = two_N // 2

                    t_stage = _stage_start()
                    gate_up = x_e @ w1_e.t()
                    _stage_end("active_w1_matmul", t_stage)
                    gate = gate_up[:, :N]
                    up = gate_up[:, N:]

                    t_stage = _stage_start()
                    if clamp_limit < float("inf"):
                        gate = torch.clamp(gate, max=clamp_limit)
                        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                    act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                        torch.bfloat16
                    )
                    _stage_end("active_activation", t_stage)

                    t_stage = _stage_start()
                    y_e = act @ w2_e.t()
                    _stage_end("active_w2_matmul", t_stage)

                    t_stage = _stage_start()
                    output_fp32.index_add_(
                        0,
                        tokens_for_e,
                        y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1),
                    )
                    _stage_end("active_scatter", t_stage)
                    del w1_e, w2_e

                t_stage = _stage_start()
                output.copy_(output_fp32.to(output.dtype))
                _stage_end("active_output_copy", t_stage)
                if timing_enabled:
                    total_ms = sum(stage_times.values())
                    parts = []
                    for name in sorted(stage_times):
                        value = stage_times[name]
                        count = max(1, stage_counts.get(name, 0))
                        parts.append(
                            f"{name}={value:.3f}ms avg={value / count:.3f}ms "
                            f"count={stage_counts.get(name, 0)}"
                        )
                    logger.warning(
                        "AtomTritonFP4SiluExperts active-only stage timing: "
                        "M=%d K=%d topk=%d local_experts=%d active_local_experts=%d "
                        "total=%.3fms %s",
                        M, K, topk, local_num_experts, int(used_experts.numel()),
                        total_ms, "; ".join(parts),
                    )
                return

        # Cudagraph-safe iteration: loop over ALL local experts (Python range,
        # unrolled at trace time) instead of torch.unique() (runtime tensor op,
        # not allowed during HIP stream capture). For experts with no routed
        # tokens we still run the full M*topk-row GEMMs but the per-row
        # contribution is zeroed via gate_w_masked, so the math is unchanged.
        # Index_add into a sentinel position (token 0) with weight 0 has no
        # numerical effect.
        zero_long = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
        zero_w = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
        for e_t in range(local_num_experts):
            t_stage = _stage_start()
            mask = flat_topk_ids == e_t  # bool [M*topk]
            safe_token_idx = torch.where(
                mask, flat_token_idx.to(torch.int64), zero_long
            )  # [M*topk]
            gate_w_masked = torch.where(
                mask, flat_topk_w, zero_w
            )  # [M*topk]

            x_e = hidden_states[safe_token_idx].to(torch.bfloat16)  # [M*topk, K]
            _stage_end("expert_select", t_stage)

            if _gfx942_mxfp4_dot is not None:
                t_stage = _stage_start()
                w1s_e = w1_scale[e_t]
                if hasattr(torch, "float8_e8m0fnu") and w1s_e.dtype == torch.float8_e8m0fnu:
                    w1s_e = w1s_e.view(torch.uint8)
                gate_up = _gfx942_mxfp4_dot(x_e, w1[e_t], w1s_e)  # [M*topk, 2N]
                _stage_end("w1_dot", t_stage)
                N = gate_up.shape[1] // 2
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                t_stage = _stage_start()
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                    torch.bfloat16
                )
                _stage_end("activation", t_stage)
                t_stage = _stage_start()
                w2s_e = w2_scale[e_t]
                if hasattr(torch, "float8_e8m0fnu") and w2s_e.dtype == torch.float8_e8m0fnu:
                    w2s_e = w2s_e.view(torch.uint8)
                y_e = _gfx942_mxfp4_dot(act, w2[e_t], w2s_e)  # [M*topk, K]
                _stage_end("w2_dot", t_stage)
            else:
                # Fallback: quark dequant + torch matmul. Correct but slower/more memory.
                t_stage = _stage_start()
                w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])  # [2N, K]
                w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])  # [K, N]
                _stage_end("dequant", t_stage)
                two_N = w1_e.shape[0]
                N = two_N // 2
                t_stage = _stage_start()
                gate_up = x_e @ w1_e.t()  # [M*topk, 2N]
                _stage_end("w1_matmul", t_stage)
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                t_stage = _stage_start()
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                    torch.bfloat16
                )
                _stage_end("activation", t_stage)
                t_stage = _stage_start()
                y_e = act @ w2_e.t()  # [M*topk, K]
                _stage_end("w2_matmul", t_stage)
                del w1_e, w2_e

            t_stage = _stage_start()
            output_fp32.index_add_(
                0,
                safe_token_idx,
                y_e.to(torch.float32) * gate_w_masked.unsqueeze(1),
            )
            _stage_end("scatter", t_stage)

        if os.getenv("VLLM_DSV4_MXFP4_COMPARE_ACTIVE_ONLY", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            if not torch.compiler.is_compiling() and not is_capturing:
                compare_fp32 = torch.zeros_like(output_fp32)
                valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < local_num_experts)
                used_experts = torch.unique(flat_topk_ids[valid_mask])
                for e_t_tensor in used_experts:
                    e_t = int(e_t_tensor.item())
                    mask = flat_topk_ids == e_t
                    tokens_for_e = flat_token_idx[mask].to(torch.int64)
                    gate_w_for_e = flat_topk_w[mask]
                    x_e = hidden_states[tokens_for_e].to(torch.bfloat16)
                    w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])
                    w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])
                    two_N = w1_e.shape[0]
                    N = two_N // 2
                    gate_up = x_e @ w1_e.t()
                    gate = gate_up[:, :N]
                    up = gate_up[:, N:]
                    if clamp_limit < float("inf"):
                        gate = torch.clamp(gate, max=clamp_limit)
                        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                    act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                        torch.bfloat16
                    )
                    y_e = act @ w2_e.t()
                    compare_fp32.index_add_(
                        0,
                        tokens_for_e,
                        y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1),
                    )
                    del w1_e, w2_e
                diff = (compare_fp32 - output_fp32).abs()
                denom = torch.clamp(output_fp32.abs(), min=1e-8)
                rel = diff / denom
                logger.warning(
                    "AtomTritonFP4SiluExperts active-only compare: "
                    "M=%d K=%d topk=%d local_experts=%d active_local_experts=%d "
                    "max_abs=%.6g mean_abs=%.6g max_rel=%.6g mean_rel=%.6g "
                    "all_std=%.6g active_std=%.6g",
                    M, K, topk, local_num_experts, int(used_experts.numel()),
                    diff.max().item() if diff.numel() else 0.0,
                    diff.mean().item() if diff.numel() else 0.0,
                    rel.max().item() if rel.numel() else 0.0,
                    rel.mean().item() if rel.numel() else 0.0,
                    output_fp32.std().item() if output_fp32.numel() else 0.0,
                    compare_fp32.std().item() if compare_fp32.numel() else 0.0,
                )

        t_stage = _stage_start()
        output.copy_(output_fp32.to(output.dtype))
        _stage_end("output_copy", t_stage)

        if timing_enabled:
            self._stage_timing_apply_calls = (
                getattr(self, "_stage_timing_apply_calls", 0) + 1
            )
            totals = getattr(self, "_stage_timing_totals_ms", {})
            counts = getattr(self, "_stage_timing_counts", {})
            for name, value in stage_times.items():
                totals[name] = totals.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + stage_counts.get(name, 0)
            self._stage_timing_totals_ms = totals
            self._stage_timing_counts = counts

            if self._stage_timing_apply_calls % stage_timing_interval == 0:
                ordered_names = (
                    "setup",
                    "expert_select",
                    "dequant",
                    "w1_matmul",
                    "w1_dot",
                    "activation",
                    "w2_matmul",
                    "w2_dot",
                    "scatter",
                    "output_copy",
                )
                total_ms = sum(totals.values())
                parts = []
                for name in ordered_names:
                    if name not in totals:
                        continue
                    value = totals[name]
                    count = max(1, counts.get(name, 0))
                    parts.append(
                        f"{name}={value:.3f}ms avg={value / count:.3f}ms "
                        f"count={counts.get(name, 0)}"
                    )
                logger.warning(
                    "AtomTritonFP4SiluExperts stage timing: apply_calls=%d "
                    "M=%d K=%d topk=%d local_experts=%d active_local_experts=%d "
                    "total=%.3fms %s",
                    self._stage_timing_apply_calls,
                    M,
                    K,
                    topk,
                    local_num_experts,
                    active_local_experts,
                    total_ms,
                    "; ".join(parts),
                )
