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

        routed_chunk_m = 0
        if os.getenv("VLLM_DSV4_MXFP4_ROUTED_KERNEL", "0") == "1":
            try:
                routed_chunk_m = int(os.getenv("VLLM_DSV4_MXFP4_ROUTED_CHUNK_M", "0"))
            except ValueError:
                routed_chunk_m = 0
        if routed_chunk_m > 0 and M > routed_chunk_m:
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
        if not hasattr(self, "_logged_scale_dtype"):
            self._logged_scale_dtype = True
        clamp_limit = quant_config.gemm1_clamp_limit
        if clamp_limit is None or clamp_limit <= 0:
            clamp_limit = float("inf")
        clamp_limit = float(clamp_limit)

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
            if _routed_mxfp4_stage1_into is None or _routed_mxfp4_stage2_scatter_into is None:
                raise RuntimeError(
                    "VLLM_DSV4_MXFP4_ROUTED_KERNEL=1 but routed_mxfp4_kernels "
                    "could not be imported"
                )
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
                if (
                    not hasattr(self, "_routed_gate_up_2d")
                    or self._routed_gate_up_2d.shape != gate_up_shape
                    or self._routed_gate_up_2d.device != hidden_states.device
                ):
                    self._routed_gate_up_2d = torch.empty(
                        gate_up_shape, dtype=torch.bfloat16, device=hidden_states.device
                    )
                routed_gate_up = self._routed_gate_up_2d
                routed_gate_up_from_workspace = False

            routed_output_fp32 = output_fp32

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
                        if (
                            not hasattr(self, "_routed_route_out")
                            or self._routed_route_out.shape != route_out_shape
                            or self._routed_route_out.device != hidden_states.device
                        ):
                            self._routed_route_out = torch.empty(
                                route_out_shape,
                                dtype=torch.float32,
                                device=hidden_states.device,
                            )
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
                        reduce_order = os.getenv(
                            "VLLM_DSV4_MXFP4_DETERMINISTIC_REDUCE_ORDER",
                            "topk",
                        ).lower()
                        if reduce_order == "expert":
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
                output.copy_(routed_output_fp32)
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
            output.copy_(routed_output_fp32)

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
                return

        # Optional diagnostic path using fixed-shape metadata but deterministic
        # expert-id order. It remains no-graph only because it slices by runtime
        # expert_counts; use it to test numerical/order behavior before writing
        # graph-safe fixed-capacity compute kernels.
        # Optional diagnostic fast path: dynamic active-expert loop for no-graph
        # experiments only. This is intentionally disabled by default because
        # torch.unique/nonzero and dynamic shapes are not HIP graph capture safe.
        if os.getenv("VLLM_DSV4_MXFP4_ACTIVE_ONLY", "0") == "1":
            try:
                is_capturing = torch.cuda.is_current_stream_capturing()
            except Exception:
                is_capturing = False
            if not torch.compiler.is_compiling() and not is_capturing:
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

                    output_fp32.index_add_(
                        0,
                        tokens_for_e,
                        y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1),
                    )
                    del w1_e, w2_e

                output.copy_(output_fp32.to(output.dtype))
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
            mask = flat_topk_ids == e_t  # bool [M*topk]
            safe_token_idx = torch.where(
                mask, flat_token_idx.to(torch.int64), zero_long
            )  # [M*topk]
            gate_w_masked = torch.where(
                mask, flat_topk_w, zero_w
            )  # [M*topk]

            x_e = hidden_states[safe_token_idx].to(torch.bfloat16)  # [M*topk, K]

            if _gfx942_mxfp4_dot is not None:
                w1s_e = w1_scale[e_t]
                if hasattr(torch, "float8_e8m0fnu") and w1s_e.dtype == torch.float8_e8m0fnu:
                    w1s_e = w1s_e.view(torch.uint8)
                gate_up = _gfx942_mxfp4_dot(x_e, w1[e_t], w1s_e)  # [M*topk, 2N]
                N = gate_up.shape[1] // 2
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                    torch.bfloat16
                )
                w2s_e = w2_scale[e_t]
                if hasattr(torch, "float8_e8m0fnu") and w2s_e.dtype == torch.float8_e8m0fnu:
                    w2s_e = w2s_e.view(torch.uint8)
                y_e = _gfx942_mxfp4_dot(act, w2[e_t], w2s_e)  # [M*topk, K]
            else:
                # Fallback: quark dequant + torch matmul. Correct but slower/more memory.
                w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])  # [2N, K]
                w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])  # [K, N]
                two_N = w1_e.shape[0]
                N = two_N // 2
                gate_up = x_e @ w1_e.t()  # [M*topk, 2N]
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                    torch.bfloat16
                )
                y_e = act @ w2_e.t()  # [M*topk, K]
                del w1_e, w2_e

            output_fp32.index_add_(
                0,
                safe_token_idx,
                y_e.to(torch.float32) * gate_w_masked.unsqueeze(1),
            )

        output.copy_(output_fp32.to(output.dtype))

