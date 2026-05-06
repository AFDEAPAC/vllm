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

logger = logging.getLogger(__name__)

# Keep this branch self-contained: the production fallback below uses quark
# dequantization plus torch matmul instead of an out-of-tree prototype kernel.
_gfx942_mxfp4_dot = None


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

        w1_scale = quant_config.w1_scale
        w2_scale = quant_config.w2_scale
        assert w1_scale is not None, "w1_scale required for mxfp4 dequant"
        assert w2_scale is not None, "w2_scale required for mxfp4 dequant"
        if not hasattr(self, "_logged_scale_dtype"):
            logger.warning(
                "AtomTritonFP4SiluExperts scale dtypes: w1_scale=%s "
                "w2_scale=%s weight dtypes: w1=%s w2=%s gfx_kernel=%s",
                getattr(w1_scale, "dtype", None), getattr(w2_scale, "dtype", None),
                getattr(w1, "dtype", None), getattr(w2, "dtype", None),
                _gfx942_mxfp4_dot is not None,
            )
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
        )

        # Output accumulator (fp32 for accumulation precision)
        output_fp32 = torch.zeros(M, K, dtype=torch.float32, device=output.device)

        # Determine which experts are used this call
        used_experts = torch.unique(flat_topk_ids)
        # Filter to local experts only (in case expert_map is identity)
        used_experts = used_experts[
            (used_experts >= 0) & (used_experts < local_num_experts)
        ]

        for e_t in used_experts.tolist():
            mask = flat_topk_ids == e_t
            tokens_for_e = flat_token_idx[mask].to(torch.int64)
            gate_w_for_e = flat_topk_w[mask]

            x_e = hidden_states[tokens_for_e].to(torch.bfloat16)  # (n_e, K)

            if _gfx942_mxfp4_dot is not None:
                # Direct gfx942 MXFP4 dequant-dot, no intermediate bf16 weight.
                w1s_e = w1_scale[e_t]
                if hasattr(torch, "float8_e8m0fnu") and w1s_e.dtype == torch.float8_e8m0fnu:
                    w1s_e = w1s_e.view(torch.uint8)
                gate_up = _gfx942_mxfp4_dot(x_e, w1[e_t], w1s_e)  # (n_e, 2N)
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
                y_e = _gfx942_mxfp4_dot(act, w2[e_t], w2s_e)  # (n_e, K)
            else:
                # Fallback: quark dequant + torch matmul. Correct but slower/more memory.
                w1_e = _dequant_mxfp4_to_bf16(w1[e_t], w1_scale[e_t])  # (2N, K)
                w2_e = _dequant_mxfp4_to_bf16(w2[e_t], w2_scale[e_t])  # (K, N)
                two_N = w1_e.shape[0]
                N = two_N // 2
                gate_up = x_e @ w1_e.t()  # (n_e, 2N)
                gate = gate_up[:, :N]
                up = gate_up[:, N:]
                if clamp_limit < float("inf"):
                    gate = torch.clamp(gate, max=clamp_limit)
                    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
                act = (F.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(
                    torch.bfloat16
                )
                y_e = act @ w2_e.t()  # (n_e, K)
                del w1_e, w2_e

            output_fp32.index_add_(
                0,
                tokens_for_e,
                y_e.to(torch.float32) * gate_w_for_e.unsqueeze(1),
            )

        output.copy_(output_fp32.to(output.dtype))
