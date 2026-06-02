# SPDX-License-Identifier: Apache-2.0
"""Experimental graph-shape-safe routed MXFP4 MoE kernels for gfx942.

These kernels consume fixed metadata produced by ATOM's metadata builder:

    expert_counts[E]
    route_positions[E, CAP]  # positions into flattened (M * topk) routing rows
    flat_token_idx[M * topk]
    flat_topk_w[M * topk]

The launch grids cover all local experts. Inactive experts return inside the
kernel, so Python does not need dynamic expert loops during capture.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fp4_e2m1_to_f32(nib):
    mag = nib & 0x7
    val = tl.where(mag == 0, 0.0, 0.0)
    val = tl.where(mag == 1, 0.5, val)
    val = tl.where(mag == 2, 1.0, val)
    val = tl.where(mag == 3, 1.5, val)
    val = tl.where(mag == 4, 2.0, val)
    val = tl.where(mag == 5, 3.0, val)
    val = tl.where(mag == 6, 4.0, val)
    val = tl.where(mag == 7, 6.0, val)
    sign = tl.where((nib & 0x8) != 0, -1.0, 1.0)
    return val * sign


@triton.jit
def _stage1_kernel(
    HIDDEN,
    W1,
    S1,
    COUNTS,
    ROUTE_POS,
    FLAT_TOKEN_IDX,
    GATE_UP,
    CAP: tl.constexpr,
    K: tl.constexpr,
    TWO_N: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wkp: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_skb: tl.constexpr,
    stride_pe: tl.constexpr,
    stride_pr: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    count = tl.load(COUNTS + expert)
    if count <= 0:
        return

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if pid_m * BLOCK_M >= count:
        return
    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    valid_rows = rows < count
    route_pos = tl.load(
        ROUTE_POS + expert * stride_pe + rows * stride_pr,
        mask=valid_rows & (rows < CAP),
        other=0,
    )
    token_pos = tl.load(FLAT_TOKEN_IDX + route_pos, mask=valid_rows, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + ks
        x = tl.load(
            HIDDEN + token_pos[:, None] * stride_hm + k[None, :] * stride_hk,
            mask=valid_rows[:, None] & (k[None, :] < K),
            other=0.0,
        )
        packed_k = k // 2
        bytes_ = tl.load(
            W1
            + expert * stride_we
            + ns[:, None] * stride_wn
            + packed_k[None, :] * stride_wkp,
            mask=(ns[:, None] < TWO_N) & (k[None, :] < K),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((k[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)

        scale_idx = k // 32
        scale_u8 = tl.load(
            S1
            + expert * stride_se
            + ns[:, None] * stride_sn
            + scale_idx[None, :] * stride_skb,
            mask=(ns[:, None] < TWO_N) & (k[None, :] < K),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), input_precision="tf32")

    tl.store(
        GATE_UP + route_pos[:, None] * stride_gm + ns[None, :] * stride_gn,
        acc,
        mask=valid_rows[:, None] & (rows[:, None] < CAP) & (ns[None, :] < TWO_N),
    )


@triton.jit
def _stage2_scatter_kernel(
    GATE_UP,
    W2,
    S2,
    COUNTS,
    ROUTE_POS,
    FLAT_TOKEN_IDX,
    FLAT_TOPK_W,
    OUT,
    CAP: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_wip: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sib: tl.constexpr,
    stride_pe: tl.constexpr,
    stride_pr: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    expert = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h = tl.program_id(2)

    count = tl.load(COUNTS + expert)
    if count <= 0:
        return

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if pid_m * BLOCK_M >= count:
        return
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    is_ = tl.arange(0, BLOCK_I)
    valid_rows = rows < count
    route_pos = tl.load(
        ROUTE_POS + expert * stride_pe + rows * stride_pr,
        mask=valid_rows & (rows < CAP),
        other=0,
    )
    token_pos = tl.load(FLAT_TOKEN_IDX + route_pos, mask=valid_rows, other=0)
    route_w = tl.load(FLAT_TOPK_W + route_pos, mask=valid_rows, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
    for i0 in range(0, I, BLOCK_I):
        i = i0 + is_
        gate = tl.load(
            GATE_UP + route_pos[:, None] * stride_gm + i[None, :] * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            GATE_UP
            + route_pos[:, None] * stride_gm
            + (I + i[None, :]) * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        if APPLY_CLAMP:
            gate = tl.minimum(gate, CLAMP_LIMIT)
            up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
        act = (gate / (1.0 + tl.exp(-gate))) * up

        packed_i = i // 2
        bytes_ = tl.load(
            W2 + expert * stride_we + hs[:, None] * stride_wh + packed_i[None, :] * stride_wip,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((i[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)

        scale_idx = i // 32
        scale_u8 = tl.load(
            S2 + expert * stride_se + hs[:, None] * stride_sh + scale_idx[None, :] * stride_sib,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(act.to(tl.bfloat16), tl.trans(w), input_precision="ieee")

    # PyTorch BF16 matmul returns BF16 before the routing weight is applied.
    # Match that rounding point before accumulating into the FP32 output.
    acc = acc.to(tl.bfloat16).to(tl.float32) * route_w[:, None]
    tl.atomic_add(
        OUT + token_pos[:, None] * stride_om + hs[None, :] * stride_oh,
        acc,
        sem="relaxed",
        mask=valid_rows[:, None] & (hs[None, :] < H),
    )


@triton.jit
def _stage1_ragged_kernel(
    HIDDEN,
    W1,
    S1,
    COUNTS,
    OFFSETS,
    SORTED_ROUTE_POS,
    BLOCK_MAP,
    FLAT_TOKEN_IDX,
    GATE_UP,
    K: tl.constexpr,
    TWO_N: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wkp: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_skb: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    packed = tl.load(BLOCK_MAP + tile)
    if packed < 0:
        return

    expert = packed & 0xFFFF
    tile_m = packed >> 16
    count = tl.load(COUNTS + expert)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if tile_m * BLOCK_M >= count:
        return

    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    valid_rows = rows < count
    base = tl.load(OFFSETS + expert)
    route_pos = tl.load(SORTED_ROUTE_POS + base + rows, mask=valid_rows, other=0)
    token_pos = tl.load(FLAT_TOKEN_IDX + route_pos, mask=valid_rows, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + ks
        x = tl.load(
            HIDDEN + token_pos[:, None] * stride_hm + k[None, :] * stride_hk,
            mask=valid_rows[:, None] & (k[None, :] < K),
            other=0.0,
        )
        packed_k = k // 2
        bytes_ = tl.load(
            W1
            + expert * stride_we
            + ns[:, None] * stride_wn
            + packed_k[None, :] * stride_wkp,
            mask=(ns[:, None] < TWO_N) & (k[None, :] < K),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((k[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)

        scale_idx = k // 32
        scale_u8 = tl.load(
            S1
            + expert * stride_se
            + ns[:, None] * stride_sn
            + scale_idx[None, :] * stride_skb,
            mask=(ns[:, None] < TWO_N) & (k[None, :] < K),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), input_precision="tf32")

    tl.store(
        GATE_UP + route_pos[:, None] * stride_gm + ns[None, :] * stride_gn,
        acc,
        mask=valid_rows[:, None] & (ns[None, :] < TWO_N),
    )


@triton.jit
def _stage2_ragged_scatter_kernel(
    GATE_UP,
    W2,
    S2,
    COUNTS,
    OFFSETS,
    SORTED_ROUTE_POS,
    BLOCK_MAP,
    FLAT_TOKEN_IDX,
    FLAT_TOPK_W,
    OUT,
    I: tl.constexpr,
    H: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_wip: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sib: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_h = tl.program_id(1)
    packed = tl.load(BLOCK_MAP + tile)
    if packed < 0:
        return

    expert = packed & 0xFFFF
    tile_m = packed >> 16
    count = tl.load(COUNTS + expert)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if tile_m * BLOCK_M >= count:
        return

    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    is_ = tl.arange(0, BLOCK_I)
    valid_rows = rows < count
    base = tl.load(OFFSETS + expert)
    route_pos = tl.load(SORTED_ROUTE_POS + base + rows, mask=valid_rows, other=0)
    token_pos = tl.load(FLAT_TOKEN_IDX + route_pos, mask=valid_rows, other=0)
    route_w = tl.load(FLAT_TOPK_W + route_pos, mask=valid_rows, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
    for i0 in range(0, I, BLOCK_I):
        i = i0 + is_
        gate = tl.load(
            GATE_UP + route_pos[:, None] * stride_gm + i[None, :] * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            GATE_UP
            + route_pos[:, None] * stride_gm
            + (I + i[None, :]) * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        if APPLY_CLAMP:
            gate = tl.minimum(gate, CLAMP_LIMIT)
            up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
        act = (gate / (1.0 + tl.exp(-gate))) * up

        packed_i = i // 2
        bytes_ = tl.load(
            W2 + expert * stride_we + hs[:, None] * stride_wh + packed_i[None, :] * stride_wip,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((i[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)

        scale_idx = i // 32
        scale_u8 = tl.load(
            S2 + expert * stride_se + hs[:, None] * stride_sh + scale_idx[None, :] * stride_sib,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(act.to(tl.bfloat16), tl.trans(w), input_precision="ieee")

    acc = acc.to(tl.bfloat16).to(tl.float32) * route_w[:, None]
    tl.atomic_add(
        OUT + token_pos[:, None] * stride_om + hs[None, :] * stride_oh,
        acc,
        sem="relaxed",
        mask=valid_rows[:, None] & (hs[None, :] < H),
    )


@triton.jit
def _stage1_offsets_kernel(
    HIDDEN,
    W1,
    S1,
    COUNTS,
    OFFSETS,
    SORTED_ROUTE_POS,
    FLAT_TOKEN_IDX,
    GATE_UP,
    K: tl.constexpr,
    TWO_N: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wkp: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_skb: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    tile_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    count = tl.load(COUNTS + expert)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if tile_m * BLOCK_M >= count:
        return

    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    valid_rows = rows < count
    base = tl.load(OFFSETS + expert)
    route_pos = tl.load(SORTED_ROUTE_POS + base + rows, mask=valid_rows, other=0)
    token_pos = tl.load(FLAT_TOKEN_IDX + route_pos, mask=valid_rows, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + ks
        x = tl.load(
            HIDDEN + token_pos[:, None] * stride_hm + k[None, :] * stride_hk,
            mask=valid_rows[:, None] & (k[None, :] < K),
            other=0.0,
        )
        packed_k = k // 2
        bytes_ = tl.load(
            W1
            + expert * stride_we
            + ns[:, None] * stride_wn
            + packed_k[None, :] * stride_wkp,
            mask=(ns[:, None] < TWO_N) & (k[None, :] < K),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((k[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)
        scale_idx = k // 32
        scale_u8 = tl.load(
            S1
            + expert * stride_se
            + ns[:, None] * stride_sn
            + scale_idx[None, :] * stride_skb,
            mask=(ns[:, None] < TWO_N) & (k[None, :] < K),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w), input_precision="tf32")

    tl.store(
        GATE_UP + route_pos[:, None] * stride_gm + ns[None, :] * stride_gn,
        acc,
        mask=valid_rows[:, None] & (ns[None, :] < TWO_N),
    )


@triton.jit
def _stage2_offsets_scatter_kernel(
    GATE_UP,
    W2,
    S2,
    COUNTS,
    OFFSETS,
    SORTED_ROUTE_POS,
    FLAT_TOKEN_IDX,
    FLAT_TOPK_W,
    OUT,
    I: tl.constexpr,
    H: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_wip: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sib: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    expert = tl.program_id(0)
    tile_m = tl.program_id(1)
    pid_h = tl.program_id(2)
    count = tl.load(COUNTS + expert)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if tile_m * BLOCK_M >= count:
        return

    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    is_ = tl.arange(0, BLOCK_I)
    valid_rows = rows < count
    base = tl.load(OFFSETS + expert)
    route_pos = tl.load(SORTED_ROUTE_POS + base + rows, mask=valid_rows, other=0)
    token_pos = tl.load(FLAT_TOKEN_IDX + route_pos, mask=valid_rows, other=0)
    route_w = tl.load(FLAT_TOPK_W + route_pos, mask=valid_rows, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
    for i0 in range(0, I, BLOCK_I):
        i = i0 + is_
        gate = tl.load(
            GATE_UP + route_pos[:, None] * stride_gm + i[None, :] * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            GATE_UP
            + route_pos[:, None] * stride_gm
            + (I + i[None, :]) * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        if APPLY_CLAMP:
            gate = tl.minimum(gate, CLAMP_LIMIT)
            up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
        act = (gate / (1.0 + tl.exp(-gate))) * up

        packed_i = i // 2
        bytes_ = tl.load(
            W2 + expert * stride_we + hs[:, None] * stride_wh + packed_i[None, :] * stride_wip,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((i[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)
        scale_idx = i // 32
        scale_u8 = tl.load(
            S2 + expert * stride_se + hs[:, None] * stride_sh + scale_idx[None, :] * stride_sib,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(act.to(tl.bfloat16), tl.trans(w), input_precision="ieee")

    acc = acc.to(tl.bfloat16).to(tl.float32) * route_w[:, None]
    tl.atomic_add(
        OUT + token_pos[:, None] * stride_om + hs[None, :] * stride_oh,
        acc,
        sem="relaxed",
        mask=valid_rows[:, None] & (hs[None, :] < H),
    )


@triton.jit
def _stage2_offsets_store_route_kernel(
    GATE_UP,
    W2,
    S2,
    COUNTS,
    OFFSETS,
    SORTED_ROUTE_POS,
    FLAT_TOPK_W,
    ROUTE_OUT,
    I: tl.constexpr,
    H: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_wip: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sib: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rh: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    expert = tl.program_id(0)
    tile_m = tl.program_id(1)
    pid_h = tl.program_id(2)
    count = tl.load(COUNTS + expert)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if tile_m * BLOCK_M >= count:
        return

    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    is_ = tl.arange(0, BLOCK_I)
    valid_rows = rows < count
    base = tl.load(OFFSETS + expert)
    route_pos = tl.load(SORTED_ROUTE_POS + base + rows, mask=valid_rows, other=0)
    route_w = tl.load(FLAT_TOPK_W + route_pos, mask=valid_rows, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
    for i0 in range(0, I, BLOCK_I):
        i = i0 + is_
        gate = tl.load(
            GATE_UP + route_pos[:, None] * stride_gm + i[None, :] * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            GATE_UP
            + route_pos[:, None] * stride_gm
            + (I + i[None, :]) * stride_gn,
            mask=valid_rows[:, None] & (i[None, :] < I),
            other=0.0,
        ).to(tl.float32)
        if APPLY_CLAMP:
            gate = tl.minimum(gate, CLAMP_LIMIT)
            up = tl.minimum(tl.maximum(up, -CLAMP_LIMIT), CLAMP_LIMIT)
        act = (gate / (1.0 + tl.exp(-gate))) * up

        packed_i = i // 2
        bytes_ = tl.load(
            W2 + expert * stride_we + hs[:, None] * stride_wh + packed_i[None, :] * stride_wip,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=0,
        )
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        nib = tl.where((i[None, :] & 1) == 0, low, high)
        w = _fp4_e2m1_to_f32(nib)
        scale_idx = i // 32
        scale_u8 = tl.load(
            S2 + expert * stride_se + hs[:, None] * stride_sh + scale_idx[None, :] * stride_sib,
            mask=(hs[:, None] < H) & (i[None, :] < I),
            other=127,
        ).to(tl.float32)
        w = (w * tl.exp2(scale_u8 - 127.0)).to(tl.bfloat16)
        acc += tl.dot(act.to(tl.bfloat16), tl.trans(w), input_precision="ieee")

    acc = acc.to(tl.bfloat16).to(tl.float32) * route_w[:, None]
    tl.store(
        ROUTE_OUT + route_pos[:, None] * stride_rm + hs[None, :] * stride_rh,
        acc,
        mask=valid_rows[:, None] & (hs[None, :] < H),
    )


@triton.jit
def _reduce_route_out_kernel(
    ROUTE_OUT,
    OUT,
    M: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for j in range(0, TOPK):
        route_pos = token * TOPK + j
        vals = tl.load(
            ROUTE_OUT + route_pos * stride_rm + hs * stride_rh,
            mask=(token < M) & (hs < H),
            other=0.0,
        ).to(tl.float32)
        acc += vals
    tl.store(
        OUT + token * stride_om + hs * stride_oh,
        acc,
        mask=(token < M) & (hs < H),
    )


@triton.jit
def _reduce_route_out_expert_order_kernel(
    ROUTE_OUT,
    FLAT_TOPK_IDS,
    OUT,
    M: tl.constexpr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    base = token * TOPK
    acc = tl.zeros((BLOCK_H,), tl.float32)

    # Match the Python compare/reference path, which accumulates routes in
    # ascending expert-id order via the per-expert loop.
    for rank in range(0, TOPK):
        for j in range(0, TOPK):
            id_j = tl.load(FLAT_TOPK_IDS + base + j, mask=token < M, other=2147483647)
            order = 0
            for q in range(0, TOPK):
                id_q = tl.load(FLAT_TOPK_IDS + base + q, mask=token < M, other=2147483647)
                order += tl.where((id_q < id_j) | ((id_q == id_j) & (q < j)), 1, 0)
            route_pos = base + j
            vals = tl.load(
                ROUTE_OUT + route_pos * stride_rm + hs * stride_rh,
                mask=(token < M) & (hs < H),
                other=0.0,
            ).to(tl.float32)
            acc += tl.where(order == rank, vals, 0.0)

    tl.store(
        OUT + token * stride_om + hs * stride_oh,
        acc,
        mask=(token < M) & (hs < H),
    )


@triton.jit
def _reduce_route_out_expert_order_topk8_kernel(
    ROUTE_OUT,
    FLAT_TOPK_IDS,
    OUT,
    M: tl.constexpr,
    H: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    base = token * 8
    valid = token < M

    id0 = tl.load(FLAT_TOPK_IDS + base + 0, mask=valid, other=2147483647)
    id1 = tl.load(FLAT_TOPK_IDS + base + 1, mask=valid, other=2147483647)
    id2 = tl.load(FLAT_TOPK_IDS + base + 2, mask=valid, other=2147483647)
    id3 = tl.load(FLAT_TOPK_IDS + base + 3, mask=valid, other=2147483647)
    id4 = tl.load(FLAT_TOPK_IDS + base + 4, mask=valid, other=2147483647)
    id5 = tl.load(FLAT_TOPK_IDS + base + 5, mask=valid, other=2147483647)
    id6 = tl.load(FLAT_TOPK_IDS + base + 6, mask=valid, other=2147483647)
    id7 = tl.load(FLAT_TOPK_IDS + base + 7, mask=valid, other=2147483647)

    # order_j = count of routes that sort before j by (expert_id, topk_index).
    r0 = (
        tl.where(id1 < id0, 1, 0)
        + tl.where(id2 < id0, 1, 0)
        + tl.where(id3 < id0, 1, 0)
        + tl.where(id4 < id0, 1, 0)
        + tl.where(id5 < id0, 1, 0)
    )
    r1 = (
        tl.where((id0 < id1) | ((id0 == id1) & (0 < 1)), 1, 0)
        + tl.where(id2 < id1, 1, 0)
        + tl.where(id3 < id1, 1, 0)
        + tl.where(id4 < id1, 1, 0)
        + tl.where(id5 < id1, 1, 0)
    )
    r2 = (
        tl.where((id0 < id2) | ((id0 == id2) & (0 < 2)), 1, 0)
        + tl.where((id1 < id2) | ((id1 == id2) & (1 < 2)), 1, 0)
        + tl.where(id3 < id2, 1, 0)
        + tl.where(id4 < id2, 1, 0)
        + tl.where(id5 < id2, 1, 0)
    )
    r3 = (
        tl.where((id0 < id3) | ((id0 == id3) & (0 < 3)), 1, 0)
        + tl.where((id1 < id3) | ((id1 == id3) & (1 < 3)), 1, 0)
        + tl.where((id2 < id3) | ((id2 == id3) & (2 < 3)), 1, 0)
        + tl.where(id4 < id3, 1, 0)
        + tl.where(id5 < id3, 1, 0)
    )
    r4 = (
        tl.where((id0 < id4) | ((id0 == id4) & (0 < 4)), 1, 0)
        + tl.where((id1 < id4) | ((id1 == id4) & (1 < 4)), 1, 0)
        + tl.where((id2 < id4) | ((id2 == id4) & (2 < 4)), 1, 0)
        + tl.where((id3 < id4) | ((id3 == id4) & (3 < 4)), 1, 0)
        + tl.where(id5 < id4, 1, 0)
    )
    r5 = (
        tl.where((id0 < id5) | ((id0 == id5) & (0 < 5)), 1, 0)
        + tl.where((id1 < id5) | ((id1 == id5) & (1 < 5)), 1, 0)
        + tl.where((id2 < id5) | ((id2 == id5) & (2 < 5)), 1, 0)
        + tl.where((id3 < id5) | ((id3 == id5) & (3 < 5)), 1, 0)
        + tl.where((id4 < id5) | ((id4 == id5) & (4 < 5)), 1, 0)
    )
    r6 = (
        tl.where(id0 <= id6, 1, 0)
        + tl.where(id1 <= id6, 1, 0)
        + tl.where(id2 <= id6, 1, 0)
        + tl.where(id3 <= id6, 1, 0)
        + tl.where(id4 <= id6, 1, 0)
        + tl.where(id5 <= id6, 1, 0)
    )
    r7 = (
        tl.where(id0 <= id7, 1, 0)
        + tl.where(id1 <= id7, 1, 0)
        + tl.where(id2 <= id7, 1, 0)
        + tl.where(id3 <= id7, 1, 0)
        + tl.where(id4 <= id7, 1, 0)
        + tl.where(id5 <= id7, 1, 0)
        + tl.where(id6 <= id7, 1, 0)
    )

    v0 = tl.load(ROUTE_OUT + (base + 0) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v1 = tl.load(ROUTE_OUT + (base + 1) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v2 = tl.load(ROUTE_OUT + (base + 2) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v3 = tl.load(ROUTE_OUT + (base + 3) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v4 = tl.load(ROUTE_OUT + (base + 4) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v5 = tl.load(ROUTE_OUT + (base + 5) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v6 = tl.load(ROUTE_OUT + (base + 6) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v7 = tl.load(ROUTE_OUT + (base + 7) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_H,), tl.float32)
    for rank in range(0, 8):
        acc += tl.where(r0 == rank, v0, 0.0)
        acc += tl.where(r1 == rank, v1, 0.0)
        acc += tl.where(r2 == rank, v2, 0.0)
        acc += tl.where(r3 == rank, v3, 0.0)
        acc += tl.where(r4 == rank, v4, 0.0)
        acc += tl.where(r5 == rank, v5, 0.0)
        acc += tl.where(r6 == rank, v6, 0.0)
        acc += tl.where(r7 == rank, v7, 0.0)

    tl.store(
        OUT + token * stride_om + hs * stride_oh,
        acc,
        mask=valid & (hs < H),
    )


@triton.jit
def _reduce_route_out_expert_order_topk6_kernel(
    ROUTE_OUT,
    FLAT_TOPK_IDS,
    OUT,
    M: tl.constexpr,
    H: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    base = token * 6
    valid = token < M

    id0 = tl.load(FLAT_TOPK_IDS + base + 0, mask=valid, other=2147483647)
    id1 = tl.load(FLAT_TOPK_IDS + base + 1, mask=valid, other=2147483647)
    id2 = tl.load(FLAT_TOPK_IDS + base + 2, mask=valid, other=2147483647)
    id3 = tl.load(FLAT_TOPK_IDS + base + 3, mask=valid, other=2147483647)
    id4 = tl.load(FLAT_TOPK_IDS + base + 4, mask=valid, other=2147483647)
    id5 = tl.load(FLAT_TOPK_IDS + base + 5, mask=valid, other=2147483647)

    # Rank routes by ascending expert id, tie-breaking by original top-k index,
    # matching the generic expert-order reduce and Python per-expert loop.
    r0 = 0
    r1 = tl.where(id0 <= id1, 1, 0)
    r2 = tl.where(id0 <= id2, 1, 0) + tl.where(id1 <= id2, 1, 0)
    r3 = (
        tl.where(id0 <= id3, 1, 0)
        + tl.where(id1 <= id3, 1, 0)
        + tl.where(id2 <= id3, 1, 0)
    )
    r4 = (
        tl.where(id0 <= id4, 1, 0)
        + tl.where(id1 <= id4, 1, 0)
        + tl.where(id2 <= id4, 1, 0)
        + tl.where(id3 <= id4, 1, 0)
    )
    r5 = (
        tl.where(id0 <= id5, 1, 0)
        + tl.where(id1 <= id5, 1, 0)
        + tl.where(id2 <= id5, 1, 0)
        + tl.where(id3 <= id5, 1, 0)
        + tl.where(id4 <= id5, 1, 0)
    )

    v0 = tl.load(ROUTE_OUT + (base + 0) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v1 = tl.load(ROUTE_OUT + (base + 1) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v2 = tl.load(ROUTE_OUT + (base + 2) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v3 = tl.load(ROUTE_OUT + (base + 3) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v4 = tl.load(ROUTE_OUT + (base + 4) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)
    v5 = tl.load(ROUTE_OUT + (base + 5) * stride_rm + hs * stride_rh, mask=valid & (hs < H), other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_H,), tl.float32)
    for rank in range(0, 6):
        acc += tl.where(r0 == rank, v0, 0.0)
        acc += tl.where(r1 == rank, v1, 0.0)
        acc += tl.where(r2 == rank, v2, 0.0)
        acc += tl.where(r3 == rank, v3, 0.0)
        acc += tl.where(r4 == rank, v4, 0.0)
        acc += tl.where(r5 == rank, v5, 0.0)

    tl.store(
        OUT + token * stride_om + hs * stride_oh,
        acc,
        mask=valid & (hs < H),
    )


def _scale_as_u8(scale: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "float8_e8m0fnu") and scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8)
    return scale


def routed_stage1_into(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    route_positions: torch.Tensor,
    flat_token_idx: torch.Tensor,
    gate_up: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 128,
) -> None:
    w1_scale = _scale_as_u8(w1_scale)
    local_experts, two_n, k_packed = w1.shape
    cap = route_positions.shape[1]
    k = hidden_states.shape[1]
    assert k_packed * 2 == k
    assert gate_up.shape[1] == two_n
    grid = (local_experts, triton.cdiv(cap, block_m), triton.cdiv(two_n, block_n))
    _stage1_kernel[grid](
        hidden_states,
        w1,
        w1_scale,
        expert_counts,
        route_positions,
        flat_token_idx,
        gate_up,
        cap,
        k,
        two_n,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        route_positions.stride(0),
        route_positions.stride(1),
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def routed_stage2_scatter_into(
    gate_up: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    route_positions: torch.Tensor,
    flat_token_idx: torch.Tensor,
    flat_topk_w: torch.Tensor,
    output_fp32: torch.Tensor,
    *,
    clamp_limit: float,
    block_m: int = 16,
    block_h: int = 16,
    block_i: int = 128,
) -> None:
    w2_scale = _scale_as_u8(w2_scale)
    local_experts, h, i_packed = w2.shape
    i = i_packed * 2
    cap = route_positions.shape[1]
    assert gate_up.shape[1] == i * 2
    assert output_fp32.shape[1] == h
    output_fp32.zero_()
    grid = (local_experts, triton.cdiv(cap, block_m), triton.cdiv(h, block_h))
    apply_clamp = clamp_limit < float("inf")
    _stage2_scatter_kernel[grid](
        gate_up,
        w2,
        w2_scale,
        expert_counts,
        route_positions,
        flat_token_idx,
        flat_topk_w,
        output_fp32,
        cap,
        i,
        h,
        apply_clamp,
        clamp_limit if apply_clamp else 0.0,
        gate_up.stride(0),
        gate_up.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        route_positions.stride(0),
        route_positions.stride(1),
        output_fp32.stride(0),
        output_fp32.stride(1),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=4,
    )


def routed_ragged_stage1_into(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_route_positions: torch.Tensor,
    block_map: torch.Tensor,
    flat_token_idx: torch.Tensor,
    gate_up: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 128,
) -> None:
    w1_scale = _scale_as_u8(w1_scale)
    local_experts, two_n, k_packed = w1.shape
    k = hidden_states.shape[1]
    assert k_packed * 2 == k
    assert gate_up.shape[1] == two_n
    assert expert_counts.shape[0] == local_experts
    grid = (block_map.shape[0], triton.cdiv(two_n, block_n))
    _stage1_ragged_kernel[grid](
        hidden_states,
        w1,
        w1_scale,
        expert_counts,
        expert_offsets,
        sorted_route_positions,
        block_map,
        flat_token_idx,
        gate_up,
        k,
        two_n,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def routed_ragged_stage2_scatter_into(
    gate_up: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_route_positions: torch.Tensor,
    block_map: torch.Tensor,
    flat_token_idx: torch.Tensor,
    flat_topk_w: torch.Tensor,
    output_fp32: torch.Tensor,
    *,
    clamp_limit: float,
    block_m: int = 16,
    block_h: int = 16,
    block_i: int = 128,
) -> None:
    w2_scale = _scale_as_u8(w2_scale)
    local_experts, h, i_packed = w2.shape
    i = i_packed * 2
    assert gate_up.shape[1] == i * 2
    assert output_fp32.shape[1] == h
    assert expert_counts.shape[0] == local_experts
    output_fp32.zero_()
    apply_clamp = clamp_limit < float("inf")
    grid = (block_map.shape[0], triton.cdiv(h, block_h))
    _stage2_ragged_scatter_kernel[grid](
        gate_up,
        w2,
        w2_scale,
        expert_counts,
        expert_offsets,
        sorted_route_positions,
        block_map,
        flat_token_idx,
        flat_topk_w,
        output_fp32,
        i,
        h,
        apply_clamp,
        clamp_limit if apply_clamp else 0.0,
        gate_up.stride(0),
        gate_up.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        output_fp32.stride(0),
        output_fp32.stride(1),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=4,
    )


def routed_offsets_stage1_into(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_route_positions: torch.Tensor,
    flat_token_idx: torch.Tensor,
    gate_up: torch.Tensor,
    *,
    max_tiles_per_expert: int,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 128,
) -> None:
    w1_scale = _scale_as_u8(w1_scale)
    local_experts, two_n, k_packed = w1.shape
    k = hidden_states.shape[1]
    assert k_packed * 2 == k
    assert gate_up.shape[1] == two_n
    grid = (local_experts, max_tiles_per_expert, triton.cdiv(two_n, block_n))
    _stage1_offsets_kernel[grid](
        hidden_states,
        w1,
        w1_scale,
        expert_counts,
        expert_offsets,
        sorted_route_positions,
        flat_token_idx,
        gate_up,
        k,
        two_n,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(1),
        w1.stride(2),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def routed_offsets_stage2_scatter_into(
    gate_up: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_route_positions: torch.Tensor,
    flat_token_idx: torch.Tensor,
    flat_topk_w: torch.Tensor,
    output_fp32: torch.Tensor,
    *,
    max_tiles_per_expert: int,
    clamp_limit: float,
    block_m: int = 16,
    block_h: int = 16,
    block_i: int = 128,
) -> None:
    w2_scale = _scale_as_u8(w2_scale)
    local_experts, h, i_packed = w2.shape
    i = i_packed * 2
    assert gate_up.shape[1] == i * 2
    assert output_fp32.shape[1] == h
    output_fp32.zero_()
    apply_clamp = clamp_limit < float("inf")
    grid = (local_experts, max_tiles_per_expert, triton.cdiv(h, block_h))
    _stage2_offsets_scatter_kernel[grid](
        gate_up,
        w2,
        w2_scale,
        expert_counts,
        expert_offsets,
        sorted_route_positions,
        flat_token_idx,
        flat_topk_w,
        output_fp32,
        i,
        h,
        apply_clamp,
        clamp_limit if apply_clamp else 0.0,
        gate_up.stride(0),
        gate_up.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        output_fp32.stride(0),
        output_fp32.stride(1),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=4,
    )


def routed_offsets_stage2_store_route_into(
    gate_up: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_route_positions: torch.Tensor,
    flat_topk_w: torch.Tensor,
    route_out: torch.Tensor,
    *,
    max_tiles_per_expert: int,
    clamp_limit: float,
    block_m: int = 16,
    block_h: int = 16,
    block_i: int = 128,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    w2_scale = _scale_as_u8(w2_scale)
    local_experts, h, i_packed = w2.shape
    i = i_packed * 2
    assert gate_up.shape[1] == i * 2
    assert route_out.shape[1] == h
    route_out.zero_()
    apply_clamp = clamp_limit < float("inf")
    grid = (local_experts, max_tiles_per_expert, triton.cdiv(h, block_h))
    _stage2_offsets_store_route_kernel[grid](
        gate_up,
        w2,
        w2_scale,
        expert_counts,
        expert_offsets,
        sorted_route_positions,
        flat_topk_w,
        route_out,
        i,
        h,
        apply_clamp,
        clamp_limit if apply_clamp else 0.0,
        gate_up.stride(0),
        gate_up.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        route_out.stride(0),
        route_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def reduce_route_out_topk_into(
    route_out: torch.Tensor,
    output_fp32: torch.Tensor,
    *,
    M: int,
    topk: int,
    block_h: int = 16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    h = output_fp32.shape[1]
    assert route_out.shape[0] == M * topk
    assert route_out.shape[1] == h
    grid = (M, triton.cdiv(h, block_h))
    _reduce_route_out_kernel[grid](
        route_out,
        output_fp32,
        M,
        topk,
        h,
        route_out.stride(0),
        route_out.stride(1),
        output_fp32.stride(0),
        output_fp32.stride(1),
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def reduce_route_out_expert_order_into(
    route_out: torch.Tensor,
    flat_topk_ids: torch.Tensor,
    output_fp32: torch.Tensor,
    *,
    M: int,
    topk: int,
    block_h: int = 16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    h = output_fp32.shape[1]
    assert route_out.shape[0] == M * topk
    assert route_out.shape[1] == h
    assert flat_topk_ids.numel() == M * topk
    grid = (M, triton.cdiv(h, block_h))
    if topk == 8:
        _reduce_route_out_expert_order_topk8_kernel[grid](
            route_out,
            flat_topk_ids,
            output_fp32,
            M,
            h,
            route_out.stride(0),
            route_out.stride(1),
            output_fp32.stride(0),
            output_fp32.stride(1),
            BLOCK_H=block_h,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return
    _reduce_route_out_expert_order_kernel[grid](
        route_out,
        flat_topk_ids,
        output_fp32,
        M,
        topk,
        h,
        route_out.stride(0),
        route_out.stride(1),
        output_fp32.stride(0),
        output_fp32.stride(1),
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
