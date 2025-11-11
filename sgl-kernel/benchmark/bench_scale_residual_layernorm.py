#!/usr/bin/env python3
"""
Benchmark + correctness test for scale_residual_layernorm_scale_shift kernel.

Usage examples:
  python bench_scale_residual_layernorm.py --verify_only
  python bench_scale_residual_layernorm.py --batch_sizes 1,2 --seq_lens 64 --dims 2048 --iters 2000
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from typing import Iterable, Tuple

import torch
import sgl_kernel

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.view(1, 1, -1)
    return y


def reference_impl(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    norm_type: str,
    force_fp32_norm: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert norm_type in {"rms", "layer"}

    def expand(t: torch.Tensor, is_gate: bool = False) -> torch.Tensor:
        if t.dim() == 4:
            b, f, one, h = t.shape
            assert one == 1
            frame_len = residual.size(1) // f
            return t.squeeze(2).unsqueeze(2).expand(b, f, frame_len, h).reshape(residual.shape)
        return t.expand(residual.shape) if t.dim() < 3 else t

    res_out = residual + (x if gate is None else x * expand(gate, True))
    work = res_out
    if force_fp32_norm and work.dtype in {torch.float16, torch.bfloat16}:
        work = work.float()

    if norm_type == "rms":
        norm = rms_norm(work, norm_weight, eps)
    else:
        norm = torch.nn.functional.layer_norm(
            work,
            (work.size(-1),),
            weight=norm_weight,
            bias=norm_bias,
            eps=eps,
        )
    if force_fp32_norm and res_out.dtype != torch.float32:
        norm = norm.to(res_out.dtype)

    shift_e = expand(shift)
    scale_e = expand(scale)
    modulated = norm * (1 + scale_e) + shift_e
    return modulated.contiguous(), res_out.contiguous()


def random_inputs(
    batch: int,
    seq: int,
    hidden: int,
    dtype: torch.dtype,
    norm_type: str,
    gate_mode: str,
    device: torch.device,
) -> tuple:
    residual = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    x = torch.randn_like(residual)
    gate = None
    if gate_mode == "scalar":
        gate = torch.randn(batch, 1, hidden, device=device, dtype=dtype)
    elif gate_mode == "frame":
        frames = max(1, seq // 4)
        gate = torch.randn(batch, frames, 1, hidden, device=device, dtype=dtype)
    elif gate_mode == "full":
        gate = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    shift = torch.randn(batch, 1, hidden, device=device, dtype=dtype)
    scale = torch.randn_like(shift)

    norm_weight = torch.randn(hidden, device=device, dtype=torch.float32)
    norm_bias = torch.randn_like(norm_weight) if norm_type == "layer" else None
    return residual, x, gate, norm_weight, norm_bias, shift, scale


def check_correctness(configs: Iterable[tuple], norm_types: Iterable[str], gate_modes: Iterable[str], eps: float):
    op = torch.ops.sgl_kernel.scale_residual_layernorm_scale_shift
    device = torch.device("cuda")
    for (batch, seq, hidden, dtype), norm_type, gate_mode in itertools.product(
        configs, norm_types, gate_modes
    ):
        residual, x, gate, norm_weight, norm_bias, shift, scale = random_inputs(
            batch, seq, hidden, dtype, norm_type, gate_mode, device
        )
        ref_out, ref_res = reference_impl(
            residual,
            x,
            gate,
            norm_weight,
            norm_bias,
            shift,
            scale,
            eps,
            norm_type,
            force_fp32_norm=True,
        )
        op_out, op_res = op(
            residual,
            x,
            gate,
            norm_weight,
            norm_bias,
            shift,
            scale,
            eps,
            norm_type == "rms",
            True,
        )
        rtol = 1e-2 if dtype in {torch.float16, torch.bfloat16} else 1e-4
        atol = 1e-3 if dtype in {torch.float16, torch.bfloat16} else 1e-5
        if not torch.allclose(ref_out, op_out, rtol=rtol, atol=atol):
            raise AssertionError(
                f"Output mismatch for config (B={batch}, L={seq}, H={hidden}, dtype={dtype}, "
                f"norm={norm_type}, gate={gate_mode})"
            )
        if not torch.allclose(ref_res, op_res, rtol=rtol, atol=atol):
            raise AssertionError("Residual mismatch for same config")
        print(
            f"[correctness] B={batch:3d} L={seq:4d} H={hidden:5d} "
            f"{str(dtype):9s} norm={norm_type:5s} gate={gate_mode:6s} ✅"
        )


def time_fn(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def benchmark(configs: Iterable[tuple], norm_type: str, gate_mode: str, eps: float, iters: int):
    op = torch.ops.sgl_kernel.scale_residual_layernorm_scale_shift
    device = torch.device("cuda")
    print(f"\n[benchmark] norm={norm_type}, gate={gate_mode}, eps={eps}, iters={iters}")
    for batch, seq, hidden, dtype in configs:
        residual, x, gate, norm_weight, norm_bias, shift, scale = random_inputs(
            batch, seq, hidden, dtype, norm_type, gate_mode, device
        )

        def ref_call():
            reference_impl(
                residual,
                x,
                gate,
                norm_weight,
                norm_bias,
                shift,
                scale,
                eps,
                norm_type,
                True,
            )

        def op_call():
            op(
                residual,
                x,
                gate,
                norm_weight,
                norm_bias,
                shift,
                scale,
                eps,
                norm_type == "rms",
                True,
            )

        t_ref = time_fn(ref_call, iters)
        t_op = time_fn(op_call, iters)
        print(
            f"B={batch:3d} L={seq:4d} H={hidden:5d} {str(dtype):9s} "
            f"ref={t_ref:7.2f} ms  kernel={t_op:7.2f} ms  speedup={t_ref / t_op:5.2f}x"
        )


def str2list(arg: str) -> list[int]:
    if not arg:
        return []
    return [int(x) for x in arg.split(",")]


def main():
    parser = argparse.ArgumentParser("scale_residual_layernorm_scale_shift benchmark")
    parser.add_argument("--batch_sizes", type=str, default="")
    parser.add_argument("--seq_lens", type=str, default="")
    parser.add_argument("--dims", type=str, default="")
    parser.add_argument("--dtypes", type=str, default="fp16,bf16")
    parser.add_argument("--gate_modes", type=str, default="none,scalar")
    parser.add_argument("--norm_types", type=str, default="rms")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    batch_sizes = str2list(args.batch_sizes) or ([1, 2] if IS_CI else [1, 2, 4])
    seq_lens = str2list(args.seq_lens) or ([16] if IS_CI else [16, 64, 256])
    dims = str2list(args.dims) or ([512] if IS_CI else [1024, 2048, 4096])
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtypes = [dtype_map[name] for name in args.dtypes.split(",")]
    gate_modes = args.gate_modes.split(",")
    norm_types = args.norm_types.split(",")

    configs = list(itertools.product(batch_sizes, seq_lens, dims, dtypes))
    check_correctness(configs, norm_types, gate_modes, args.eps)
    if not args.verify_only:
        benchmark(configs, norm_types[0], gate_modes[0], args.eps, args.iters)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this benchmark")
    main()
