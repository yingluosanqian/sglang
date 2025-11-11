#!/usr/bin/env python3

import argparse
import os
import time

import torch
from torch.utils.cpp_extension import load

SRC_PATH = os.path.join(os.path.dirname(__file__), "fused.cu")
EXT = load(name="fused_scale_residual", sources=[SRC_PATH], verbose=False)
print("compilation done.")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def expand_gate(gate, b, s, d):
    if gate is None:
        return torch.ones(b, s, d, device="cuda", dtype=torch.float32)
    g = gate.float()
    if g.dim() == 3:
        return g.expand(b, s, d)
    num_frames = g.size(1)
    frame_len = s // num_frames
    return g.expand(b, num_frames, frame_len, d).reshape(b, s, d)


@torch.compile
def fused_op(x, gate_e, res):
    return x * gate_e + res


def reference(residual, x, gate, norm_weight, norm_bias, scale, shift, eps, rms):
    res = residual.float()
    x = x.float()
    b, s, d = res.shape
    gate_e = expand_gate(gate, b, s, d)
    residual_out = fused_op(x, gate_e, res)

    if rms:
        inv = torch.rsqrt(residual_out.pow(2).mean(-1, keepdim=True) + eps)
        norm = residual_out * inv
    else:
        mean = residual_out.mean(-1, keepdim=True)
        diff = residual_out - mean
        inv = torch.rsqrt(diff.pow(2).mean(-1, keepdim=True) + eps)
        norm = diff * inv
    if not rms and norm_weight is not None and norm_bias is not None:
        norm = fused_op(norm, norm_weight.view(
            1, 1, -1), norm_bias.view(1, 1, -1))
    else:
        if norm_weight is not None:
            norm = norm * norm_weight.view(1, 1, -1)
        if not rms and norm_bias is not None:
            norm = norm + norm_bias.view(1, 1, -1)

    scale_e = scale.float().expand(b, s, d)
    shift_e = shift.float().expand(b, s, d)
    mod = fused_op(norm, (1 + scale_e), shift_e)
    return mod.to(residual.dtype), residual_out.to(residual.dtype)


def time_fn(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def save_result_matrices(
    save_dir, dtype, b, s, d, rms, ref_out, ker_out, save_format="pt"
):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    dtype_name = str(dtype).split(".")[-1]
    suffix = ".pt" if save_format == "pt" else ".txt"
    fname = f"result_dtype-{dtype_name}_rms-{int(rms)}_B{b}_S{s}_D{d}{suffix}"
    fpath = os.path.join(save_dir, fname)
    if save_format == "pt":
        payload = {
            "dtype": str(dtype),
            "B": b,
            "S": s,
            "D": d,
            "rms": rms,
            "reference": ref_out.detach().cpu(),
            "kernel": ker_out.detach().cpu(),
        }
        torch.save(payload, fpath)
    else:
        ref_cpu = ref_out.detach().cpu()
        ker_cpu = ker_out.detach().cpu()
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"dtype={dtype}\nB={b}\nS={s}\nD={d}\nrms={rms}\n")
            f.write("reference:\n")
            f.write(f"{ref_cpu}\n")
            f.write("kernel:\n")
            f.write(f"{ker_cpu}\n")
    print(f"Saved result matrices to {fpath}")


def run_once(dtype, b, s, d, rms, iters, save_dir=None, save_format="pt"):
    residual = torch.randn(b, s, d, device="cuda", dtype=dtype)
    x = torch.randn_like(residual)
    gate = torch.randn(b, 4, 1, d, device="cuda", dtype=torch.float32)
    scale = torch.randn(1, s, d, device="cuda", dtype=torch.float32)
    shift = torch.randn_like(scale)
    weight = torch.randn(d, device="cuda", dtype=torch.float32)
    bias = torch.randn(d, device="cuda", dtype=torch.float32)

    ref_out, ref_res = reference(
        residual, x, gate, weight, bias, scale, shift, 1e-6, rms
    )
    ker_out, ker_res = EXT.forward(
        residual, x, gate, weight, bias, scale, shift, 1e-6, rms
    )
    diff_tensor = (ref_out - ker_out).abs()
    diff_flat = diff_tensor.view(-1)
    max_diff_val, max_diff_idx = diff_flat.max(dim=0)
    max_diff = max_diff_val.item()
    max_diff_pos = tuple(
        idx.item() for idx in torch.unravel_index(max_diff_idx, diff_tensor.shape)
    )
    avg_diff = diff_tensor.mean().item()
    rel_tensor = diff_tensor / (ref_out.abs() + 1e-6)
    rel_flat = rel_tensor.view(-1)
    rel_diff_val, rel_diff_idx = rel_flat.max(dim=0)
    rel_diff = rel_diff_val.item()
    rel_diff_pos = tuple(
        idx.item() for idx in torch.unravel_index(rel_diff_idx, rel_tensor.shape)
    )
    rel_ref_val = ref_out[rel_diff_pos].item()
    rel_diff_avg = rel_tensor.mean().item()

    res_diff_tensor = (ref_res - ker_res).abs()
    res_diff_flat = res_diff_tensor.view(-1)
    resi_max_diff_val, resi_max_diff_idx = res_diff_flat.max(dim=0)
    resi_max_diff = resi_max_diff_val.item()
    resi_max_diff_pos = tuple(
        idx.item()
        for idx in torch.unravel_index(resi_max_diff_idx, res_diff_tensor.shape)
    )
    resi_avg_diff = res_diff_tensor.mean().item()
    res_rel_tensor = res_diff_tensor / (ref_res.abs() + 1e-6)
    res_rel_flat = res_rel_tensor.view(-1)
    resi_rel_diff_val, resi_rel_diff_idx = res_rel_flat.max(dim=0)
    resi_rel_diff = resi_rel_diff_val.item()
    resi_rel_diff_pos = tuple(
        idx.item()
        for idx in torch.unravel_index(resi_rel_diff_idx, res_rel_tensor.shape)
    )
    res_rel_ref_val = ref_res[resi_rel_diff_pos].item()
    resi_rel_diff_avg = res_rel_tensor.mean().item()
    # print(f"max diff position: {max_diff_pos}")
    # print(f"residual max diff position: {resi_max_diff_pos}")
    # print(
    #     f"rel diff position: {rel_diff_pos}, reference value: {rel_ref_val:.6e}, ker_out: {ker_out[rel_diff_pos].item():.6e}, ker_out: {ref_out[rel_diff_pos].item():.6e}"
    # )
    # print(
    #     f"residual rel diff position: {resi_rel_diff_pos}, reference value: {res_rel_ref_val:.6e}"
    # )

    ref_t = time_fn(
        lambda: reference(residual, x, gate, weight,
                          bias, scale, shift, 1e-6, rms),
        iters,
    )
    ker_t = time_fn(
        lambda: EXT.forward(residual, x, gate, weight,
                            bias, scale, shift, 1e-6, rms),
        iters,
    )
    save_result_matrices(
        save_dir, dtype, b, s, d, rms, ref_out, ker_out, save_format=save_format
    )

    # ref_t, ker_t = 1, 1
    return (
        max_diff,
        avg_diff,
        rel_diff,
        rel_diff_avg,
        resi_max_diff,
        resi_avg_diff,
        resi_rel_diff,
        resi_rel_diff_avg,
        ref_t,
        ker_t,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to store the result matrices from each configuration.",
    )
    parser.add_argument(
        "--save-format",
        choices=("pt", "txt"),
        default="pt",
        help="Choose between PyTorch (.pt) or human-readable text (.txt) outputs.",
    )
    args = parser.parse_args()

    configs = [
        (torch.float16, 16, 64, 128),
        (torch.bfloat16, 16, 64, 128),
        (torch.float32, 16, 64, 128),
    ]
    for dtype, b, s, d in configs:
        for rms in (False, True):
            (
                diff,
                diff_avg,
                rel,
                rel_avg,
                resi_max_diff,
                resi_avg_diff,
                resi_rel_diff,
                resi_rel_diff_avg,
                ref_t,
                ker_t,
            ) = run_once(
                dtype,
                b,
                s,
                d,
                rms,
                args.iters,
                save_dir=args.save_dir,
                save_format=args.save_format,
            )
            speedup = ref_t / ker_t if ker_t > 0 else float("inf")
            status = (
                "PASS"
                if all(
                    torch.isfinite(torch.tensor(val))
                    and val < 1e-2
                    for val in (rel, resi_rel_diff)
                )
                else "WARN"
            )
            print(
                f"[{status}] dtype={dtype}, rms={rms}, B={b}, S={s}, D={d}\n"
                f"    output  : max_diff={diff:.3e}, avg_diff={diff_avg:.3e}, rel_diff={rel:.3e}, rel_avg={rel_avg:.3e}\n"
                f"    residual: max_diff={resi_max_diff:.3e}, avg_diff={resi_avg_diff:.3e}, rel_diff={resi_rel_diff:.3e}, rel_avg={resi_rel_diff_avg:.3e}\n"
                f"    timing  : ref={ref_t:.2f} ms, kernel={ker_t:.2f} ms, speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    torch.set_printoptions(threshold=float('inf'))
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required.")
    main()
