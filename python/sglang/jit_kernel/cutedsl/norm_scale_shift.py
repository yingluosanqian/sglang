import cutlass
import cutlass.cute as cute
from typing import Optional, Tuple
import torch
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from sglang.jit_kernel.cutedsl.common.norm_fusion import (
    apply_norm,
    preprocess_tensor,
    tensor_slice,
)


class NormScaleShift:
    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mWeight: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        mScale: cute.Tensor,
        mShift: cute.Tensor,
        norm_type: cutlass.Constexpr = "rms",
        eps: cutlass.Float32 = cutlass.Float32(1e-6),
        stream: cuda.CUstream = cuda.CUstream(
            cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        shape: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Constexpr] = mX.shape
        self.B, self.S, self.D = shape
        self.F = cutlass.Int32(1)
        if cutlass.const_expr(len(mScale.shape) == 4):
            self.F = mScale.shape[1]
        if cutlass.const_expr(len(mShift.shape) == 4):
            self.F = mShift.shape[1]
        self.len_f = cutlass.Int32(self.S // self.F)
        self.num_threads: cutlass.Constexpr = self.D // 256 * 32
        self.norm_type = norm_type
        num_vectorized = 8

        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        t_layout = cute.make_layout(self.num_threads)
        v_layout = cute.make_layout(num_vectorized)
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)

        # Launch
        self.kernel(
            mY,
            mX,
            mWeight,
            mBias,
            mScale,
            mShift,
            tiled_copy,
            eps,
        ).launch(
            grid=[self.B * self.S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mWeight: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        mScale: cute.Tensor,
        mShift: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        eps: cutlass.Float32,
    ):
        tidx  = cutlass.Int32(cute.arch.thread_idx()[0])
        bid = cutlass.Int32(cute.arch.block_idx()[0])
        bidx = cutlass.Int32(bid // self.S)
        bidy = cutlass.Int32(bid % self.S)

        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def currying_slice(mV):
            if cutlass.const_expr(isinstance(mV, cute.Tensor)):
                return tensor_slice(mV, thr_copy, bidx, bidy, self.len_f)
            else:
                return mV, mV
            
        @cute.jit
        def currying_copy(src, dst):
            if cutlass.const_expr(isinstance(src, cute.Tensor) & isinstance(src, cute.Tensor)):
                cute.autovec_copy(src, dst)

        @cute.jit
        def currying_norm(x, weight, bias):
            return apply_norm(self.norm_type, self.num_threads, tidx, x, weight, bias, self.D, eps)

        tXgX, tXrX = currying_slice(mX)   
        tWgW, tWrW = currying_slice(mWeight)
        tBgB, tBrB = currying_slice(mBias)
        tSCgSC, tSCrSC = currying_slice(mScale)  
        tSHgSH, tSHrSH = currying_slice(mShift)  
        tYgY, tYrY = currying_slice(mY)          

        currying_copy(tXgX, tXrX)
        currying_copy(tWgW, tWrW)
        currying_copy(tBgB, tBrB)
        tNrN = currying_norm(tXrX, tWrW, tBrB)
        currying_copy(tSCgSC, tSCrSC)
        currying_copy(tSHgSH, tSHrSH)
        res = (1 + tSCrSC.load()) * tNrN.load() + tSHrSH.load()
        tYrY.store(res.to(tYrY.element_type))
        currying_copy(tYrY, tYgY)


def fused_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm(x, weight, bias) followed by fused scale/shift.
    or RMSNorm(x, weight) followed by fused scale/shift.

    Expects:
      - x: [B, S, D], contiguous on last dim
      - weight/bias: None, [D]
      - scale/shift: [D], [1/B, D],  [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    Supported D values (must be 256's multiple and <= 8192, etc.
    """
    if norm_type == "layer" or norm_type == "rms":
        head_dim = x.shape[-1]
        if head_dim % 256 != 0 or head_dim > 8192:
            raise ValueError(
                f"D={head_dim} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)
        scale, shift = (
            preprocess_tensor(ten, *x.shape)
            for ten in (scale, shift)
        )
        mY, mX, mWeight, mBias, mScale, mShift = [
            from_dlpack(ten, assumed_align=16) if isinstance(ten, torch.Tensor) else ten
            for ten in (y, x, weight, bias, scale, shift)
        ]
        kernel = NormScaleShift()
        compiled_fn = cute.compile(
            kernel, mY, mX, mWeight, mBias, mScale, mShift, norm_type)
        compiled_fn(mY, mX, mWeight, mBias, mScale, mShift, eps=eps)
        return y
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')
