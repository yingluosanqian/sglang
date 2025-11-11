

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

struct BroadcastDesc {
  const float *ptr;
  int64_t stride_b;
  int32_t frame_len;
};

constexpr int THREADS_PER_WARP = 32;
constexpr int THREADS_PER_CTA = 256;
constexpr int WARP_PER_CTA = THREADS_PER_CTA / THREADS_PER_WARP;

enum NormType : int {
  LayerNorm = 0,
  RMSNorm = 1,
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <NormType norm_type>
__device__ __forceinline__ void
cta_reduce_sum(int thr_id, float sum, float sum_sq, int D, float eps,
               float *__restrict__ shm_sum, float *__restrict__ shm_sum_sq) {
  const int lane = thr_id & 31;
  const int warp = thr_id >> 5;

  if (lane == 0) {
    if constexpr (norm_type == LayerNorm) {
      shm_sum[warp] = sum;
      shm_sum_sq[warp] = sum_sq;
    } else {
      shm_sum_sq[warp] = sum_sq;
    }
  }
  __syncthreads();

  float cta_sum = 0.f, cta_sum_sq = 0.f;

  if (warp == 0) {
    float v = (lane < WARP_PER_CTA) ? shm_sum_sq[lane] : 0.f;
    cta_sum_sq = warp_reduce_sum(v);

    if constexpr (norm_type == LayerNorm) {
      float u = (lane < WARP_PER_CTA) ? shm_sum[lane] : 0.f;
      cta_sum = warp_reduce_sum(u);
    }
  }

  if (thr_id == 0) {
    if constexpr (norm_type == LayerNorm) {
      float mean = cta_sum / D;
      float var = cta_sum_sq / D - mean * mean;
      shm_sum[0] = mean;
      shm_sum_sq[0] = rsqrtf(var + eps);
    } else {
      shm_sum_sq[0] = rsqrtf(cta_sum_sq / D + eps);
    }
  }
}

/**
 * @brief ScaleResidualLayerNormScaleShift.
 * This CUDA kernel performs a fused sequence of operations commonly used
 * in transformer inference pipelines. The fused computation combines
 * gated residual connection, normalization (LayerNorm or RMSNorm), and
 * scale–shift modulation into a single pass over the data for improved
 * efficiency and reduced memory bandwidth.
 *
 * The computation can be summarized as:
 *   1. residual_output = gate * x + residual
 *   2. norm_x = Norm(residual_output)
 *        where Norm ∈ { LayerNorm, RMSNorm }
 *   3. modulated = norm_x * (1 + scale) + shift
 *
 * @param residual         Input tensor of shape [B, S, D].
 * @param x                Input tensor of shape [B, S, D].
 * @param gate             Gate tensor: [B, 1, D] or [B, num_frames, 1, D].
 * @param norm_weight      Learnable weight (γ) of shape [D].
 * @param norm_bias        Learnable bias (β) of shape [D].
 * @param scale            Scale modulation tensor of shape [B, S, D].
 * @param shift            Shift modulation tensor of shape [B, S, D].
 * @param eps              Numerical epsilon for stability.
 * @param modulated        Output tensor after fused operations [B, S, D].
 * @param residual_output  Intermediate residual output [B, S, D].
 * @param B                Batch size.
 * @param S                Sequence length.
 * @param D                Hidden dimension (normalization size).
 * @param num_frames       Number of temporal frames in gate. When 1, gate is
 * [B, 1, D]; when >1, gate is [B, num_frames, 1, D] and each frame’s gate is
 * broadcast over (S / num_frames) sequence positions.
 *
 */
template <typename ActType, NormType norm_type>
__global__ void fused(const ActType *residual, const ActType *x,
                      const float *gate, const float *norm_weight,
                      const float *norm_bias, BroadcastDesc scale_desc,
                      BroadcastDesc shift_desc, double eps, ActType *modulated,
                      ActType *residual_output, int B, int S, int D,
                      int gate_frame_len) {
  uint32_t cta_id = blockIdx.x;
  uint32_t thr_id = threadIdx.x;

  // ---------------------------------------------------------
  // Pointer offsets for the current (batch, sequence) row
  // ---------------------------------------------------------
  // Each CUDA block (bid) processes one row of length D in [B, S, D].
  // So we advance x/residual/residual_output pointers by bid * D
  // to reach the start of this row.
  const int batch_idx = cta_id / S;
  const int seq_idx = cta_id % S;
  residual += cta_id * D;
  x += cta_id * D;
  gate += cta_id / gate_frame_len * D;
  const float *scale =
      scale_desc.ptr +
      (batch_idx * scale_desc.stride_b + seq_idx) / scale_desc.frame_len * D;
  const float *shift =
      shift_desc.ptr +
      (batch_idx * shift_desc.stride_b + seq_idx) / shift_desc.frame_len * D;
  modulated += cta_id * D;
  residual_output += cta_id * D;

  // 1. Residual Output
  float sum = 0.0, sum_sq = 0.0;
  for (uint32_t i = thr_id; i < D; i += THREADS_PER_CTA) {
    float new_x = fmaf(static_cast<float>(x[i]), gate[i],
                       static_cast<float>(residual[i]));
    // 2. Norm: partial reduce
    if constexpr (norm_type == LayerNorm) {
      sum += new_x;
      sum_sq += new_x * new_x;
    } else if constexpr (norm_type == RMSNorm) {
      sum_sq += new_x * new_x;
    }

    // Output
    residual_output[i] = static_cast<ActType>(new_x);
  }
  __syncthreads();

  // 2. Norm: reduce
  // shm_sum[0]    = mean   (LayerNorm only, unused for RMS)
  // shm_sum_sq[0] = inv    (inv_std or inv_rms)
  __shared__ float shm_sum[WARP_PER_CTA];
  __shared__ float shm_sum_sq[WARP_PER_CTA];
  // warp-level reduce
  if constexpr (norm_type == LayerNorm) {
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);
  } else {
    sum_sq = warp_reduce_sum(sum_sq);
  }
  // cta-level reduce
  cta_reduce_sum<norm_type>(thr_id, sum, sum_sq, D, eps, shm_sum, shm_sum_sq);
  __syncthreads();

  // 2. Norm: element-wise
  for (int i = thr_id; i < D; i += THREADS_PER_CTA) {
    float new_x = static_cast<float>(residual_output[i]);
    float norm_x;
    if constexpr (norm_type == LayerNorm) {
      float inv = shm_sum_sq[0];
      norm_x = (new_x - shm_sum[0]) * inv;
      norm_x = fmaf(norm_weight[i], norm_x, norm_bias[i]);
    } else if constexpr (norm_type == RMSNorm) {
      float inv = shm_sum_sq[0];
      norm_x = norm_weight[i] * new_x * inv;
    }
    // 3. Modulate
    float mod = fmaf(norm_x, (1.0f + scale[i]), shift[i]);
    modulated[i] = static_cast<ActType>(mod);
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DIM(x, n)                                                        \
  TORCH_CHECK((x).dim() == (n), #x " must have " #n " dimensions")

namespace {

struct BroadcastParam {
  at::Tensor tensor;
  BroadcastDesc desc;
};

at::Tensor ensure_bias(const c10::optional<at::Tensor> &bias_opt, int64_t D,
                       const at::TensorOptions &options) {
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    return bias_opt.value().contiguous().to(options.dtype());
  }
  return at::zeros({D}, options);
}

/*
 * Canonicalize scale/shift tensors of varying rank into contiguous fp32
 * storage, computing strides and optional frame metadata for use inside
 * the CUDA kernel. Supports scalar, [D], [B,D], [B,S,D], and [B,F,1,D].
 */
BroadcastParam prepare_broadcast_tensor(const at::Tensor &tensor, int64_t B,
                                        int64_t S, int64_t D) {
  BroadcastParam param;
  TORCH_CHECK(tensor.defined(), "Tensor must be defined.");
  auto t = tensor.to(torch::kFloat);
  param.desc.frame_len = S;

  const int64_t ndim = t.dim();
  if (ndim == 0) {
    // (scalar) -> layout(shape=(1,1,1), stride=(0,0,0))
    t = t.reshape({1, 1, 1});
    param.desc.stride_b = 0;
  } else if (ndim == 1) {
    // (D) -> layout(shape=(1,1,D), stride=(0,0,1))
    TORCH_CHECK(t.size(0) == D, "Expected shape [D] for broadcast tensor.");
    t = t.reshape({1, 1, D});
    param.desc.stride_b = 0;
  } else if (ndim == 2) {
    // (B,D) -> layout(shape=(B,1,D), stride=(stride_B,0,1))
    // (1,D) -> layout(shape=(1,1,D), stride=(0,0,1))
    TORCH_CHECK(t.size(1) == D, "Trailing dim must match hidden size.");
    TORCH_CHECK(t.size(0) == B || t.size(0) == 1,
                "Leading dim must be batch size or 1.");
    param.desc.stride_b = t.size(0) == B ? S : 0;
    param.desc.frame_len = S;
    t = t.reshape({t.size(0), 1, D});
  } else if (ndim == 3) {
    // (B,S,D), (B,1,D), (1,S,D), (1,1,D) -> layout(shape=(B,S,D), stride=...)
    TORCH_CHECK(t.size(2) == D, "Trailing dim must match hidden size.");
    TORCH_CHECK(t.size(0) == B || t.size(0) == 1,
                "Leading dim must be batch size or 1.");
    TORCH_CHECK(t.size(1) == S || t.size(1) == 1,
                "Middle dim must be sequence length or 1.");
    param.desc.stride_b = t.size(0) == B ? S : 0;
    param.desc.frame_len = S / t.size(1);
  } else if (ndim == 4) {
    // (B,F,1,D) -> (B,F,D)
    TORCH_CHECK(t.size(2) == 1 && t.size(3) == D,
                "Expected [B,F,1,D] for frame broadcast.");
    TORCH_CHECK(t.size(0) == B || t.size(0) == 1,
                "Leading dim must be batch size or 1.");
    auto num_frames = t.size(1);
    TORCH_CHECK(S % num_frames == 0,
                "Sequence length must be divisible by num_frames.");
    t = t.reshape({t.size(0), num_frames, D});
    param.desc.stride_b = S;
    param.desc.frame_len = S / num_frames;
  } else {
    TORCH_CHECK(false, "Unsupported rank for broadcast tensor.");
  }

  t = t.contiguous();
  param.tensor = t;
  param.desc.ptr = t.data_ptr<float>();
  return param;
}

struct GateParam {
  at::Tensor storage;
  int64_t frame_len;
};

/**
 * @brief Normalize the optional gate tensor to contiguous fp32 storage with
 * explicit frame metadata. Accepts [B,1,D] or [B,F,1,D]; falls back to ones
 * when gate_opt is not provided.
 */
GateParam prepare_gate(const c10::optional<at::Tensor> &gate_opt, int64_t B,
                       int64_t S, int64_t D,
                       const at::TensorOptions &float_opts) {
  GateParam gate_param;
  int64_t num_frames = 1;
  at::Tensor gate_prepared;
  if (gate_opt.has_value() && gate_opt.value().defined()) {
    const auto &gate = gate_opt.value();
    CHECK_CUDA(gate);
    if (gate.dim() == 3) {
      TORCH_CHECK(gate.size(0) == B, "gate batch size mismatch");
      TORCH_CHECK(gate.size(2) == D, "gate hidden size mismatch");
      TORCH_CHECK(gate.size(1) == 1, "gate tensor must be [B,1,D]");
      gate_prepared = gate.contiguous().to(torch::kFloat).view({B, 1, D});
    } else if (gate.dim() == 4) {
      TORCH_CHECK(gate.size(0) == B, "gate batch size mismatch");
      TORCH_CHECK(gate.size(3) == D, "gate hidden size mismatch");
      TORCH_CHECK(gate.size(2) == 1, "gate tensor must be [B,F,1,D]");
      num_frames = gate.size(1);
      TORCH_CHECK(S % num_frames == 0,
                  "sequence length must be divisible by num_frames");
      gate_prepared =
          gate.contiguous().to(torch::kFloat).view({B, num_frames, 1, D});
    } else {
      TORCH_CHECK(false, "gate tensor must be rank 3 or 4");
    }
  } else {
    gate_prepared = at::ones({B, 1, D}, float_opts);
  }
  gate_param.storage = gate_prepared.view({B * num_frames, D}).contiguous();
  gate_param.frame_len = S / num_frames;
  return gate_param;
}

struct NormParams {
  at::Tensor weight;
  at::Tensor bias;
};

/**
 * @brief Canonicalize optional norm weight/bias tensors: ensure they live on
 * CUDA, have length D, and are contiguous fp32 buffers (falling back to
 * ones/zeros when not provided).
 */
NormParams prepare_norm_params(const c10::optional<at::Tensor> &weight_opt,
                               const c10::optional<at::Tensor> &bias_opt,
                               int64_t D, const at::TensorOptions &float_opts) {
  NormParams params;
  if (weight_opt.has_value() && weight_opt.value().defined()) {
    const auto &norm_weight = weight_opt.value();
    CHECK_CUDA(norm_weight);
    TORCH_CHECK(norm_weight.numel() == D, "norm_weight must have length D");
    params.weight = norm_weight.contiguous().to(torch::kFloat);
  } else {
    params.weight = at::ones({D}, float_opts);
  }
  params.bias = ensure_bias(bias_opt, D, float_opts);
  return params;
}

} // namespace

/*==========================================================================*
 *  Public entry point invoked from Python. It validates inputs, prepares   *
 *  all broadcast buffers (gate/norm/scale/shift), and dispatches the CUDA  *
 *  kernel that fuses gate + normalization + scale/shift.                   *
 *==========================================================================*/
std::tuple<at::Tensor, at::Tensor> fused_scale_residual_norm_scale_shift(
    const at::Tensor &residual, const at::Tensor &x,
    const c10::optional<at::Tensor> &gate_opt,
    const c10::optional<at::Tensor> &norm_weight_opt,
    const c10::optional<at::Tensor> &norm_bias_opt, const at::Tensor &scale,
    const at::Tensor &shift, double eps, bool use_rms_norm) {
  // --- basic input validation ---
  CHECK_CUDA(residual);
  CHECK_CUDA(x);
  CHECK_CUDA(scale);
  CHECK_CUDA(shift);
  TORCH_CHECK(residual.dtype() == x.dtype(), "residual and x must share dtype");
  TORCH_CHECK(residual.dim() == 3, "residual must be [B, S, D]");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x must match residual shape");

  const auto B = residual.size(0);
  const auto S = residual.size(1);
  const auto D = residual.size(2);
  auto residual_f = residual.contiguous();
  auto x_f = x.contiguous();
  const auto float_opts =
      torch::TensorOptions().device(residual.device()).dtype(torch::kFloat);

  // --- preprocess gate, norm, scale, shift, parameters ---
  auto gate_param = prepare_gate(gate_opt, B, S, D, float_opts);

  auto norm_params =
      prepare_norm_params(norm_weight_opt, norm_bias_opt, D, float_opts);
  auto scale_param = prepare_broadcast_tensor(scale, B, S, D);
  auto shift_param = prepare_broadcast_tensor(shift, B, S, D);

  // --- allocate outputs ---
  auto modulated = at::empty_like(residual_f);
  auto residual_output = at::empty_like(residual_f);

  // --- configure kernel launch ---
  dim3 block(THREADS_PER_CTA);
  dim3 grid(B * S);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto launch = [&](auto type_tag) {
    using scalar_t_ = decltype(type_tag);
    if (use_rms_norm) {
      fused<scalar_t_, NormType::RMSNorm><<<grid, block, 0, stream>>>(
          residual_f.data_ptr<scalar_t_>(), x_f.data_ptr<scalar_t_>(),
          gate_param.storage.data_ptr<float>(),
          norm_params.weight.data_ptr<float>(),
          norm_params.bias.data_ptr<float>(), scale_param.desc,
          shift_param.desc, eps, modulated.data_ptr<scalar_t_>(),
          residual_output.data_ptr<scalar_t_>(), B, S, D,
          static_cast<int>(gate_param.frame_len));
    } else {
      fused<scalar_t_, NormType::LayerNorm><<<grid, block, 0, stream>>>(
          residual_f.data_ptr<scalar_t_>(), x_f.data_ptr<scalar_t_>(),
          gate_param.storage.data_ptr<float>(),
          norm_params.weight.data_ptr<float>(),
          norm_params.bias.data_ptr<float>(), scale_param.desc,
          shift_param.desc, eps, modulated.data_ptr<scalar_t_>(),
          residual_output.data_ptr<scalar_t_>(), B, S, D,
          static_cast<int>(gate_param.frame_len));
    }
  };

  switch (residual.scalar_type()) {
  case at::ScalarType::Float:
    launch(float{});
    break;
  case at::ScalarType::Half:
    launch(at::Half{});
    break;
  case at::ScalarType::BFloat16:
    launch(at::BFloat16{});
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype for fused kernel.");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

  auto orig_dtype = residual.dtype();
  return {modulated.to(orig_dtype), residual_output.to(orig_dtype)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_scale_residual_norm_scale_shift,
        "Fused scale residual layernorm scale shift (CUDA)");
}
