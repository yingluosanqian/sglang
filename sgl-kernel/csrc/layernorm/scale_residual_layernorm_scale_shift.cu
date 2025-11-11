/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Normalization.h>

#include <tuple>

namespace {

inline void check_3d_cuda_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cuda(), name, " must be on CUDA");
  TORCH_CHECK_EQ(tensor.dim(), 3, name, " must be rank-3 [B, L, H]");
  TORCH_CHECK(
      tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat,
      name,
      " supports fp16/bf16/fp32 tensors only.");
}

inline at::Tensor ensure_contiguous_fp(const at::Tensor& tensor, c10::ScalarType dtype) {
  if (!tensor.defined()) {
    return tensor;
  }
  at::Tensor output = tensor;
  if (tensor.scalar_type() != dtype) {
    output = tensor.to(dtype);
  }
  if (!output.is_contiguous()) {
    output = output.contiguous();
  }
  return output;
}

at::Tensor expand_temporal_tensor(
    const at::Tensor& tensor,
    int64_t batch,
    int64_t seq_len,
    int64_t hidden,
    const char* name) {
  TORCH_CHECK(tensor.device().is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(
      tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat,
      name,
      " supports fp16/bf16/fp32 tensors only.");

  at::Tensor expanded;
  if (tensor.dim() == 4) {
    TORCH_CHECK_EQ(tensor.size(0), batch, name, " batch mismatch.");
    TORCH_CHECK_EQ(tensor.size(2), 1, name, " expected singleton temporal dim at dim=2.");
    TORCH_CHECK_EQ(tensor.size(3), hidden, name, " hidden size mismatch.");
    const int64_t num_frames = tensor.size(1);
    TORCH_CHECK(num_frames > 0, name, " invalid num_frames.");
    TORCH_CHECK(seq_len % num_frames == 0, name, " seq_len must be divisible by num_frames.");
    const int64_t frame_seqlen = seq_len / num_frames;
    expanded = tensor.squeeze(2).unsqueeze(2).expand({batch, num_frames, frame_seqlen, hidden}).reshape(
        {batch, seq_len, hidden});
  } else if (tensor.dim() == 3) {
    TORCH_CHECK_EQ(tensor.size(0), batch, name, " batch mismatch.");
    TORCH_CHECK_EQ(tensor.size(2), hidden, name, " hidden size mismatch.");
    if (tensor.size(1) == seq_len) {
      expanded = tensor;
    } else {
      TORCH_CHECK_EQ(tensor.size(1), 1, name, " temporal dim must be 1 or seq_len.");
      expanded = tensor.expand({batch, seq_len, hidden});
    }
  } else if (tensor.dim() == 2) {
    TORCH_CHECK(
        tensor.size(0) == batch || tensor.size(0) == 1,
        name,
        " dim-0 must be broadcastable to batch.");
    TORCH_CHECK_EQ(tensor.size(1), hidden, name, " hidden size mismatch.");
    expanded = tensor.unsqueeze(1).expand({batch, seq_len, hidden});
  } else if (tensor.dim() == 1) {
    TORCH_CHECK_EQ(tensor.size(0), hidden, name, " hidden size mismatch.");
    expanded = tensor.view({1, 1, hidden}).expand({batch, seq_len, hidden});
  } else if (tensor.dim() == 0) {
    expanded = tensor.view({1, 1, 1}).expand({batch, seq_len, hidden});
  } else {
    TORCH_CHECK(false, name, " rank ", tensor.dim(), " tensors are not supported.");
  }

  return expanded.contiguous();
}

at::Tensor apply_gate(
    const at::Tensor& residual,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& gate_opt) {
  if (!gate_opt.has_value()) {
    return residual + x;
  }

  const auto gate = gate_opt.value();
  const auto batch = residual.size(0);
  const auto seq_len = residual.size(1);
  const auto hidden = residual.size(2);

  auto expanded_gate = expand_temporal_tensor(gate, batch, seq_len, hidden, "gate");
  expanded_gate = expanded_gate.to(x.dtype());

  return residual + x * expanded_gate;
}

at::Tensor apply_norm(
    const at::Tensor& residual_output,
    const c10::optional<at::Tensor>& norm_weight_opt,
    const c10::optional<at::Tensor>& norm_bias_opt,
    double eps,
    bool is_rms_norm,
    bool force_fp32_norm) {
  const auto orig_dtype = residual_output.scalar_type();
  const bool need_cast =
      force_fp32_norm && (orig_dtype == at::kHalf || orig_dtype == at::kBFloat16 || orig_dtype == at::kFloat16);
  auto working = need_cast ? residual_output.to(at::kFloat) : residual_output;
  if (!working.is_contiguous()) {
    working = working.contiguous();
  }

  if (is_rms_norm) {
    at::Tensor normalized = working;
    normalized = normalized * at::rsqrt(normalized.pow(2).mean(-1, true) + eps);
    if (norm_weight_opt.has_value()) {
      auto weight = norm_weight_opt.value();
      TORCH_CHECK_EQ(weight.size(-1), working.size(-1), "RMSNorm weight must match hidden size.");
      if (weight.dim() == 1) {
        weight = weight.view({1, 1, weight.size(0)});
      }
      weight = weight.to(working.dtype());
      normalized = normalized * weight;
    }
    if (need_cast) {
      normalized = normalized.to(orig_dtype);
    }
    return normalized;
  }

  c10::optional<at::Tensor> weight_opt;
  c10::optional<at::Tensor> bias_opt;
  if (norm_weight_opt.has_value()) {
    weight_opt = ensure_contiguous_fp(norm_weight_opt.value(), working.scalar_type());
  }
  if (norm_bias_opt.has_value()) {
    bias_opt = ensure_contiguous_fp(norm_bias_opt.value(), working.scalar_type());
  }

  auto normalized =
      std::get<0>(at::native_layer_norm(working, {working.size(-1)}, weight_opt, bias_opt, eps, false));
  if (need_cast) {
    normalized = normalized.to(orig_dtype);
  }
  return normalized;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> scale_residual_layernorm_scale_shift(
    const at::Tensor& residual,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& gate,
    const c10::optional<at::Tensor>& norm_weight,
    const c10::optional<at::Tensor>& norm_bias,
    const at::Tensor& shift,
    const at::Tensor& scale,
    double eps,
    bool is_rms_norm,
    bool force_fp32_norm) {
  check_3d_cuda_tensor(residual, "residual");
  check_3d_cuda_tensor(x, "x");
  TORCH_CHECK(
      residual.sizes() == x.sizes(),
      "residual and x must have identical shapes, got ",
      residual.sizes(),
      " vs ",
      x.sizes());

  const auto batch = residual.size(0);
  const auto seq_len = residual.size(1);
  const auto hidden = residual.size(2);

  auto residual_output = apply_gate(residual, x, gate);

  auto normalized = apply_norm(residual_output, norm_weight, norm_bias, eps, is_rms_norm, force_fp32_norm);

  auto expanded_scale = expand_temporal_tensor(scale, batch, seq_len, hidden, "scale").to(normalized.dtype());
  auto expanded_shift = expand_temporal_tensor(shift, batch, seq_len, hidden, "shift").to(normalized.dtype());

  auto modulated = normalized * (1 + expanded_scale) + expanded_shift;
  return std::make_tuple(modulated.contiguous(), residual_output.contiguous());
}
