#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/impl/norm.cuh>
#include <sgl_kernel/impl/norm_fusion.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cuda_fp16.h>
#include <type_traits>

namespace {

using host::norm::NormEnum;
using host::norm_fusion::IndexEnum;

template <typename T>
struct ParamStorageTraits {
  static constexpr bool kIsFp32 = std::is_same_v<T, fp32_t>;
  using PairT = std::conditional_t<kIsFp32, fp32x2_t, packed_t<T>>;
  static constexpr int kPairsPerStorage = kIsFp32 ? 2 : 4;
  using Storage = AlignedVector<PairT, kPairsPerStorage>;
};

template <typename Traits>
SGL_DEVICE fp32x2_t
get_param_pair(const typename Traits::Storage& vec0, const typename Traits::Storage& vec1, int pair_idx) {
  if constexpr (Traits::kIsFp32) {
    return pair_idx < 2 ? vec0[pair_idx] : vec1[pair_idx - 2];
  } else {
    return cast<fp32x2_t>(vec0[pair_idx]);
  }
}

template <typename T>
struct ParamLoader {
  using Traits = ParamStorageTraits<T>;
  using Storage = typename Traits::Storage;
  Storage vec0;
  Storage vec1;

  template <typename Mem>
  SGL_DEVICE void load(const Mem& mem, const T* ptr, int row) {
    vec0 = mem.load(ptr + row, 0);
    if constexpr (Traits::kIsFp32) {
      vec1 = mem.load(ptr + row, 1);
    }
  }

  SGL_DEVICE void fill_scalar(float v) {
    if constexpr (Traits::kIsFp32) {
      vec0.fill(fp32x2_t{v, v});
      vec1.fill(fp32x2_t{v, v});
    } else {
      vec0.fill(cast<packed_t<T>, fp32x2_t>({v, v}));
    }
  }

  SGL_DEVICE fp32x2_t pair(int pair_idx) const {
    return get_param_pair<Traits>(vec0, vec1, pair_idx);
  }
};

template <
    typename XDType,
    typename ScaleDType,
    typename ShiftDType,
    int64_t kDim,
    NormEnum norm_enum,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum>
__global__ void norm_fused_scale_shift_kernel(
    XDType* __restrict__ output,
    const XDType* __restrict__ input,
    const XDType* __restrict__ gamma,
    const XDType* __restrict__ beta,
    const ScaleDType* __restrict__ scale,
    const ShiftDType* __restrict__ shift,
    const int S,
    const int F,
    bool affine,
    float eps) {
  using namespace device;
  using namespace device::norm;
  using XPackedT = packed_t<XDType>;
  using XStorage = norm::StorageType<XDType, kDim>;
  using ScaleTraits = ParamStorageTraits<ScaleDType>;
  using ShiftTraits = ParamStorageTraits<ShiftDType>;
  using ScaleStorage = typename ScaleTraits::Storage;
  using ShiftStorage = typename ShiftTraits::Storage;

  constexpr int kStorageSize = 4;
  __shared__ float smem_buffer[kSmemBufferSize];
  const int bidx = blockIdx.x;
  const int b_id = bidx / S, s_id = bidx % S;
  const auto xmem = tile::Memory<XStorage>::cta();
  const auto scale_mem = tile::Memory<ScaleStorage>::cta();
  const auto shift_mem = tile::Memory<ShiftStorage>::cta();

  // Compute offsets
  const int scale_row = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id);
  const int shift_row = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id);

  // ============ Step 1: normed = norm(input) * gamma + beta ============
  XStorage beta_vec;
  const auto input_vec = xmem.load(input + bidx * kDim);
  const auto gamma_vec = affine ? xmem.load(gamma) : XStorage(cast<XPackedT, fp32x2_t>({1.0f, 1.0f}));
  if constexpr (norm_enum == NormEnum::LayerNorm)
    beta_vec = affine ? xmem.load(beta) : XStorage(cast<XPackedT, fp32x2_t>({0.0f, 0.0f}));
  else
    beta_vec = XStorage(cast<XPackedT, fp32x2_t>({0.0f, 0.0f}));
  const auto norm_output = apply_norm_cta<norm_enum, kDim>(input_vec, gamma_vec, beta_vec, eps, smem_buffer);
  // ============ Step 2: output = normed * (1 + scale) + shift ============
  ParamLoader<ScaleDType> scale_loader;
  ParamLoader<ShiftDType> shift_loader;
  if constexpr (scale_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(scale[0]);
    scale_loader.fill_scalar(s);
  } else {
    scale_loader.load(scale_mem, scale + scale_row * kDim, 0);
  }
  if constexpr (shift_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(shift[0]);
    shift_loader.fill_scalar(s);
  } else {
    shift_loader.load(shift_mem, shift + shift_row * kDim, 0);
  }
  XStorage output_vec;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto norm_fp32 = cast<fp32x2_t>(norm_output[i]);
    auto scale_fp32 = scale_loader.pair(i);
    auto shift_fp32 = shift_loader.pair(i);
    float out_x = norm_fp32.x * (1.0f + scale_fp32.x) + shift_fp32.x;
    float out_y = norm_fp32.y * (1.0f + scale_fp32.y) + shift_fp32.y;
    output_vec[i] = cast<XPackedT, fp32x2_t>({out_x, out_y});
  }
  xmem.store(output + bidx * kDim, output_vec);
}

template <
    NormEnum norm_enum,
    typename XDType,
    typename ScaleDType,
    typename ShiftDType,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    int64_t kDim>
void fused_norm_scale_shift(
    tvm::ffi::TensorView out,
    const tvm::ffi::TensorView x,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gamma_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> beta_opt,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView shift,
    double eps) {
  using namespace host;

  static_assert(
      std::is_same_v<XDType, half> || std::is_same_v<XDType, nv_bfloat16>,
      "Only support fp16, bf16 for norm template version");
  static_assert(
      norm_enum == NormEnum::LayerNorm || norm_enum == NormEnum::RMSNorm, "norm_enum must be layernorm or rmsnorm.");
  static_assert(host::norm::is_config_supported<XDType, kDim>(), "Unsupported norm configuration for kDim");

  host::norm_fusion::Matcher matcher;
  matcher.template match<XDType, IndexEnum::NoBroadcast>(out);
  matcher.template match<XDType, IndexEnum::NoBroadcast>(x);
  matcher.template match<ScaleDType, scale_index_enum>(scale);
  matcher.template match<ShiftDType, shift_index_enum>(shift);
  bool affine = gamma_opt.has_value();
  if (affine) {
    matcher.template match<XDType, IndexEnum::BroadcastBS>(gamma_opt.value());
    if (beta_opt.has_value()) {
      matcher.template match<XDType, IndexEnum::BroadcastBS>(beta_opt.value());
    }
  }

  const auto B = matcher.B_.unwrap();
  const auto S = matcher.S_.unwrap();
  const auto F = matcher.has_value_F ? matcher.F_.unwrap() : 0;
  const auto D = matcher.D_.unwrap();
  RuntimeCheck(D == kDim, "Tensor dimension D must match template kDim");

  // Compute thread configuration based on kDim
  constexpr uint32_t kThreads = host::norm::get_cta_threads<XDType, kDim>();

  dim3 grid(B * S);
  dim3 block(kThreads);

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;

  // Launch kernel
  LaunchKernel(grid, block, x.device())(
      norm_fused_scale_shift_kernel<
          XDType,
          ScaleDType,
          ShiftDType,
          kDim,
          norm_enum,
          scale_index_enum,
          shift_index_enum>,
      (XDType*)out.data_ptr(),
      (const XDType*)x.data_ptr(),
      (const XDType*)gamma_ptr,
      (const XDType*)beta_ptr,
      (const ScaleDType*)scale.data_ptr(),
      (const ShiftDType*)shift.data_ptr(),
      S,
      F,
      affine,
      static_cast<float>(eps));
}

template <
    typename XDType,
    typename GateDType,
    typename ScaleDType,
    typename ShiftDType,
    int64_t kDim,
    NormEnum norm_enum,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    IndexEnum gate_index_enum>
__global__ void norm_fused_res_gate_scale_shift_kernel(
    XDType* __restrict__ output,
    XDType* __restrict__ residual_out,
    const XDType* __restrict__ x,
    const XDType* __restrict__ residual,
    const XDType* __restrict__ gamma,
    const XDType* __restrict__ beta,
    const ScaleDType* __restrict__ scale,
    const ShiftDType* __restrict__ shift,
    const GateDType* __restrict__ gate,
    const int S,
    const int F,
    bool affine,
    float eps) {
  using namespace device;
  using namespace device::norm;
  using PackedT = packed_t<XDType>;
  using XStorage = norm::StorageType<XDType, kDim>;
  using GateTraits = ParamStorageTraits<GateDType>;
  using GateStorage = typename GateTraits::Storage;
  using ScaleTraits = ParamStorageTraits<ScaleDType>;
  using ShiftTraits = ParamStorageTraits<ShiftDType>;
  using ScaleStorage = typename ScaleTraits::Storage;
  using ShiftStorage = typename ShiftTraits::Storage;
  // ============ Setup ============
  __shared__ float smem_buffer[kSmemBufferSize];
  const int bidx = blockIdx.x;
  const int b_id = bidx / S, s_id = bidx % S;
  const auto gmem = tile::Memory<XStorage>::cta();
  const auto gate_mem = tile::Memory<GateStorage>::cta();
  const auto scale_mem = tile::Memory<ScaleStorage>::cta();
  const auto shift_mem = tile::Memory<ShiftStorage>::cta();
  constexpr int kStorageSize = 4;

  // Compute offsets
  const int scale_row = norm_fusion::get_offset<scale_index_enum>(S, F, b_id, s_id);
  const int shift_row = norm_fusion::get_offset<shift_index_enum>(S, F, b_id, s_id);
  const int gate_row = norm_fusion::get_offset<gate_index_enum>(S, F, b_id, s_id);

  // ============ Step 1: normed = norm(residual + x * gate) * gamma + beta ============
  const auto x_vec = gmem.load(x + bidx * kDim);
  const auto r_vec = gmem.load(residual + bidx * kDim);
  ParamLoader<GateDType> gate_loader;
  if constexpr (gate_index_enum == IndexEnum::NotATensor) {
    gate_loader.fill_scalar(1.0f);
  } else if constexpr (gate_index_enum == IndexEnum::Scalar) {
    float g = static_cast<float>(gate[0]);
    gate_loader.fill_scalar(g);
  } else {
    gate_loader.load(gate_mem, gate + gate_row * kDim, 0);
  }
  XStorage gated;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto x_fp32 = cast<fp32x2_t>(x_vec[i]);
    auto r_fp32 = cast<fp32x2_t>(r_vec[i]);
    auto g_fp32 = gate_loader.pair(i);
    float sum_x = r_fp32.x + x_fp32.x * g_fp32.x;
    float sum_y = r_fp32.y + x_fp32.y * g_fp32.y;
    gated[i] = cast<PackedT, fp32x2_t>({sum_x, sum_y});
  }
  if (residual_out != nullptr) {
    gmem.store(residual_out + bidx * kDim, gated);
  }
  XStorage beta_vec;
  const auto gamma_vec = affine ? gmem.load(gamma) : XStorage(cast<PackedT, fp32x2_t>({1.0f, 1.0f}));
  if constexpr (norm_enum == NormEnum::LayerNorm)
    beta_vec = affine ? gmem.load(beta) : XStorage(cast<PackedT, fp32x2_t>({0.0f, 0.0f}));
  else
    beta_vec = XStorage(cast<PackedT, fp32x2_t>({0.0f, 0.0f}));
  const auto normed = apply_norm_cta<norm_enum, kDim>(gated, gamma_vec, beta_vec, eps, smem_buffer);

  // ============ Step 2: output = normed * (1 + scale) + shift ============
  ParamLoader<ScaleDType> scale_loader;
  ParamLoader<ShiftDType> shift_loader;
  if constexpr (scale_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(scale[0]);
    scale_loader.fill_scalar(s);
  } else {
    scale_loader.load(scale_mem, scale + scale_row * kDim, 0);
  }
  if constexpr (shift_index_enum == IndexEnum::Scalar) {
    float s = static_cast<float>(shift[0]);
    shift_loader.fill_scalar(s);
  } else {
    shift_loader.load(shift_mem, shift + shift_row * kDim, 0);
  }
  XStorage output_vec;
#pragma unroll
  for (int i = 0; i < kStorageSize; ++i) {
    auto norm_fp32 = cast<fp32x2_t>(normed[i]);
    auto scale_fp32 = scale_loader.pair(i);
    auto shift_fp32 = shift_loader.pair(i);
    float out_x = norm_fp32.x * (1.0f + scale_fp32.x) + shift_fp32.x;
    float out_y = norm_fp32.y * (1.0f + scale_fp32.y) + shift_fp32.y;
    output_vec[i] = cast<PackedT, fp32x2_t>({out_x, out_y});
  }
  gmem.store(output + bidx * kDim, output_vec);
}

template <
    NormEnum norm_enum,
    typename XDType,
    typename GateDType,
    typename ScaleDType,
    typename ShiftDType,
    IndexEnum scale_index_enum,
    IndexEnum shift_index_enum,
    IndexEnum gate_index_enum,
    int64_t kDim>
void fused_scale_residual_norm_scale_shift(
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView residual_out,
    const tvm::ffi::TensorView residual,
    const tvm::ffi::TensorView x,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gate_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> gamma_opt,
    const tvm::ffi::Optional<tvm::ffi::TensorView> beta_opt,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView shift,
    double eps) {
  using namespace host;

  static_assert(
      std::is_same_v<XDType, half> || std::is_same_v<XDType, nv_bfloat16>,
      "Only support fp16, bf16 for norm template version");
  static_assert(
      norm_enum == NormEnum::LayerNorm || norm_enum == NormEnum::RMSNorm, "norm_enum must be layernorm or rmsnorm.");
  static_assert(host::norm::is_config_supported<XDType, kDim>(), "Unsupported norm configuration for kDim");

  norm_fusion::Matcher matcher;
  matcher.template match<XDType, IndexEnum::NoBroadcast>(y);
  matcher.template match<XDType, IndexEnum::NoBroadcast>(residual_out);
  matcher.template match<XDType, IndexEnum::NoBroadcast>(x);
  matcher.template match<XDType, IndexEnum::NoBroadcast>(residual);
  matcher.template match<ScaleDType, scale_index_enum>(scale);
  matcher.template match<ShiftDType, shift_index_enum>(shift);
  if (gate_opt.has_value()) {
    matcher.template match<GateDType, gate_index_enum>(gate_opt.value());
  }
  bool affine = gamma_opt.has_value();
  if (affine) {
    matcher.template match<XDType, IndexEnum::BroadcastBS>(gamma_opt.value());
    if (beta_opt.has_value()) {
      matcher.template match<XDType, IndexEnum::BroadcastBS>(beta_opt.value());
    }
  }

  const auto B = matcher.B_.unwrap();
  const auto S = matcher.S_.unwrap();
  const auto F = matcher.has_value_F ? matcher.F_.unwrap() : 0;
  const auto D = matcher.D_.unwrap();
  RuntimeCheck(D == kDim, "Tensor dimension D must match template kDim");

  // Compute thread configuration based on kDim
  constexpr uint32_t kThreads = host::norm::get_cta_threads<XDType, kDim>();

  dim3 grid(B * S);
  dim3 block(kThreads);

  auto gamma_ptr = gamma_opt.has_value() ? gamma_opt.value().data_ptr() : nullptr;
  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;
  auto gate_ptr = gate_opt.has_value() ? gate_opt.value().data_ptr() : nullptr;

  // Launch kernel
  LaunchKernel(grid, block, x.device())(
      norm_fused_res_gate_scale_shift_kernel<
          XDType,
          GateDType,
          ScaleDType,
          ShiftDType,
          kDim,
          norm_enum,
          scale_index_enum,
          shift_index_enum,
          gate_index_enum>,
      (XDType*)y.data_ptr(),
      (XDType*)residual_out.data_ptr(),
      (const XDType*)x.data_ptr(),
      (const XDType*)residual.data_ptr(),
      (const XDType*)gamma_ptr,
      (const XDType*)beta_ptr,
      (const ScaleDType*)scale.data_ptr(),
      (const ShiftDType*)shift.data_ptr(),
      (const GateDType*)gate_ptr,
      S,
      F,
      affine,
      static_cast<float>(eps));
}

}  // namespace
