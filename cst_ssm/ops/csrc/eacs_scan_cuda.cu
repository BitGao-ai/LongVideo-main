// eacs_scan_cuda.cu —— 变步长对角复数选择性扫描 CUDA 内核（设计方案 §6 工程落地）
//
// 求解一阶线性递推： h_k = a_k ⊙ h_{k-1} + b_k   （逐元素、复数、对角）
// "变步长"体现在 a_k = exp(λ·Δt_k) 由上层按**真实物理 Δt_k** 逐步算好后传入——本内核对任意逐步
// 变化的 a_k 均适用，无需固定步长（对应把 Mamba-2 固定 Δ 扫描改造为可变 Δ 连续状态转移）。
//
// 并行策略：对 (B·H·N) 个独立通道并行（每线程一个），线程内沿 L 顺序扫描（前向）/逆序（反向）。
// 布局：a,b,h 均为 (B,L,H,N) 连续 complex64；offset = ((b*L+l)*H+h)*N+n。
// 反向：线性递推的伴随（reverse-mode），
//   s_l = grad_h_l + conj(a_{l+1})·s_{l+1};  grad_b_l = s_l;  grad_a_l = s_l·conj(h_{l-1}).
// 该 conj 约定与 PyTorch 复数 autograd 一致（已用纯 PyTorch 镜像在 CPU 对拍验证）。

#include <torch/extension.h>
#include <c10/util/complex.h>
#include <vector>

using cf = c10::complex<float>;

__device__ __forceinline__ cf dconj(cf z) { return cf(z.real(), -z.imag()); }

__global__ void scan_fwd_kernel(const cf* __restrict__ a, const cf* __restrict__ b,
                                cf* __restrict__ h, int B, int L, int H, int N) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long BHN = (long)B * H * N;
    if (idx >= BHN) return;
    int ni = idx % N;
    long tmp = idx / N; int hi = tmp % H; int bi = tmp / H;
    long stride_l = (long)H * N;
    long base = (long)bi * L * H * N + (long)hi * N + ni;
    cf state(0.f, 0.f);
    for (int l = 0; l < L; ++l) {
        long off = base + (long)l * stride_l;
        state = a[off] * state + b[off];
        h[off] = state;
    }
}

__global__ void scan_bwd_kernel(const cf* __restrict__ a, const cf* __restrict__ h,
                                const cf* __restrict__ grad_h,
                                cf* __restrict__ grad_a, cf* __restrict__ grad_b,
                                int B, int L, int H, int N) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long BHN = (long)B * H * N;
    if (idx >= BHN) return;
    int ni = idx % N;
    long tmp = idx / N; int hi = tmp % H; int bi = tmp / H;
    long stride_l = (long)H * N;
    long base = (long)bi * L * H * N + (long)hi * N + ni;
    cf s(0.f, 0.f);
    for (int l = L - 1; l >= 0; --l) {
        long off = base + (long)l * stride_l;
        s = s + grad_h[off];
        grad_b[off] = s;
        cf h_prev = (l > 0) ? h[off - stride_l] : cf(0.f, 0.f);
        grad_a[off] = s * dconj(h_prev);
        s = dconj(a[off]) * s;
    }
}

std::vector<torch::Tensor> scan_fwd(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "expects CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kComplexFloat, "expects complex64");
    TORCH_CHECK(a.dim() == 4, "expects (B,L,H,N)");
    a = a.contiguous(); b = b.contiguous();
    int B = a.size(0), L = a.size(1), H = a.size(2), N = a.size(3);
    auto h = torch::empty_like(b);
    long BHN = (long)B * H * N; int threads = 256;
    long blocks = (BHN + threads - 1) / threads;
    scan_fwd_kernel<<<blocks, threads>>>(
        reinterpret_cast<cf*>(a.data_ptr()), reinterpret_cast<cf*>(b.data_ptr()),
        reinterpret_cast<cf*>(h.data_ptr()), B, L, H, N);
    return {h};
}

std::vector<torch::Tensor> scan_bwd(torch::Tensor a, torch::Tensor h, torch::Tensor grad_h) {
    a = a.contiguous(); h = h.contiguous(); grad_h = grad_h.contiguous();
    int B = a.size(0), L = a.size(1), H = a.size(2), N = a.size(3);
    auto grad_a = torch::empty_like(a), grad_b = torch::empty_like(a);
    long BHN = (long)B * H * N; int threads = 256;
    long blocks = (BHN + threads - 1) / threads;
    scan_bwd_kernel<<<blocks, threads>>>(
        reinterpret_cast<cf*>(a.data_ptr()), reinterpret_cast<cf*>(h.data_ptr()),
        reinterpret_cast<cf*>(grad_h.data_ptr()),
        reinterpret_cast<cf*>(grad_a.data_ptr()), reinterpret_cast<cf*>(grad_b.data_ptr()),
        B, L, H, N);
    return {grad_a, grad_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_fwd", &scan_fwd, "variable-step diagonal complex scan forward (CUDA)");
    m.def("scan_bwd", &scan_bwd, "variable-step diagonal complex scan backward (CUDA)");
}
