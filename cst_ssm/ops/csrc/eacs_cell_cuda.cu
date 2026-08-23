// eacs_cell_cuda.cu —— 门控 EACS cell 融合前向内核（推理，硬门控）。
//
// 复刻 modules/eacs.py::EACSLayer._step 的 segment 形式逐步扫描（保持输入 ZOH 预测 + 预测编码残差门控），
// 一次内核跑完整段 L：每个 block 处理一个 batch 元素，沿 L 顺序（门控依赖状态，无法跨步并行），
// 块内 256 线程按通道 H 网格跨步并行，残差范数按 H 跨线程规约。committed 状态 (h_π,t_π,u_π,B_π,C_π) 常驻
// 全局，显存 O(B·H·N)（与视频长度无关）。仅前向/推理；训练走 eacs.py 的 autograd 序列 cell。
//
// 残差里 obs=(u-μ)/σ, pred=(ŷ-μ)/σ ⇒ obs-pred=(u-ŷ)/σ（μ 抵消），与 PyTorch 路径一致（已 CPU 对拍）。

#include <torch/extension.h>
#include <c10/util/complex.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

using cf = c10::complex<float>;

__device__ __forceinline__ cf cscale(cf z, float s) { return cf(z.real() * s, z.imag() * s); }
__device__ __forceinline__ cf cexp(cf z) {
    float e = expf(z.real());
    return cf(e * cosf(z.imag()), e * sinf(z.imag()));
}
// 无分支稳定复数 expm1：与 Python 端 ops/discretization.complex_expm1 **同一个公式**。
//   Re = expm1(a)·cos b − 2·sin²(b/2)，  Im = (expm1(a)+1)·sin b
// 两条实部项各自都不抵消，z→0 时天然稳定，不需要 |z| 小走 Taylor 的分支。
// 两端必须保持同式：B10 修的"前向轨迹与反向重算轨迹分叉"依赖 CPU/CUDA 门控判据一致，
// 分支阈值处的 1e-7 级跳变足以让残差落在 ε 附近的帧翻转门控。
__device__ __forceinline__ cf cexpm1(cf z) {
    float a = z.real(), b = z.imag();
    float em1 = expm1f(a);
    float sh = sinf(0.5f * b);
    return cf(em1 * cosf(b) - 2.f * sh * sh, (em1 + 1.f) * sinf(b));
}

// 前向内核：一个 block 处理一个 batch b。
__global__ void eacs_cell_fwd_kernel(
        const cf* __restrict__ lam,          // (H,N)
        const float* __restrict__ log_dt,    // (H,)
        const float* __restrict__ u,         // (B,L,H)
        const cf* __restrict__ Bc,           // (B,L,N)
        const cf* __restrict__ Cc,           // (B,L,N)
        const float* __restrict__ t,         // (B,L)
        const float* __restrict__ Dp,        // (H,)
        const float* __restrict__ mean,      // (H,)
        const float* __restrict__ var,       // (H,)
        cf* __restrict__ h_pi,               // (B,H,N) scratch
        float* __restrict__ u_pi,            // (B,H)   scratch
        float* __restrict__ t_pi,            // (B,)    scratch
        cf* __restrict__ B_pi,               // (B,N)   scratch
        cf* __restrict__ C_pi,               // (B,N)   scratch
        float* __restrict__ y_out,           // (B,L,H)
        float* __restrict__ gate_out,        // (B,L)
        float* __restrict__ resid_out,       // (B,L)
        int B, int L, int H, int N,
        float eps, float eta, float var_eps, float dt_init,
        float dt_min, float dt_max) {

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int nth = blockDim.x;
    __shared__ float sdiff[256];
    __shared__ float sobs[256];
    __shared__ float sg;

    // ---- 初始化 committed 状态 ----
    for (int h = tid; h < H; h += nth) {
        for (int n = 0; n < N; ++n) h_pi[(b * H + h) * N + n] = cf(0.f, 0.f);
        u_pi[b * H + h] = u[(b * L + 0) * H + h];
    }
    for (int n = tid; n < N; n += nth) {
        B_pi[b * N + n] = Bc[(b * L + 0) * N + n];
        C_pi[b * N + n] = Cc[(b * L + 0) * N + n];
    }
    if (tid == 0) t_pi[b] = t[b * L + 0] - dt_init;
    __syncthreads();

    for (int l = 0; l < L; ++l) {
        float tl = t[b * L + l];
        float tpi = t_pi[b];
        float delta = tl - tpi; if (delta < 0.f) delta = 0.f;

        // ---- Phase A：预测读出 ŷ[h] + 累积残差/观测平方 ----
        float pdiff = 0.f, pobs = 0.f;
        for (int h = tid; h < H; h += nth) {
            float dte = delta * expf(log_dt[h]);
            dte = fminf(fmaxf(dte, dt_min), dt_max);
            float ul = u[(b * L + l) * H + h];
            float upi = u_pi[b * H + h];
            float yhat = 0.f;
            for (int n = 0; n < N; ++n) {
                cf lam_hn = lam[h * N + n];
                cf z = cscale(lam_hn, dte);
                cf dA = cexp(z);
                cf dBbar = cexpm1(z) / lam_hn;
                cf hpi = h_pi[(b * H + h) * N + n];
                cf xhat = dA * hpi + cscale(dBbar * B_pi[b * N + n], upi);
                yhat += (C_pi[b * N + n] * xhat).real();
            }
            float s = sqrtf(var[h] + var_eps);
            float diff = (ul - yhat) / s;      // (u-ŷ)/σ（μ 抵消）
            float obs = (ul - mean[h]) / s;
            pdiff += diff * diff; pobs += obs * obs;
        }
        sdiff[tid] = pdiff; sobs[tid] = pobs;
        __syncthreads();
        for (int stride = nth / 2; stride > 0; stride >>= 1) {   // 树规约
            if (tid < stride) { sdiff[tid] += sdiff[tid + stride]; sobs[tid] += sobs[tid + stride]; }
            __syncthreads();
        }
        if (tid == 0) {
            float r = sqrtf(sdiff[0]) / (sqrtf(sobs[0]) + eta);
            float g = (r > eps) ? 1.f : 0.f;
            sg = g; gate_out[b * L + l] = g; resid_out[b * L + l] = r;
        }
        __syncthreads();
        float g = sg;

        // ---- Phase C1：更新/选择/输出 + 提交 h_pi,u_pi（读 B_pi/C_pi 旧值）----
        for (int h = tid; h < H; h += nth) {
            float dte = delta * expf(log_dt[h]);
            dte = fminf(fmaxf(dte, dt_min), dt_max);
            float ul = u[(b * L + l) * H + h];
            float upi = u_pi[b * H + h];
            float yk = 0.f;
            for (int n = 0; n < N; ++n) {
                cf lam_hn = lam[h * N + n];
                cf z = cscale(lam_hn, dte);
                cf dA = cexp(z);
                cf dBbar = cexpm1(z) / lam_hn;
                cf hpi = h_pi[(b * H + h) * N + n];
                cf xhat = dA * hpi + cscale(dBbar * B_pi[b * N + n], upi);
                cf hupd = dA * hpi + cscale(dBbar * Bc[(b * L + l) * N + n], ul);
                cf hcur = (g > 0.5f) ? hupd : xhat;
                if (g > 0.5f) h_pi[(b * H + h) * N + n] = hupd;
                yk += (Cc[(b * L + l) * N + n] * hcur).real();
            }
            y_out[(b * L + l) * H + h] = yk + Dp[h] * ul;
            if (g > 0.5f) u_pi[b * H + h] = ul;
        }
        __syncthreads();   // 确保所有通道读完 B_pi/C_pi 再提交

        // ---- Phase C2：提交 B_pi,C_pi,t_pi ----
        if (g > 0.5f) {
            for (int n = tid; n < N; n += nth) {
                B_pi[b * N + n] = Bc[(b * L + l) * N + n];
                C_pi[b * N + n] = Cc[(b * L + l) * N + n];
            }
            if (tid == 0) t_pi[b] = tl;
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> eacs_cell_fwd(
        torch::Tensor lam, torch::Tensor log_dt, torch::Tensor u,
        torch::Tensor Bc, torch::Tensor Cc, torch::Tensor t,
        torch::Tensor Dp, torch::Tensor mean, torch::Tensor var,
        double eps, double eta, double var_eps, double dt_init,
        double dt_min, double dt_max) {
    TORCH_CHECK(u.is_cuda(), "expects CUDA tensors");
    TORCH_CHECK(u.dim() == 3 && lam.dim() == 2, "u:(B,L,H), lam:(H,N)");
    TORCH_CHECK(lam.size(0) == u.size(2), "lam 的 H 必须与 u 的 H 一致");
    // .contiguous() 的结果必须先绑到具名变量再取 data_ptr：写成
    // `lam.contiguous().data_ptr()` 时那个临时张量在语句结束即析构，而 kernel 是异步的
    // ——同流下 caching allocator 恰好能保证复用顺序，但这是不该依赖的行为。
    auto lam_c = lam.to(torch::kComplexFloat).contiguous();
    auto logdt_c = log_dt.to(torch::kFloat32).contiguous();
    auto u_c = u.to(torch::kFloat32).contiguous();
    auto Bc_c = Bc.to(torch::kComplexFloat).contiguous();
    auto Cc_c = Cc.to(torch::kComplexFloat).contiguous();
    auto t_c = t.to(torch::kFloat32).contiguous();
    auto Dp_c = Dp.to(torch::kFloat32).contiguous();
    auto mean_c = mean.to(torch::kFloat32).contiguous();
    auto var_c = var.to(torch::kFloat32).contiguous();

    int B = u_c.size(0), L = u_c.size(1), H = u_c.size(2), N = lam_c.size(1);
    auto copt = torch::TensorOptions().dtype(torch::kComplexFloat).device(u.device());
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(u.device());
    auto h_pi = torch::zeros({B, H, N}, copt);
    auto u_pi = torch::zeros({B, H}, fopt);
    auto t_pi = torch::zeros({B}, fopt);
    auto B_pi = torch::zeros({B, N}, copt);
    auto C_pi = torch::zeros({B, N}, copt);
    auto y = torch::zeros({B, L, H}, fopt);
    auto gate = torch::zeros({B, L}, fopt);
    auto resid = torch::zeros({B, L}, fopt);

    int threads = 256;
    // 必须在 PyTorch 的**当前流**上发射：默认流与 AMP / CUDA graph / 多流流水线下
    // PyTorch 实际使用的流不是同一个，会与前后算子产生 race。
    auto stream = c10::cuda::getCurrentCUDAStream();
    eacs_cell_fwd_kernel<<<B, threads, 0, stream>>>(
        reinterpret_cast<cf*>(lam_c.data_ptr()),
        logdt_c.data_ptr<float>(), u_c.data_ptr<float>(),
        reinterpret_cast<cf*>(Bc_c.data_ptr()),
        reinterpret_cast<cf*>(Cc_c.data_ptr()),
        t_c.data_ptr<float>(), Dp_c.data_ptr<float>(),
        mean_c.data_ptr<float>(), var_c.data_ptr<float>(),
        reinterpret_cast<cf*>(h_pi.data_ptr()), u_pi.data_ptr<float>(),
        t_pi.data_ptr<float>(), reinterpret_cast<cf*>(B_pi.data_ptr()),
        reinterpret_cast<cf*>(C_pi.data_ptr()),
        y.data_ptr<float>(), gate.data_ptr<float>(), resid.data_ptr<float>(),
        B, L, H, N, (float)eps, (float)eta, (float)var_eps, (float)dt_init,
        (float)dt_min, (float)dt_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();   // 否则启动失败会静默返回全零
    return {y, gate, resid};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eacs_cell_fwd", &eacs_cell_fwd, "fused gated EACS cell forward (CUDA, inference)");
}
