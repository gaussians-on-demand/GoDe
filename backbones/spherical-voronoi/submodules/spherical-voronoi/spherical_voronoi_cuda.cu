#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static constexpr int   MAX_LOBES = 12;
static constexpr float EPS       = 1e-12f;

// ─── Forward ─────────────────────────────────────────────────────────────────
// 1 thread per Gaussian n.
// dirs[N,3], sites[N,S,3], tau[N,S], colors[N,S,3]
// → rgb[N,3]  = relu(feat+0.5)
// → w[N,S], dist[N,S], feat[N,3]  (saved for backward)

__global__ void sv_fwd_kernel(
    const float* __restrict__ dirs,    // [N, 3]
    const float* __restrict__ sites,   // [N, S, 3]
    const float* __restrict__ tau,     // [N, S]
    const float* __restrict__ colors,  // [N, S, 3]
    float* __restrict__ out_rgb,       // [N, 3]
    float* __restrict__ out_w,         // [N, S]
    float* __restrict__ out_dist,      // [N, S]
    float* __restrict__ out_feat,      // [N, 3]
    int N, int S
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float d0 = dirs[n*3+0], d1 = dirs[n*3+1], d2 = dirs[n*3+2];

    float lw[MAX_LOBES], ldist[MAX_LOBES];
    float max_logit = -1e30f;

    for (int s = 0; s < S; s++) {
        int sb  = (n*S + s) * 3;
        float dx = sites[sb+0] - d0;
        float dy = sites[sb+1] - d1;
        float dz = sites[sb+2] - d2;
        float dist = sqrtf(fmaxf(dx*dx + dy*dy + dz*dz, EPS));
        ldist[s] = dist;
        float logit = -tau[n*S + s] * dist;
        lw[s] = logit;
        if (logit > max_logit) max_logit = logit;
    }

    float denom = 0.f;
    for (int s = 0; s < S; s++) {
        float e = expf(lw[s] - max_logit);
        lw[s] = e;
        denom += e;
    }
    float inv_denom = 1.f / denom;
    for (int s = 0; s < S; s++) lw[s] *= inv_denom;

    float f0=0.f, f1=0.f, f2=0.f;
    for (int s = 0; s < S; s++) {
        int cb = (n*S + s) * 3;
        float ws = lw[s];
        f0 += ws * colors[cb+0];
        f1 += ws * colors[cb+1];
        f2 += ws * colors[cb+2];
    }

    out_feat[n*3+0] = f0; out_feat[n*3+1] = f1; out_feat[n*3+2] = f2;
    out_rgb[n*3+0] = fmaxf(f0 + 0.5f, 0.f);
    out_rgb[n*3+1] = fmaxf(f1 + 0.5f, 0.f);
    out_rgb[n*3+2] = fmaxf(f2 + 0.5f, 0.f);
    for (int s = 0; s < S; s++) {
        out_w[n*S + s]    = lw[s];
        out_dist[n*S + s] = ldist[s];
    }
}

// ─── Backward ────────────────────────────────────────────────────────────────
// 1 thread per Gaussian n.
// No v_dirs: view directions are not optimized.

__global__ void sv_bwd_kernel(
    const float* __restrict__ dirs,    // [N, 3]
    const float* __restrict__ sites,   // [N, S, 3]
    const float* __restrict__ tau,     // [N, S]
    const float* __restrict__ colors,  // [N, S, 3]
    const float* __restrict__ w,       // [N, S]
    const float* __restrict__ dist,    // [N, S]
    const float* __restrict__ feat,    // [N, 3]
    const float* __restrict__ v_rgb,   // [N, 3]
    float* __restrict__ v_sites,       // [N, S, 3]
    float* __restrict__ v_tau,         // [N, S]
    float* __restrict__ v_colors,      // [N, S, 3]
    int N, int S
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float d0 = dirs[n*3+0], d1 = dirs[n*3+1], d2 = dirs[n*3+2];

    // relu backward: v_feat[c] = v_rgb[c] * (feat[c] + 0.5 > 0)
    float vf0 = v_rgb[n*3+0] * (feat[n*3+0] + 0.5f > 0.f ? 1.f : 0.f);
    float vf1 = v_rgb[n*3+1] * (feat[n*3+1] + 0.5f > 0.f ? 1.f : 0.f);
    float vf2 = v_rgb[n*3+2] * (feat[n*3+2] + 0.5f > 0.f ? 1.f : 0.f);

    // v_colors[n,s,c] = w[s] * vf[c]
    // v_w[s]          = dot(colors[n,s,:], vf)
    float vw[MAX_LOBES];
    for (int s = 0; s < S; s++) {
        float ws = w[n*S + s];
        int cb = (n*S + s) * 3;
        v_colors[cb+0] = ws * vf0;
        v_colors[cb+1] = ws * vf1;
        v_colors[cb+2] = ws * vf2;
        vw[s] = colors[cb+0]*vf0 + colors[cb+1]*vf1 + colors[cb+2]*vf2;
    }

    // softmax backward: dot_wvw = sum_k w[k]*vw[k]
    float dot_wvw = 0.f;
    for (int s = 0; s < S; s++) dot_wvw += w[n*S + s] * vw[s];

    for (int s = 0; s < S; s++) {
        float ws     = w[n*S + s];
        float ds     = dist[n*S + s];
        float ts     = tau[n*S + s];
        float vsc    = ws * (vw[s] - dot_wvw);  // v_score[s]

        v_tau[n*S + s] = -ds * vsc;

        float vd = -ts * vsc;  // v_dist[s]

        int sb = (n*S + s) * 3;
        if (ds > 1e-9f) {
            float inv_d = vd / ds;
            v_sites[sb+0] = inv_d * (sites[sb+0] - d0);
            v_sites[sb+1] = inv_d * (sites[sb+1] - d1);
            v_sites[sb+2] = inv_d * (sites[sb+2] - d2);
        } else {
            v_sites[sb+0] = v_sites[sb+1] = v_sites[sb+2] = 0.f;
        }
    }
}

// ─── C++ wrappers ────────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
spherical_voronoi_forward(
    torch::Tensor dirs,    // [N, 3]  unit directions
    torch::Tensor sites,   // [N, S, 3]  unit sites
    torch::Tensor tau,     // [N, S]  temperatures (already exp'd, > 0)
    torch::Tensor colors   // [N, S, 3]
) {
    CHECK_INPUT(dirs); CHECK_INPUT(sites);
    CHECK_INPUT(tau);  CHECK_INPUT(colors);
    TORCH_CHECK(dirs.scalar_type()   == at::kFloat, "dirs must be float32");
    TORCH_CHECK(sites.scalar_type()  == at::kFloat, "sites must be float32");
    TORCH_CHECK(tau.scalar_type()    == at::kFloat, "tau must be float32");
    TORCH_CHECK(colors.scalar_type() == at::kFloat, "colors must be float32");
    TORCH_CHECK(dirs.dim() == 2 && dirs.size(1) == 3,   "dirs must be [N,3]");
    TORCH_CHECK(sites.dim() == 3 && sites.size(2) == 3, "sites must be [N,S,3]");
    TORCH_CHECK(colors.dim() == 3 && colors.size(2) == 3, "colors must be [N,S,3]");
    const int N = dirs.size(0), S = sites.size(1);
    TORCH_CHECK(S <= MAX_LOBES, "num_lobes exceeds MAX_LOBES=", MAX_LOBES);
    TORCH_CHECK(sites.size(0)  == N && tau.size(0) == N && colors.size(0) == N, "N mismatch");
    TORCH_CHECK((int)tau.size(1) == S && (int)colors.size(1) == S, "S mismatch");

    auto opts  = torch::TensorOptions().device(dirs.device()).dtype(at::kFloat);
    auto rgb   = torch::empty({N, 3}, opts);
    auto out_w = torch::empty({N, S}, opts);
    auto out_d = torch::empty({N, S}, opts);
    auto feat  = torch::empty({N, 3}, opts);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    sv_fwd_kernel<<<blocks, threads, 0, stream>>>(
        dirs.data_ptr<float>(), sites.data_ptr<float>(),
        tau.data_ptr<float>(), colors.data_ptr<float>(),
        rgb.data_ptr<float>(), out_w.data_ptr<float>(),
        out_d.data_ptr<float>(), feat.data_ptr<float>(),
        N, S);

    return {rgb, out_w, out_d, feat};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spherical_voronoi_backward(
    torch::Tensor dirs,    // [N, 3]
    torch::Tensor sites,   // [N, S, 3]
    torch::Tensor tau,     // [N, S]
    torch::Tensor colors,  // [N, S, 3]
    torch::Tensor w,       // [N, S]
    torch::Tensor dist,    // [N, S]
    torch::Tensor feat,    // [N, 3]
    torch::Tensor v_rgb    // [N, 3]
) {
    CHECK_INPUT(dirs); CHECK_INPUT(sites); CHECK_INPUT(tau); CHECK_INPUT(colors);
    CHECK_INPUT(w);    CHECK_INPUT(dist);  CHECK_INPUT(feat); CHECK_INPUT(v_rgb);

    const int N = dirs.size(0), S = sites.size(1);

    auto opts    = torch::TensorOptions().device(dirs.device()).dtype(at::kFloat);
    auto v_sites  = torch::zeros({N, S, 3}, opts);
    auto v_tau    = torch::zeros({N, S},    opts);
    auto v_colors = torch::zeros({N, S, 3}, opts);

    if (N == 0) return {v_sites, v_tau, v_colors};

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    sv_bwd_kernel<<<blocks, threads, 0, stream>>>(
        dirs.data_ptr<float>(), sites.data_ptr<float>(),
        tau.data_ptr<float>(), colors.data_ptr<float>(),
        w.data_ptr<float>(), dist.data_ptr<float>(),
        feat.data_ptr<float>(), v_rgb.data_ptr<float>(),
        v_sites.data_ptr<float>(), v_tau.data_ptr<float>(),
        v_colors.data_ptr<float>(),
        N, S);

    return {v_sites, v_tau, v_colors};
}
