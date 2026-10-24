#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
spherical_voronoi_forward(
    torch::Tensor dirs,
    torch::Tensor sites,
    torch::Tensor tau,
    torch::Tensor colors
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spherical_voronoi_backward(
    torch::Tensor dirs,
    torch::Tensor sites,
    torch::Tensor tau,
    torch::Tensor colors,
    torch::Tensor w,
    torch::Tensor dist,
    torch::Tensor feat,
    torch::Tensor v_rgb
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward",  &spherical_voronoi_forward,
          "Spherical Voronoi forward  (CUDA): returns (rgb, w, dist, feat)");
    m.def("backward", &spherical_voronoi_backward,
          "Spherical Voronoi backward (CUDA): returns (v_sites, v_tau, v_colors)");
}
