#pragma once
#include <torch/extension.h>
#include "src/nms.h"


at::Tensor collect_extreme_point(
    const at::Tensor& ext_hm,
    const at::Tensor& bbox,
    const at::Tensor& radius,
    const at::Tensor& vote,
    const at::Tensor& ct
);


void calculate_edge_num(
    at::Tensor& edge_num,
    const at::Tensor& edge_num_sum,
    const at::Tensor& edge_idx_sort,
    const int p_num
);


std::tuple<at::Tensor, at::Tensor> calculate_wnp(
    const at::Tensor& edge_num,
    const at::Tensor& edge_start_idx,
    const int p_num
);


at::Tensor roll_array(
    const at::Tensor& array,
    const at::Tensor& step
);


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
    if (dets.numel() == 0)
        return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
}

