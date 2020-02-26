// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ROIAlign.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
}
