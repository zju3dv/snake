#include "utils.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collect_extreme_point", &collect_extreme_point, "collect_extreme_point");
    m.def("calculate_edge_num", &calculate_edge_num, "calculate_edge_num");
    m.def("calculate_wnp", &calculate_wnp, "calculate_wnp");
    m.def("roll_array", &roll_array, "roll_array");
    m.def("nms", &nms, "non-maximum suppression");
}
