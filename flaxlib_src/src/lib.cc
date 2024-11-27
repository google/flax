#include <string>

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"

namespace flaxlib {
std::string sum_as_string(int a, int b) {
  return std::to_string(a + b);
}

NB_MODULE(flaxlib, m) {
  m.def("sum_as_string", &sum_as_string);
}
}  // namespace flaxlib