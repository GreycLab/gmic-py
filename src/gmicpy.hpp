#ifndef GMICPY_H
#define GMICPY_H
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

// Include gmic et CImg after nanobind
#include <CImg.h>
#include <gmic.h>

#include <bit>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <source_location>
#include <sstream>
#include <type_traits>

#include "logging.hpp"

#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace gmicpy {
void bind_gmic_image(nanobind::module_ &m);
void bind_gmic_list(nanobind::module_ &m);

std::ostream &operator<<(std::ostream &out, const cimg_library::CImg<> &img);
}  // namespace gmicpy

#endif  // GMICPY_H
