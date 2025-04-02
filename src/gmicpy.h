#ifndef GMICPY_H
#define GMICPY_H
// Common headers to include
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sstream>

// Include gmic et CImg after nanobind
#include <CImg.h>
#include <gmic.h>

#include "translate_args.h"
#endif  // GMICPY_H
