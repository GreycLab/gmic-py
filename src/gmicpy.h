#pragma once
#define PY_SSIZE_T_CLEAN
#include <CImg.h>
#include <Python.h>

#include "gmic.h"

//------- G'MIC MAIN TYPES ----------//

static PyObject *GmicException;

static PyTypeObject PyGmicImageType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "gmic.GmicImage" /* tp_name */
};

static PyTypeObject PyGmicType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "gmic.Gmic" /* tp_name */
};

struct PyGmicImage {
    PyObject_HEAD;
    gmic_image<gmic_pixel_type> *_gmic_image;  // G'MIC library's Gmic Image
};

struct PyGmic {
    PyObject_HEAD;
    // Using a pointer here and PyGmic_new()-time instantiation fixes a
    // crash with empty G'MIC command-set.
    gmic *_gmic;  // G'MIC library's interpreter instance
};