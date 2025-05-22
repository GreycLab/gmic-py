#include "nb_ndarray_buffer.hpp"

#include "logging.hpp"

namespace gmicpy {
using namespace nanobind;
using namespace std;
// This code is an adaptated version of some internal code of nanobind and
// is thus licensed under the BSD 3-clause license
// see: https://github.com/wjakob/nanobind

int nd_ndarray_tpbuffer(ndarray<ro> array, const bool ro,
                        const handle &exporter, Py_buffer *view,
                        const int flags) noexcept
{
    LOG_TRACE();
    if (ro && flags & PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError,
                        "Writable view requested of read-only buffer");
        return -1;
    }

    try {
        fill_pybuf_view(array.device_type(), array.dtype(), view);
    }
    catch (std::exception &ex) {
        PyErr_SetString(PyExc_BufferError, ex.what());
        return -1;
    }

    view->buf = const_cast<void *>(array.data());
    view->obj = exporter.inc_ref().ptr();

    Py_ssize_t len = view->itemsize;
    const auto strides = static_cast<Py_ssize_t *>(
                   PyMem_Malloc(array.ndim() * sizeof(Py_ssize_t))),
               shape = static_cast<Py_ssize_t *>(
                   PyMem_Malloc(array.ndim() * sizeof(Py_ssize_t)));

    for (size_t i = 0; i < array.ndim(); ++i) {
        len *= (Py_ssize_t)array.shape(i);
        strides[i] = (Py_ssize_t)array.stride(i) * view->itemsize;
        shape[i] = (Py_ssize_t)array.shape(i);
    }

    view->ndim = static_cast<int>(array.ndim());
    view->len = len;
    view->readonly = ro;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    view->strides = strides;
    view->shape = shape;

    return 0;
}

void nb_ndarray_releasebuffer(PyObject *, Py_buffer *view)
{
    LOG_TRACE();
    PyMem_Free(view->shape);
    PyMem_Free(view->strides);
}

}  // namespace gmicpy