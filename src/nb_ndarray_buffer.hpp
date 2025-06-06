#ifndef NB_NDARRAY_BUFFER_HPP
#define NB_NDARRAY_BUFFER_HPP
#include <nanobind/ndarray.h>

#include <complex>

namespace gmicpy {

int ndarray_tpbuffer(const nanobind::ndarray<nanobind::ro> &array, bool ro,
                     const nanobind::handle &exporter, Py_buffer *view,
                     int flags) noexcept;

void nb_ndarray_releasebuffer(PyObject *, Py_buffer *view);

constexpr const char *ndarray_get_pybuf_format(nanobind::dlpack::dtype type)
{
    return nullptr;
}

constexpr bool ndarray_fill_pybuf_view(const int device_type,
                                       const nanobind::dlpack::dtype type,
                                       Py_buffer *view) noexcept(false)
{
    using namespace nanobind;
    if (device_type != device::cpu::value) {
        throw buffer_error(
            "Only CPU-allocated ndarrays can be accessed via the buffer "
            "protocol!");
    }

    const char *format = nullptr;
    switch (type.code) {
        // Workaround for a conflict with Complex and Bool being macros in the
        // X11 lib
        case dtype<int>().code:
            switch (type.bits) {
                case 8:
                    format = "b";
                    break;
                case 16:
                    format = "h";
                    break;
                case 32:
                    format = "i";
                    break;
                case 64:
                    format = "q";
                    break;
                default:
                    break;
            }
            break;

        case nanobind::dtype<unsigned int>().code:
            switch (type.bits) {
                case 8:
                    format = "B";
                    break;
                case 16:
                    format = "H";
                    break;
                case 32:
                    format = "I";
                    break;
                case 64:
                    format = "Q";
                    break;
                default:
                    break;
            }
            break;

        case nanobind::dtype<float>().code:
            switch (type.bits) {
                case 16:
                    format = "e";
                    break;
                case 32:
                    format = "f";
                    break;
                case 64:
                    format = "d";
                    break;
                default:
                    break;
            }
            break;

        case nanobind::dtype<std::complex<int>>().code:
            switch (type.bits) {
                case 64:
                    format = "Zf";
                    break;
                case 128:
                    format = "Zd";
                    break;
                default:
                    break;
            }
            break;

        case nanobind::dtype<bool>().code:
            format = "?";
            break;

        default:
            break;
    }

    // Allow void (bits = 0) if we're only static checking (view is null)
    if (((type.bits || view) && !format) ||
        (type.lanes != 1 && (view || type.bits != 0))) {
        throw buffer_error(
            "Don't know how to convert DLPack dtype into buffer "
            "protocol format!");
    }
    if (view) {
        view->format = const_cast<char *>(format);
        view->itemsize = (type.bits + 7) / 8;
    }
    return true;
}

template <int device_type, nanobind::dlpack::dtype type>
constexpr bool ndarray_fill_pybuf_view() noexcept(false)
{
    static_assert(ndarray_fill_pybuf_view(device_type, type, nullptr));
    return true;
}

template <class... Args>
int ndarray_tpbuffer(const nanobind::ndarray<Args...> &array,
                     const nanobind::handle &exporter, Py_buffer *view,
                     const int flags) noexcept
{
    using namespace nanobind;
    ndarray_fill_pybuf_view<
        ndarray<Args...>::DeviceType,
        nanobind::dtype<typename ndarray<Args...>::Scalar>()>();
    return ndarray_tpbuffer(ndarray<ro>(array), ndarray<Args...>::ReadOnly,
                            exporter, view, flags);
}

}  // namespace gmicpy

#endif  // NB_NDARRAY_BUFFER_HPP
