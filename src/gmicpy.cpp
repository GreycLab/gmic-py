#include "gmicpy.hpp"

#include <nanobind/intrusive/counter.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
// ReSharper disable CppUnusedIncludeDirective
#include <nanobind/intrusive/ref.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <nanobind/intrusive/counter.inl>
// ReSharper restore CppUnusedIncludeDirective

#include <CImg.h>

#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#define IS_DEFINED(macro)                                                   \
    ", " << #macro "="                                                      \
         << (strcmp(#macro, Py_STRINGIFY(macro)) != 0 ? Py_STRINGIFY(macro) \
                                                      : "0")

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;

void inspect(nb::ndarray<gmic_pixel_type, nb::device::cpu> a)
{
    printf("Array data pointer : %p\n", a.data());
    printf("Array dimension : %zu\n", a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) {
        printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
        printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
    }
    printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
           int(a.device_type() == nb::device::cpu::value),
           int(a.device_type() == nb::device::cuda::value));
    printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
           a.dtype() == nb::dtype<int16_t>(),
           a.dtype() == nb::dtype<uint32_t>(),
           a.dtype() == nb::dtype<float>());
}

template <typename T = gmic_pixel_type>
class gmic_image_py final : public nb::intrusive_base {
   public:
    gmic_image_py() : intrusive_base() {}
    gmic_image_py(gmic_image_py &) = delete;
    gmic_image_py(gmic_image_py &&other) noexcept
        : intrusive_base(other),
          image(other.image),
          name(std::move(other.name))
    {
        other.image._is_shared = true;
    };

    explicit gmic_image_py(gmic_image<T> &img, optional<string> name = {})
        : image(img), name(std::move(name))
    {
        image._is_shared = true;
    }

    explicit gmic_image_py(gmic_image<T> &&img, optional<string> name = {})
        : image(img), name(std::move(name))
    {
        img._is_shared = true;
    }

    ~gmic_image_py() = default;

    gmic_image_py &operator=(const gmic_image_py &other)
    {
        if (this == &other)
            return *this;
        intrusive_base::operator=(other);
        image = other.image;
        image._is_shared = true;
        name = other.name;
        return *this;
    }
    gmic_image_py &operator=(gmic_image_py &&other) noexcept
    {
        if (this == &other)
            return *this;
        intrusive_base::operator=(std::move(other));
        image = other.image;
        other.image._is_shared = true;
        name = std::move(other.name);
        return *this;
    }
    nb::ndarray<T, nb::device::cpu, nb::ndim<4>> to_ndarray()
    {
        size_t shape[] = {image._spectrum, image._depth, image._height,
                          image._width};
        return nb::ndarray<T, nb::device::cpu, nb::ndim<4>>(
            image._data, 4, shape, nb::handle());
    }

    string str() const
    {
        stringstream out;
        out << "<gmic._gmic.GmicImage object at "
            << static_cast<const void *>(this) << ">";
        return out.str();
    }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_image_py>(
            m, "GmicImage",
            nb::intrusive_ptr<gmic_image_py>(
                [](gmic_image_py *o, PyObject *po) noexcept {
                    o->set_self_py(po);
                }))
            .def("to_ndarray", &gmic_image_py::to_ndarray)
            .def("__str__", &gmic_image_py::str)
            .def("__repr__", &gmic_image_py::str);
    }

   private:
    gmic_image<T> image;
    optional<string> name;
};

template <typename T = gmic_pixel_type>
class gmic_list_py final : public nb::intrusive_base {
   public:
    gmic_list_py() = default;
    // gmic_list_py(nb::iterable &iter)
    // {
    //     for (auto img : iter) {
    //     }
    // }

    [[nodiscard]] gmic_list<T> &get_list() { return list; }
    [[nodiscard]] gmic_list<char> &get_name_list() { return name_list; }

    auto begin() { return list._data; }
    auto end() { return list._data + list._width; }

    auto iter()
    {
        return nb::make_iterator(nb::type<gmic_list_py>(), "iterator", begin(),
                                 end());
    }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_list_py>(
            m, "GmicImage",
            nb::intrusive_ptr<gmic_list_py>(
                [](gmic_list_py *o, PyObject *po) noexcept {
                    o->set_self_py(po);
                }))
            .def("__iter__", &gmic_list_py::iter);
    }

    class iterator
        : std::iterator<std::forward_iterator_tag, gmic_image_py<T>> {
        gmic_list_py<T> list_py;
        gmic_image<T> *iter;

       public:
        explicit iterator(gmic_list_py<T> &list_py) : list_py(list_py) {}
    };

   private:
    gmic_list<T> list;
    gmic_list<char> name_list;
};

class gmic_py final : public nb::intrusive_base {
   public:
    gmic_py() = default;

    explicit gmic_py(gmic &inter) : interpreter(inter) {}

    gmic_py(gmic_py &other) = default;

    template <typename T = gmic_pixel_type>
    shared_ptr<gmic_list_py<T>> run(const char *cmd,
                                    shared_ptr<gmic_list_py<T>> img_list)
    {
        if (!img_list)
            img_list = make_shared<gmic_list_py<T>>();
        try {
            this->interpreter.run(cmd, img_list->get_list(),
                                  img_list->get_name_list());
        }
        catch (gmic_exception &ex) {
            cerr << ex.what();
            if (errno)
                cerr << ": " << strerror(errno);
            cerr << endl;
            throw;
        }

        return img_list;
    }

    static void bind(const nb::module_ &m)
    {
        nb::class_<gmic_py>(
            m, "Gmic",
            nb::intrusive_ptr<gmic_py>(
                [](gmic_py *o, PyObject *po) noexcept { o->set_self_py(po); }))
            .def("run", &gmic_py::run<>, "cmd"_a,
                 "img_list"_a = shared_ptr<gmic_list_py<>>())
            .def(nb::init());
    }

   private:
    gmic interpreter;
};

NB_MODULE(_gmic, m)
{
    {
        static char version[16];
        constexpr auto patch = gmic_version % 10,
                       minor = (gmic_version / 10) % 10,
                       major = gmic_version / 100;
        snprintf(version, std::size(version), "%d.%d.%d", major, minor, patch);
        m.attr("__version__") = version;
    }
    {
        static char build[256];
        stringstream build_str;
        build_str << "Built on " __DATE__ << " at " << __TIME__
                  << IS_DEFINED(gmic_py_numpy) << IS_DEFINED(__cplusplus);
        strncpy(build, build_str.str().c_str(), size(build));
        m.attr("__build__") = build;
    };

    m.def("inspect", &inspect, "array"_a, "Inspects a N-dimensional array");
    gmic_image_py<>::bind(m);
    gmic_list_py<>::bind(m);
    gmic_py::bind(m);

    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
    const auto cimg_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        cimg_library::CImgException>(m, "CImgException");

    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });
}
}  // namespace gmicpy
