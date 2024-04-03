#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

// Include gmic et CImg after nanobind
#include <CImg.h>
#include <gmic.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#define IS_DEFINED(macro)                                                   \
    ", " << #macro "="                                                      \
         << (strcmp(#macro, Py_STRINGIFY(macro)) != 0 ? Py_STRINGIFY(macro) \
                                                      : "0")

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

static char ERRBUF[256];

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
class gmic_image_py {
    CImg<T> image;
    static constexpr auto DIMS = "SDHW";

    /**
     * Parses a dimension order specification. e.g.:
     *  "SDHW" (native gmic order) => {0, 1, 2, 3}
     *  "HWS" (Pillow order) => {2, -1, 0, 1}
     * @param spec String representation of a dimension order specification
     * @param allow_unset Whether or not to allow unspecified dimensions
     * @return fixed size array of the index of each dimension specifier in the
     * input array, or -1 unspecified and allow_unset us true
     * @throws nb::value_error if spec is too long (&gt; 4), too short (&lt, 4)
     * and allow_unset is false, or if any character in spec is unrecognized
     */
    vector<short> parse_dims(const string &spec, bool allow_unset = true)
    {
        vector<short> arr{-1, -1, -1, -1};
        if (spec.length() > strlen(DIMS))
            throw nb::value_error("Invalid dims argument size (> 4)");
        else if (!allow_unset && spec.length() < strlen(DIMS))
            throw nb::value_error("Missing dimensions in dims argument (< 4)");

        auto end = DIMS + strlen(DIMS);
        for (short i = 0; i < (short)spec.length(); i++) {
            char chr = (char)toupper(spec[i]);
            auto index = find(DIMS, end, chr) - DIMS;
            if (index == strlen(DIMS)) {
                snprintf(ERRBUF, size(ERRBUF),
                         "Unknown dimension specifier: '%c' (not in \"%s\")",
                         chr, DIMS);
                throw nb::value_error(ERRBUF);
            }
            if (arr[index] >= 0) {
                snprintf(ERRBUF, size(ERRBUF),
                         "Duplicate dimension specifier: '%c'", chr);
                throw nb::value_error(ERRBUF);
            }
            arr[index] = i;
        }

        return arr;
    }

    /**
     * Reorders a strides or shape vector according to the given dimensions
     * spec
     * @param spec dimension specification
     * @param native source array to reorder (strides() or shape())
     * @return an array of the input data reordered according to spec
     */
    template <typename V>
    auto reorder_to_spec(const vector<short> &spec, const vector<V> native)
    {
        vector<V> arr(native.size());
        for (short i : spec) {
            if (i >= 0)
                arr.push_back(native[i]);
        }
        return arr;
    }

    /**
     * Copies a ndarray. Will reorder the data so that the data is contiguous
     * in C-style order.
     * @tparam P Optional ndarray specifiers
     * @param array Input array whose data is <strong>in SDHW (gmic)
     * order</strong>
     * @return A copy of the ndarray with the same data for a given set of
     * coordinates, but reordered C-style
     */
    template <typename... P>
    nb::ndarray<T, P...> copy_ndarray(nb::ndarray<T, P...> array)
    {
        T *src = array.data(), *dest = new T[array.size()];
        nb::capsule owner(dest, [](void *p) noexcept { delete[] (float *)p; });
        int64_t strides_src[4]{0, 0, 0, 0}, strides_dst[4]{0, 0, 0, 0};
        size_t shape[4]{1, 1, 1, 1};

        for (int64_t s = 1, i = array.ndim() - 1; i >= 0; i--) {
            strides_src[i] = array.stride(i);
            strides_dst[i] = s;
            s *= shape[i] = array.shape(i);
        }

        for (size_t a = 0; a < shape[0]; a++) {
            const size_t ass = a * strides_src[0], ads = a * strides_dst[0];
            for (size_t b = 0; b < shape[1]; b++) {
                const size_t bss = b * strides_src[1] + ass,
                             bds = b * strides_dst[1] + ads;
                for (size_t c = 0; c < shape[2]; c++) {
                    const size_t css = c * strides_src[2] + bss,
                                 cds = c * strides_dst[2] + bds;
                    for (size_t d = 0; d < shape[3]; d++) {
                        size_t si = d * strides_src[3] + css,
                               di = d * strides_dst[3] + cds;
                        dest[di] = src[si];
                    }
                }
            }
        }

        return nb::ndarray<T, P...>(dest, array.ndim(), shape, owner);
    }

   public:
    constexpr static auto CLASSNAME = "GmicImage";
    gmic_image_py() : image() {}
    explicit gmic_image_py(CImg<T> &img) { img.move_to(image); }

    explicit gmic_image_py(CImg<T> &&img) { img.move_to(image); }

    template <typename... P>
    explicit gmic_image_py(nb::ndarray<T, nb::ndim<4>, P...> array)
        : image(array.data(), array.shape(3), array.shape(2), array.shape(1),
                array.shape(0), false)
    {
    }

    ~gmic_image_py() = default;

    static constexpr auto TO_NDARRAY_DOC =
        "Returns a ndarray wrapper of the underlying data of the image in its "
        "native (SDHW) order";
    nb::ndarray<T, nb::numpy, nb::device::cpu, nb::ndim<4>> to_native_ndarray()
    {
        return nb::ndarray<T, nb::numpy, nb::device::cpu, nb::ndim<4>>(
            image._data,
            {image._spectrum, image._depth, image._height, image._width},
            nb::handle());
    }

    template <typename... P>
    nb::ndarray<T, P...> to_ndarray(const string &dims_str, int64_t x,
                                    int64_t y, int64_t z, int64_t c, bool copy)
    {
        const auto self = nb::find(this);
        auto dims = parse_dims(dims_str);
        size_t ndims = dims_str.length();
        auto nshape = this->shape<size_t>();
        auto nstrides = this->strides<int64_t>();

        auto shape = reorder_to_spec(dims, nshape);
        auto strides = reorder_to_spec(dims, nstrides);

        auto arr = nb::ndarray<T, P...>(image._data, ndims, shape.data(), self,
                                        strides.data());

        if (copy)
            return copy_ndarray(arr);
        else
            return arr;
    }

    unsigned int width() { return image._width; }
    unsigned int height() { return image._height; }
    unsigned int depth() { return image._depth; }
    unsigned int spectrum() { return image._spectrum; }

    template <typename I = size_t>
    vector<I> strides()
    {
        constexpr I S = sizeof(T);
        return {S * width() * height() * depth(), S * width() * height(),
                S * width(), S};
    }

    nb::tuple strides_tuple()
    {
        const auto s = strides();
        return nb::make_tuple(s[0], s[1], s[2], s[3]);
    }

    template <typename I = size_t>
    vector<I> shape()
    {
        return {static_cast<I>(spectrum()), static_cast<I>(depth()),
                static_cast<I>(height()), static_cast<I>(width())};
    }

    nb::tuple shape_tuple()
    {
        return nb::make_tuple(spectrum(), depth(), height(), width());
    }

    [[nodiscard]] string str() const
    {
        stringstream out;
        out << "<" << nb::type_name(nb::type<gmic_image_py>()).c_str()
            << " at " << static_cast<const void *>(this)
            << ", data at: " << static_cast<const void *>(image._data)
            << ", w×h×s×d=" << image._width << "×" << image._height << "×"
            << image._spectrum << "×" << image._depth << ">";
        return out.str();
    }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_image_py<T>>(m, CLASSNAME)
            .def_ro_static("DIMS", &gmic_image_py::DIMS)
            .def(nb::init())
            .def(nb::init_implicit<nb::ndarray<T, nb::ndim<4>>>())
            .def("to_ndarray", &gmic_image_py::to_native_ndarray,
                 TO_NDARRAY_DOC, nb::rv_policy::reference_internal)
            .def("to_ndarray",
                 &gmic_image_py::to_ndarray<nb::numpy, nb::device::cpu>)
            .def_prop_ro("shape", &gmic_image_py::shape_tuple)
            .def_prop_ro("strides", &gmic_image_py::strides_tuple)
            .def("__str__", &gmic_image_py::str)
            .def("__repr__", &gmic_image_py::str);
    }
};

template <typename T = gmic_pixel_type>
class gmic_list_py {
    constexpr static auto CLASSNAME = "GmicList";
    CImgList<T> list;
    vector<gmic_image_py<T> *> data_py;

   public:
    explicit gmic_list_py() = default;

    explicit gmic_list_py(CImgList<T> &list) : data_py(list._width)
    {
        list.move_to(this->list);
    }

    explicit gmic_list_py(CImgList<T> &&list) : data_py(list._width)
    {
        list.move_to(this->list);
    }

    virtual ~gmic_list_py()
    {
        for (const auto *img : data_py)
            if (img != nullptr)
                delete img;
        data_py.clear();
    }

    class iterator
        : std::iterator<std::forward_iterator_tag, gmic_image_py<T>> {
        gmic_list_py<T> &list;
        unsigned int iter = 0;

       public:
        explicit iterator(gmic_list_py<T> &list, const unsigned int start = 0)
            : list(list), iter(start)
        {
        }

        iterator &operator++()
        {
            ++iter;
            return *this;
        }

        bool operator==(iterator other) const { return iter == other.iter; }
        bool operator!=(iterator other) const { return !operator==(other); }

        gmic_image_py<T> &operator*() const { return list[iter]; }
    };

    size_t size() { return list._width; }

    iterator begin() { return iterator(*this); }

    iterator end() { return iterator(*this, size()); }

    gmic_image_py<T> &operator[](const unsigned int i)
    {
        if (i >= size())
            throw out_of_range("Out of range or gmic_list_py object");
        if (i >= data_py.size())
            data_py.resize(i + 1);

        auto *img = data_py[i];
        if (img == nullptr) {
            img = data_py[i] = new gmic_image_py<T>(list._data[i]);
        }
        return *img;
    }

    [[nodiscard]] auto iter()
    {
        return nb::make_iterator(nb::type<gmic_list_py>(), "iterator", begin(),
                                 end());
    }

    string str()
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<gmic_list_py>()).c_str() << '[';
        bool first = true;
        for (const auto &image : *this) {
            if (first)
                first = false;
            else
                out << ", ";
            out << image.str();
        }
        out << "]>";

        return out.str();
    }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_list_py<T>>(m, CLASSNAME)
            .def("__iter__", &gmic_list_py::iter)
            .def("__len__", &gmic_list_py::size)
            .def("__str__", &gmic_list_py::str)
            .def("__getitem__", &gmic_list_py::operator[],
                 nb::rv_policy::reference_internal);
    }

    CImgList<float> &get_list() noexcept { return this->list; }

    /**
     * Resizes the gmic_image_py cache to the
     */
    void resize() { data_py.resize(list._width); }
};

namespace gmic_py {
constexpr static auto CLASSNAME = "Gmic";
template <typename T = gmic_pixel_type>
gmic_list_py<T> *run(gmic &gmic, const char *cmd, gmic_list_py<T> *img_list)
{
    if (img_list == nullptr)
        img_list = new gmic_list_py<T>();
    try {
        gmic_list<char> names;
        gmic.run(cmd, img_list->get_list(), names);
        img_list->resize();
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

string str(const gmic inst)
{
    stringstream out;
    out << '<' << nb::type_name(nb::type<gmic>()).c_str() << " object at "
        << static_cast<const void *>(&inst) << '>';
    return out.str();
}

template <typename T = gmic_pixel_type>
static void bind(const nb::module_ &m)
{
    nb::class_<gmic>(m, CLASSNAME)
        .def("run", &gmic_py::run<T>, "cmd"_a, "img_list"_a = nb::none(),
             nb::rv_policy::take_ownership)
        .def("__str__", &gmic_py::str)
        .def(nb::init());
}

}  // namespace gmic_py

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
    }

    m.def("inspect", &inspect, "array"_a, "Inspects a N-dimensional array");
    gmic_image_py<>::bind(m);
    gmic_list_py<>::bind(m);
    gmic_py::bind(m);

    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
}
}  // namespace gmicpy
