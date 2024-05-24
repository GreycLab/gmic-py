#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

// Include gmic et CImg after nanobind
#include <CImg.h>
#include <gmic.h>

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

static const char *const ARRAY_INTERFACE = "__array_interface__";
static const char *const DLPACK_INTERFACE = "__dlpack__";
static const char *const DLPACK_DEVICE_INTERFACE = "__dlpack_device__";

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

template <typename>
class gmic_image_py;

template <typename T>
class gmic_list_py;

/**
 * Class that manages the translation of values back and forth between the
 * python binding and the native GMIC/CImg library. Within this class,
 * "translating" describes the python binding -> libgmic direction, and
 * "untranslating" the opposite process.
 */
namespace Trans {

/// Get a void pointer from any pointer, for storing a into a registry
template <typename A>
[[maybe_unused]] static inline void *get_void_p(A *val)
{
    return (void *)val;
}

/// Get a void pointer from a reference, for storing a into a registry
template <typename A>
static inline void *get_void_p(A &val)
{
    return (void *)&val;
}

/**
 * \defgroup substitute-un (Un)Substitution methods
 * substitute(A) is defined for every A we want to "unwrap" before calling
 * gmic, and unsubstitute(A) is the other way around.
 * \remark Some unsubstitute(A) are registry-only and will always throw
 * exceptions if called. They exist for unsubstitutions that are only valid
 * if the value to return was unwrapped during translation in a first
 * place, they are not valid for new values.
 * \remark There is no requirement that translatable types should be
 * untranslatable or vice versa
 * \remark There are however static_assert's ensuring
 * "stability" of both translation and untranslation : none of the
 * (un)substitute functions should return a type that another overload
 * would accept as argument.
 * @{
 */
/// gmic_image_py&lt;T&gt; -> CImg&lt;T&gt; unwrapping
template <typename gmic_image_py_t>
[[maybe_unused]] static inline auto substitute(gmic_image_py_t img)
    -> decltype(img.get_image())
{
    return img.get_image();
}

/// gmic_list_py&lt;T&gt -> CImgList&lt;T&gt unwrapping
template <typename gmic_list_py_t>
[[maybe_unused]] static inline auto substitute(gmic_list_py_t list)
    -> decltype(list.get_list())
{
    return list.get_list();
}

/// std::string&lt;char_t&gt; -> const char_t* unwrapping
template <typename string_t>
[[maybe_unused]] static inline auto substitute(string_t str)
    -> decltype(str.c_str())
{
    return str.c_str();
}

/// CImg&lt;T&gt -> gmic_image_py&lt;T&gt wrapping
template <typename A,
          enable_if_t<is_same_v<A, CImg<typename A::value_type>>, bool> = true>
[[maybe_unused]] static inline auto unsubstitute(A img)
{
    return gmic_image_py<typename A::value_type>(img);
}

/// CImg&lt;T&gt& -> gmic_image_py&lt;T&gt& wrapping (registry-only)
template <typename A, enable_if_t<is_same_v<A, CImg<typename A::value_type> &>,
                                  bool> = true>
[[maybe_unused]] [[noreturn]] static inline auto unsubstitute(A &)
{
    throw runtime_error("Cannot untranslate a CImg<T> reference");
}

/// CImgList&lt;T&gt& -> gmic_list_py&lt;T&gt& wrapping
template <
    typename A,
    enable_if_t<is_same_v<A, CImgList<typename A::value_type>>, bool> = true>
[[maybe_unused]] static inline auto unsubstitute(A list)
{
    return gmic_list_py<typename A::value_type>(list);
}

/// CImgList&lt;T&gt& -> gmic_list_py&lt;T&gt& wrapping (registry-only)
template <
    typename A,
    enable_if_t<is_same_v<A, CImgList<typename A::value_type> &>, bool> = true>
[[maybe_unused]] static inline auto unsubstitute(A &)
{
    throw runtime_error("Cannot untranslate a CImgList<T> reference");
}
/// @}

/**
 * The registry is a simple map that will register every effective
 * translation of values during argument translation, so that if any value
 * to be untranslated back, its predecessor can be returned instead
 *
 * \remark can only be recorded translations where both the input and
 * output values are either a pointer or a reference
 */
using registry = std::map<void *, pair<void *, const type_info *>>;

/**
 * \defgroup helper-types
 * @{
 * Helper type to check whether a type A is translatable
 */
template <typename A, typename = void>
struct is_translatable : false_type {
    /**
     * alias for:
     * - the return type of substitute(A) if defined
     * - A otherwise */
    using result = A;
};

template <typename A>
struct is_translatable<A, void_t<decltype(substitute(declval<A>()))>>
    : true_type {
    using result = decltype(substitute(declval<A>()));
};

/**
 * Helper type to check whether a type A is untranslatable
 */
template <typename A, typename = void>
struct is_untranslatable : false_type {
    /**
     * alias for:
     * - the return type of unsubstitute(A) if defined
     * - A otherwise */
    using result = A;
};

template <typename A>
struct is_untranslatable<A, void_t<decltype(unsubstitute(declval<A>()))>>
    : true_type {
    using result = decltype(unsubstitute(declval<A>()));
};

template <typename A>
using translated = typename is_translatable<A>::result;
template <typename A>
using untranslated = typename is_untranslatable<A>::result;
/// }@

/**
 * Entry point of the translation system
 * @tparam A Type to translate
 * @param a argument to translate
 * @param reg registry to record translations
 * @return substitute(a) if a suitable overload exists, otherwise a
 */
template <typename A>
[[nodiscard]] static inline translated<A> translate(A a,
                                                    registry *reg = nullptr)
{
    using B = translated<A>;
    if constexpr (is_translatable<A>::value) {
        B b = substitute(a);
        static_assert(is_same_v<B, decltype(translate<B>(declval<B>()))>,
                      "Translated value is not translate-stable");
        if constexpr ((is_lvalue_reference_v<A> || is_pointer_v<A>) &&
                      (is_lvalue_reference_v<B> || is_pointer_v<B>)) {
            if (reg && (void *)&a != (void *)&b) {
                reg->emplace(get_void_p(b),
                             make_pair(get_void_p(a), &typeid(A)));
            }
        }
        return b;
    }
    else {
        return a;
    }
}

/**
 * Entry point of the untranslation system
 * @tparam A Type to untranslate
 * @param a argument to untranslate
 * @param reg registry to check
 * @return substitute(a) if a suitable overload exists, otherwise a
 */
template <typename A>
[[nodiscard]] static inline untranslated<A> untranslate(
    A a, registry *reg = nullptr)
{
    if constexpr (is_untranslatable<A>::value) {
        using B = translated<A>;
        static_assert(is_same_v<B, decltype(untranslate<B>(declval<B>()))>,
                      "Untranslated value is not untranslate-stable");
        if (reg) {
            auto it = reg->find(&a);
            if (it != reg->end()) {
                auto [b, t1] = it->second;
                const type_info &t2 = typeid(B);
                if (*t1 != t2) {
                    throw runtime_error(
                        string("Mismatched un/translated typeid. In: ") +
                        t1->name() + ", Out: " + t2.name());
                }
                return *static_cast<decltype(unsubstitute<A>(declval<A>())) *>(
                    b);
            }
        }
        return unsubstitute<A>(a);
    }
    else {
        return a;
    }
}

/**
 * Appends the signature of a given function, with its arguments' types
 * translated, to a given docstring, and writes it to a char buffer
 * @tparam Args Types of the pre-translation arguments
 * @param buf Char buffer to write to
 * @param N Buffer size
 * @param doc Documentation to append the signature to
 * @param func Name to use in the signature for the documented function
 * @return buf passthrough
 */
template <typename... Args, size_t N = 1024>
static const char *assign_signature(char buf[N], const char *doc,
                                    const char *func)
{
    char *max = buf + N;
    buf += snprintf(buf, N, "%s\n\nBinds %s(", doc, func);
    vector<const char *> argtypes{typeid(translated<Args>).name()...};
    bool first = true;
    for (auto &t : argtypes) {
        if (first) {
            first = false;
            buf += snprintf(buf, max - buf, "%s", t);
        }
        else
            buf += snprintf(buf, max - buf, ", %s", t);
    }
    buf += snprintf(buf, max - buf, ")");
    if (buf == max)
        throw runtime_error("Ran out of char buffer");
    return buf;
}
};  // namespace Trans

template <typename Sh, typename St>
bool is_c_contig(unsigned short ndim, Sh *shape, St *strides)
{
    decltype((*shape) * (*strides)) acc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != acc)
            return false;
        acc *= shape[i];
    }
    return true;
}

template <typename... P>
bool is_c_contig(nb::ndarray<P...> arr)
{
    return is_c_contig(arr.ndim(), arr.shape_ptr(), arr.stride_ptr());
}

template <typename T = gmic_pixel_type>
class gmic_image_py {
   public:
    typedef CImg<T> CImgT;

   private:
    template <typename... P>
    using TNDArray = nb::ndarray<T, nb::device::cpu, P...>;
    template <typename... P>
    using T4DArray = TNDArray<nb::ndim<4>, P...>;

    CImgT image;

    /**
     * Copies a ndarray. Will reorder the data so that the data is
     * C-contiguous.
     * @tparam P Optional ndarray specifiers
     * @param array Input array whose data is <strong>in SDHW (gmic)
     * order</strong>
     * @return A copy of the ndarray with the same data for a given set of
     * coordinates, but reordered C-style
     */
    template <typename... P>
    TNDArray<P...> copy_ndarray(TNDArray<P...> array)
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

        return TNDArray<P...>(dest, array.ndim(), shape, owner);
    }

   public:
    constexpr static auto CLASSNAME = "GmicImage";
    gmic_image_py() : image() {}

    gmic_image_py(const gmic_image_py &other) : image(other.image) {}

    gmic_image_py(gmic_image_py &&other) noexcept
        : image(std::move(other.image))
    {
    }

    template <typename... Args>
    explicit gmic_image_py(Args... args)
    {
        assign(args...);
    }

    ~gmic_image_py() = default;

    template <typename... Args>
    auto assign(Args... args)
        -> enable_if_t<is_lvalue_reference<decltype(CImgT{}.assign(
                           declval<Trans::translated<Args>>()...))>::value,
                       gmic_image_py &>
    {
        image.assign(Trans::translate<Args>(args)...);
        return *this;
    }

    template <class... P>
    gmic_image_py &assign(TNDArray<P...> arr)
    {
        if (arr.ndim() == 0 || arr.ndim() > 4) {
            throw nb::value_error(
                "Invalid ndarray dimensions for image "
                "(should be 1 <= N <= 4)");
        }
        const auto N = arr.ndim();
        array<size_t, 4> dim{1, 1, 1, 1}, strides{1, 1, 1, 1};
        auto assign = [&](size_t from, size_t to) {
            dim[to] = arr.shape(from);
            strides[to] = static_cast<size_t>(arr.stride(from));
        };
        switch (N) {
            case 4:
                assign(1, 2);
            case 3:
                assign(0, 3);
            case 2:
                assign(N - 2, 1);
            default:
                assign(N - 1, 0);
        }
        image.assign(dim[0], dim[1], dim[2], dim[3]);
        if (is_c_contig(arr)) {
            std::copy_n(arr.data(), arr.size(), image.data());
        }
        else {
            for (size_t c = 0; c < dim[3]; c++) {
                const size_t offc = c * strides[3];
                for (size_t d = 0; d < dim[2]; d++) {
                    const size_t offd = offc + d * strides[2];
                    for (size_t y = 0; y < dim[1]; y++) {
                        const size_t offy = offd + y * strides[1];
                        for (size_t x = 0; x < dim[0]; x++) {
                            image(x, y, d, c) = offy + x * strides[0];
                        }
                    }
                }
            }
        }
        return *this;
    }

    template <typename t, class... P>
        requires(!same_as<t, T>)
    gmic_image_py &assign(nb::ndarray<t, nb::device::cpu, P...> arr)
    {
        gmic_image_py<t> img(arr);
        assign(img);
        return *this;
    }

    gmic_image_py &assign(nb::object &obj)
    {
        if (!hasattr(obj, ARRAY_INTERFACE))
            throw nb::next_overload("Missing __array_interface__");
        nb::dict ai = obj.attr(ARRAY_INTERFACE);
    }

    template <typename... P>
    T4DArray<P...> as_ndarray()
    {
        return T4DArray<P...>(
            image.data(),
            {image._spectrum, image._depth, image._height, image._width},
            nb::handle());
    }

    template <typename... P>
    T4DArray<P...> to_ndarray()
    {
        return copy_ndarray(as_ndarray<P...>());
    }

    auto dlpack_device() { return nb::make_tuple(nb::device::cpu::value, 0); }

    nb::object array_interface()
    {
        auto arr = as_ndarray<nb::numpy>();
        return cast(arr).attr(ARRAY_INTERFACE);
    }

    unsigned int width() { return image.width(); }
    unsigned int height() { return image.height(); }
    unsigned int depth() { return image.depth(); }
    unsigned int spectrum() { return image.spectrum(); }

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
            << ", data at: " << static_cast<const void *>(image.data())
            << ", w×h×d×s=" << image.width() << "×" << image.height() << "×"
            << image.depth() << "×" << image.spectrum() << ">";
        return out.str();
    }

    [[nodiscard]] explicit operator string() const { return str(); }

    [[nodiscard]] CImgT &get_image() noexcept { return image; }

    [[nodiscard]] const CImgT &get_image() const noexcept { return image; }

    static void bind(nb::module_ &m)
    {
        auto cls =
            nb::class_<gmic_image_py<T>>(m, CLASSNAME)
                .def(DLPACK_INTERFACE, &gmic_image_py::as_ndarray<>,
                     nb::rv_policy::reference_internal)
                .def(DLPACK_DEVICE_INTERFACE, &gmic_image_py::dlpack_device)
                .def_prop_ro(ARRAY_INTERFACE, &gmic_image_py::array_interface,
                             nb::rv_policy::reference_internal)
                .def("as_dlpack", &gmic_image_py::as_ndarray<>,
                     nb::rv_policy::reference_internal,
                     "Returns a view of the underlying data as a"
                     " DLPack capsule")
                .def("to_dlpack", &gmic_image_py::to_ndarray<>,
                     nb::rv_policy::reference_internal,
                     "Returns a copy of the underlying data as a"
                     " DLPack capsule")
                .def(
                    "as_numpy", &gmic_image_py::as_ndarray<nb::numpy>,
                    nb::rv_policy::reference_internal,
                    "Returns a view of the underlying data as a Numpy NDArray")
                .def(
                    "to_numpy", &gmic_image_py::to_ndarray<nb::numpy>,
                    nb::rv_policy::reference_internal,
                    "Returns a copy of the underlying data as a Numpy NDArray")
                .def_prop_ro("shape", &gmic_image_py::shape_tuple,
                             "Tuple of dimensions of this image")
                .def_prop_ro("strides", &gmic_image_py::strides_tuple,
                             "Tuple of strides of this image")
                .def_prop_ro("width", &gmic_image_py::width,
                             "Width (1st dimension) of the image")
                .def_prop_ro("height", &gmic_image_py::height,
                             "Height (2nd dimension) of the image")
                .def_prop_ro("depth", &gmic_image_py::depth,
                             "Depth (3rd dimension) of the image")
                .def_prop_ro(
                    "spectrum", &gmic_image_py::spectrum,
                    "Spectrum (i.e. channels, 4th dimension) of the image")
                .def("__str__", &gmic_image_py::str)
                .def("__repr__", &gmic_image_py::str);
        char doc_buf[1024];
#define ARGS(...) __VA_ARGS__
#define IMAGE_ASSIGN(doc, TYPES, ...)                                         \
    cls.def(nb::init<TYPES>(),                                                \
            Trans::assign_signature<TYPES>(doc_buf, doc, "CImg<T>"),          \
            ##__VA_ARGS__)                                                    \
        .def("assign",                                                        \
             static_cast<gmic_image_py &(gmic_image_py::*)(TYPES)>(           \
                 &gmic_image_py::assign),                                     \
             Trans::assign_signature<TYPES>(doc_buf, doc, "CImg<T>::assign"), \
             nb::rv_policy::none, ##__VA_ARGS__)
        // Bindings for CImg constructors and assign()'s
        IMAGE_ASSIGN("Construct an empty image", ARGS());
        IMAGE_ASSIGN("Construct image copy", ARGS(gmic_image_py &), "other"_a);
        IMAGE_ASSIGN("Advanced copy constructor", ARGS(gmic_image_py &, bool),
                     "other"_a, "is_shared"_a);
        IMAGE_ASSIGN(
            "Construct image with specified size",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int),
            "width"_a, "height"_a, "depth"_a = 1, "channels"_a = 1);
        IMAGE_ASSIGN(
            "Construct image with specified size and initialize pixel "
            "values",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int, T &),
            "width"_a, "height"_a, "depth"_a, "channels"_a, "value"_a);
        IMAGE_ASSIGN(
            "Construct image with specified size and initialize pixel "
            "values from a value string",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int,
                 string &, bool),
            "width"_a, "height"_a, "depth"_a, "channels"_a, "value_string"_a,
            "repeat"_a);
        IMAGE_ASSIGN("Construct image from reading an image file",
                     ARGS(string), "filename"_a);
        IMAGE_ASSIGN("Construct image copy", ARGS(gmic_image_py &));
        IMAGE_ASSIGN(
            "Construct image with dimensions borrowed from another image",
            ARGS(gmic_image_py &, string &), "other"_a, "dimensions"_a);
        IMAGE_ASSIGN(
            "Construct image with dimensions borrowed from another image and "
            "initialize pixel values",
            ARGS(gmic_image_py &, string &, T &), "other"_a, "dimensions"_a,
            "value"_a);

        // gmic-py specific bindings
        IMAGE_ASSIGN("Construct an image from an ndarray", ARGS(TNDArray<>),
                     "array"_a);
    }
#undef IMAGE_ASSIGN
};

template <typename T, typename I>
class gmic_list_base {
   public:
    using Item = I;
    using List = CImgList<T>;

   protected:
    List list;

   public:
    explicit gmic_list_base() = default;

    explicit gmic_list_base(List &list) : list(list) {}

    explicit gmic_list_base(List &&list) : list(list) {}

    class iterator : std::iterator<std::forward_iterator_tag, Item> {
        gmic_list_base &list;
        unsigned int iter = 0;

       public:
        explicit iterator(gmic_list_base &list, const unsigned int start = 0)
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

        auto operator*() const { return list[iter]; }
    };

    size_t size() { return list._width; }

    iterator begin() { return iterator(*this); }

    iterator end() { return iterator(*this, size()); }

    virtual Item operator[](unsigned int i) = 0;

    virtual void set(unsigned int, Item) = 0;

    [[nodiscard]] auto iter()
    {
        return nb::make_iterator(nb::type<decltype(*this)>(), "iterator",
                                 begin(), end());
    }

    string str()
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<decltype(*this)>()).c_str()
            << '[';
        bool first = true;
        for (const auto &image : *this) {
            if (first)
                first = false;
            else
                out << ", ";
            out << (string)image;
        }
        out << "]>";

        return out.str();
    }

    [[nodiscard]] List &get_list() noexcept { return this->list; }
    [[nodiscard]] const List &get_list() const noexcept { return this->list; }

    /**
     * Resizes the gmic_image_py cache to the
     */
    virtual void resize() {}

    template <class C>
    static void bind(nb::module_ &m, const char *classname)
    {
        nb::class_<C>(m, classname)
            .def("__iter__", &gmic_list_base::iter)
            .def("__len__", &gmic_list_base::size)
            .def("__str__", &gmic_list_base::str)
            .def("__getitem__", &gmic_list_base::operator[], "i"_a,
                 nb::rv_policy::reference_internal)
            .def("__setitem__", &gmic_list_base::set, "i"_a, "v"_a);
    }
};

template <typename T = gmic_pixel_type>
class gmic_list_py : public gmic_list_base<T, gmic_image_py<T> &> {
   private:
    using Base = gmic_list_base<T, gmic_image_py<T>>;
    vector<gmic_image_py<T> *> data_py;

   public:
    constexpr static auto CLASSNAME = "GmicList";

    explicit gmic_list_py() = default;

    explicit gmic_list_py(CImgList<T> &list) : data_py(list._width), Base(list)
    {
    }

    explicit gmic_list_py(CImgList<T> &&list)
        : data_py(list._width), Base(list)
    {
    }

    virtual ~gmic_list_py()
    {
        for (const auto *img : data_py)
            if (img != nullptr)
                delete img;
        data_py.clear();
    }

    Base::Item &operator[](const unsigned int i) override
    {
        if (i >= this->size())
            throw out_of_range("Out of range or gmic_list_py object");
        if (i >= data_py.size())
            data_py.resize(i + 1);

        auto *img = data_py[i];
        if (img == nullptr) {
            img = data_py[i] = new gmic_image_py<T>(this->list(i), true);
        }
        return *img;
    }

    void set(unsigned int i, Base::Item &item) override
    {
        this->list(i).assign(item.get_image());
    }

    void resize() override { data_py.resize(this->size()); }

    static void bind(nb::module_ &m)
    {
        Base::template bind<gmic_list_py>(m, CLASSNAME);
    }
};

class gmic_charlist_py : public gmic_list_base<char, string> {
   private:
    using Base = gmic_list_base<char, string>;

   public:
    constexpr static auto CLASSNAME = "GmicList";

    explicit gmic_charlist_py() = default;

    explicit gmic_charlist_py(List &list) : Base(list) {}

    explicit gmic_charlist_py(List &&list) : Base(list) {}

    Item operator[](const unsigned int i) override { return {this->list(i)}; }

    void set(unsigned int i, Item item) override
    {
        this->list(i).assign(CImg<char>::string(item.c_str()));
    }

    static void bind(nb::module_ &m)
    {
        Base::template bind<gmic_charlist_py>(m, CLASSNAME);
    }
};

namespace interpreter_py {
constexpr static auto CLASSNAME = "Gmic";
template <typename T = gmic_pixel_type>
gmic_list_py<T> *run(gmic &gmic, const char *cmd, gmic_list_py<T> *img_list,
                     optional<nb::handle> img_names)
{
    if (img_list == nullptr)
        img_list = new gmic_list_py<T>();

    CImgList<char> _names, *names = &_names;

    if (img_names) {
        try {
            auto &charlist = nb::cast<gmic_charlist_py &>(*img_names);
            names = &charlist.get_list();
        }
        catch (nb::cast_error &) {
            auto list = nb::cast<nb::list>(*img_names);
            names->assign(list.size());
            for (size_t i = 0; i < list.size(); i++) {
                auto str = nb::cast<const char *>(list[i]);
                CImg<char>::string(str).move_to(*names, i);
            }
        }
    }

    try {
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

template <typename T = gmic_pixel_type>
gmic_list_py<T> *static_run(const char *cmd, gmic_list_py<T> *img_list,
                            optional<nb::handle> img_names)
{
    static unique_ptr<gmic> inter;
    if (!inter)
        inter = make_unique<gmic>();
    return inter->run(cmd, img_list, img_names);
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
        .def("run", &interpreter_py::run<T>, "cmd"_a,
             "img_list"_a = nb::none(), "img_names"_a = nb::none(),
             nb::rv_policy::take_ownership)
        .def("__str__", &interpreter_py::str)
        .def(nb::init());
}
}  // namespace interpreter_py

NB_MODULE(gmic, m)
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

    static_assert(Trans::is_translatable<string>::value);

    m.def("inspect", &inspect, "array"_a, "Inspects a N-dimensional array");
    static_assert(Trans::is_translatable<gmic_image_py<> &>::value);
    gmic_image_py<>::bind(m);
    static_assert(Trans::is_translatable<gmic_list_py<> &>::value);
    gmic_list_py<>::bind(m);
    gmic_charlist_py::bind(m);
    interpreter_py::bind(m);

    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
}

}  // namespace gmicpy
