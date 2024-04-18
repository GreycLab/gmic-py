#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

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

template <typename T>
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
template <typename... Args>
static const char *assign_signature(char *buf, size_t N, const char *doc,
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
    for (size_t i = ndim; i >= 0; --i) {
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
        assign<Args...>(args...);
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

    template <class A>
    auto assign(A arr)
        -> enable_if_t<is_same<A, T4DArray<>>::value, gmic_image_py &>
    {
        image.assign(arr.shape(0), arr.shape(1), arr.shape(2), arr.shape(3));
        if (is_c_contig(arr)) {
            std::copy_n(arr.data(), arr.size(), image.data());
        }
        else {
            auto v = arr.view();
            for (size_t c = 0; c < image.spectrum(); c++)
                for (size_t d = 0; d < image.depth(); d++)
                    for (size_t y = 0; y < image.depth(); y++)
                        for (size_t x = 0; x < image.depth(); x++)
                            image(x, y, d, c) = v(x, y, d, c);
        }
        return *this;
    }

    static constexpr auto TO_NDARRAY_DOC =
        "Returns a ndarray wrapper of the underlying data of the image in "
        "its "
        "native (SDHW) order";
    nb::ndarray<T, nb::numpy, nb::device::cpu, nb::ndim<4>> to_native_ndarray()
    {
        return nb::ndarray<T, nb::numpy, nb::device::cpu, nb::ndim<4>>(
            image.data(),
            {image._spectrum, image._depth, image._height, image._width},
            nb::handle());
    }

    template <typename... P>
    TNDArray<P...> to_ndarray(const string &dims_str, bool copy)
    {
        const auto self = nb::find(this);
        auto dims = parse_dims(dims_str);
        size_t ndims = dims_str.length();
        auto nshape = this->shape<size_t>();
        auto nstrides = this->strides<int64_t>();

        auto shape = reorder_to_spec(dims, nshape);
        auto strides = reorder_to_spec(dims, nstrides);

        auto arr = TNDArray<P...>(image.data(), ndims, shape.data(), self,
                                  strides.data());

        if (copy)
            return copy_ndarray<P...>(arr);
        else
            return arr;
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
            << ", w×h×s×d=" << image.width() << "×" << image.height() << "×"
            << image.spectrum() << "×" << image.depth() << ">";
        return out.str();
    }

    [[nodiscard]] CImgT &get_image() noexcept { return image; }

    [[nodiscard]] const CImgT &get_image() const noexcept { return image; }

    static void bind(nb::module_ &m)
    {
        auto cls =
            nb::class_<gmic_image_py<T>>(m, CLASSNAME)
                .def_ro_static("DIMS", &gmic_image_py::DIMS)
                .def("to_ndarray", &gmic_image_py::to_native_ndarray,
                     TO_NDARRAY_DOC, nb::rv_policy::reference_internal)
                .def("to_ndarray",
                     &gmic_image_py::to_ndarray<nb::numpy, nb::device::cpu>)
                .def_prop_ro("shape", &gmic_image_py::shape_tuple)
                .def_prop_ro("strides", &gmic_image_py::strides_tuple)
                .def("__str__", &gmic_image_py::str)
                .def("__repr__", &gmic_image_py::str);
        char doc_buf[1024];
#define ARGS(...) __VA_ARGS__
#define IMAGE_ASSIGN(doc, TYPES, ...)                                    \
    cls.def(nb::init<TYPES>(),                                           \
            Trans::assign_signature<TYPES>(doc_buf, size(doc_buf), doc,  \
                                           "CImg<T>"),                   \
            ##__VA_ARGS__)                                               \
        .def("assign",                                                   \
             static_cast<gmic_image_py &(gmic_image_py::*)(TYPES)>(      \
                 &gmic_image_py::assign<TYPES>),                         \
             Trans::assign_signature<TYPES>(doc_buf, size(doc_buf), doc, \
                                            "CImg<T>::assign"),          \
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
        IMAGE_ASSIGN("Construct an image from an ndarray", ARGS(T4DArray<>),
                     "array"_a);
    }
#undef IMAGE_ASSIGN
};

template <typename T = gmic_pixel_type>
class gmic_list_py {
   public:
    typedef CImgList<T> CImgListT;

   private:
    constexpr static auto CLASSNAME = "GmicList";
    CImgListT list;
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
            img = data_py[i] = new gmic_image_py<T>(list(i), true);
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

    [[nodiscard]] CImgList<float> &get_list() noexcept { return this->list; }
    [[nodiscard]] const CImgList<float> &get_list() const noexcept
    {
        return this->list;
    }

    /**
     * Resizes the gmic_image_py cache to the
     */
    void resize() { data_py.resize(list._width); }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_list_py<T>>(m, CLASSNAME)
            .def("__iter__", &gmic_list_py::iter)
            .def("__len__", &gmic_list_py::size)
            .def("__str__", &gmic_list_py::str)
            .def("__getitem__", &gmic_list_py::operator[],
                 nb::rv_policy::reference_internal);
    }
};

namespace interpreter_py {
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
        .def("run", &interpreter_py::run<T>, "cmd"_a,
             "img_list"_a = nb::none(), nb::rv_policy::take_ownership)
        .def("__str__", &interpreter_py::str)
        .def(nb::init());
}
}  // namespace interpreter_py

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
    interpreter_py::bind(m);

    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
}

}  // namespace gmicpy
