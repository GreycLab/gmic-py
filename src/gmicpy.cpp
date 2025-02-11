#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

// Include gmic et CImg after nanobind
#include <CImg.h>
#include <gmic.h>

#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

static const char *const ARRAY_INTERFACE = "__array_interface__";
static const char *const DLPACK_INTERFACE = "__dlpack__";
static const char *const DLPACK_DEVICE_INTERFACE = "__dlpack_device__";

/// Debug function
string inspect(const nb::ndarray<nb::ro> &a)
{
    stringstream buf;
    buf << "Array :\n\tdata pointer : " << a.data() << endl;
    buf << "\tdimensions : " << a.ndim() << endl;
    for (size_t i = 0; i < a.ndim(); ++i) {
        buf << "\t\t[" << i << "]: size=" << a.shape(i)
            << ", stride=" << a.stride(i) << endl;
    }

    buf << "\tdevice = " << a.device_id() << "(";
    if (a.device_type() == nb::device::cpu::value) {
        buf << "CPU)" << endl;
    }
    else if (a.device_type() == nb::device::cuda::value) {
        buf << "CUDA)" << endl;
    }
    else {
        buf << "<unknown>)" << endl;
    }
    buf << "\tdtype: ";
    auto dtypes = {make_pair(nb::dtype<int8_t>(), "int8_t"),
                   make_pair(nb::dtype<int16_t>(), "int16_t"),
                   make_pair(nb::dtype<int32_t>(), "int32_t"),
                   make_pair(nb::dtype<int64_t>(), "int64_t"),
                   make_pair(nb::dtype<uint8_t>(), "uint8_t"),
                   make_pair(nb::dtype<uint16_t>(), "uint16_t"),
                   make_pair(nb::dtype<uint32_t>(), "uint32_t"),
                   make_pair(nb::dtype<uint64_t>(), "uint64_t"),
                   make_pair(nb::dtype<float>(), "float"),
                   make_pair(nb::dtype<double>(), "double"),
                   make_pair(nb::dtype<bool>(), "bool")};
    auto dt = find_if(dtypes.begin(), dtypes.end(),
                      [&](auto pair) { return pair.first == a.dtype(); });
    if (dt != dtypes.end()) {
        buf << dt->second << endl;
    }
    else {
        buf << "<unknown>" << endl;
    }
    return buf.str();
}

/**
 * Class that manages the translation of values back and forth between the
 * python binding and the native GMIC/CImg library. Within this class,
 * "translating" describes the python binding -> libgmic direction, and
 * "untranslating" the opposite process.
 */
namespace Trans {

/// Get a void pointer from any pointer or reference, for storing a into a
/// registry
template <class A>
    requires is_reference_v<A> || is_pointer_v<A>
static inline void *get_void_p(A val)
{
    if constexpr (is_reference_v<A>) {
        return (void *)&val;
    }
    else {
        return (void *)val;
    }
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
template <class gmic_image_py_t>
[[maybe_unused]] static inline auto substitute(gmic_image_py_t img)
    -> decltype(img.get_image())
{
    return img.get_image();
}

/// gmic_list_py&lt;T&gt -> CImgList&lt;T&gt unwrapping
template <class gmic_list_py_t>
[[maybe_unused]] static inline auto substitute(gmic_list_py_t list)
    -> decltype(list.get_list())
{
    return list.get_list();
}

/// std::string&lt;char_t&gt; -> const char_t* unwrapping
template <class string_t>
[[maybe_unused]] static inline auto substitute(
    string_t str)  // NOLINT(*-unnecessary-value-param)
    -> decltype(str.c_str())
{
    return str.c_str();
}

/// CImg&lt;T&gt -> gmic_image_py&lt;T&gt wrapping
template <class A,
          enable_if_t<is_same_v<A, CImg<typename A::value_type>>, bool> = true>
[[maybe_unused]] static inline auto unsubstitute(A img)
{
    return gmic_image_py<typename A::value_type>(img);
}

/// CImg&lt;T&gt& -> gmic_image_py&lt;T&gt& wrapping (registry-only)
template <class A, enable_if_t<is_same_v<A, CImg<typename A::value_type> &>,
                               bool> = true>
[[maybe_unused]] [[noreturn]] static inline auto unsubstitute(A &)
{
    throw runtime_error("Cannot untranslate a CImg<T> reference");
}

/// CImgList&lt;T&gt& -> gmic_list_py&lt;T&gt& wrapping
template <class A, enable_if_t<is_same_v<A, CImgList<typename A::value_type>>,
                               bool> = true>
[[maybe_unused]] static inline auto unsubstitute(A list)
{
    return gmic_list_py<typename A::value_type>(list);
}

/// CImgList&lt;T&gt& -> gmic_list_py&lt;T&gt& wrapping (registry-only)
template <
    class A,
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
template <class A, class = void>
struct is_translatable : false_type {
    /**
     * alias for:
     * - the return type of substitute(A) if defined
     * - A otherwise */
    using result = A;
};

template <class A>
struct is_translatable<A, void_t<decltype(substitute(declval<A>()))>>
    : true_type {
    using result = decltype(substitute(declval<A>()));
};

/**
 * Helper type to check whether a type A is untranslatable
 */
template <class A, class = void>
struct is_untranslatable : false_type {
    /**
     * alias for:
     * - the return type of unsubstitute(A) if defined
     * - A otherwise */
    using result = A;
};

template <class A>
struct is_untranslatable<A, void_t<decltype(unsubstitute(declval<A>()))>>
    : true_type {
    using result = decltype(unsubstitute(declval<A>()));
};

template <class A>
using translated = typename is_translatable<A>::result;
template <class A>
using untranslated = typename is_untranslatable<A>::result;
/// }@

/**
 * Entry point of the translation system
 * @tparam A Type to translate
 * @param a argument to translate
 * @param reg registry to record translations
 * @return substitute(a) if a suitable overload exists, otherwise a
 */
template <class A>
[[nodiscard]]
static inline translated<A> translate(A a, registry *reg = nullptr)
{
    using B = translated<A>;
    if constexpr (is_translatable<A>::value) {
        B b = substitute(a);
        static_assert(is_same_v<B, decltype(translate<B>(declval<B>()))>,
                      "Translated value is not translate-stable");
        if constexpr ((is_lvalue_reference_v<A> || is_pointer_v<A>) &&
                      (is_lvalue_reference_v<B> || is_pointer_v<B>)) {
            if (reg && (void *)&a != (void *)&b) {
                reg->emplace(get_void_p<B>(b),
                             make_pair(get_void_p<A>(a), &typeid(A)));
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
template <class A>
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
template <class... Args, size_t N = 1024>
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

template <class Sh, class St>
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

template <class... P>
bool is_c_contig(nb::ndarray<P...> arr)  // NOLINT(*-unnecessary-value-param)
{
    return is_c_contig(arr.ndim(), arr.shape_ptr(), arr.stride_ptr());
}

template <class T = gmic_pixel_type>
class gmic_image_py {
   public:
    using Img = CImg<T>;
    /// ndarray of type T on the CPU
    template <class... P>
    using TNDArray = nb::ndarray<T, nb::device::cpu, P...>;
    /// read-only ndarray of type T on the CPU
    template <class... P>
    using CTNDArray = nb::ndarray<const T, nb::device::cpu, P...>;

    /**
     * Copies a ndarray. Will reorder the data so that the data is
     * C-contiguous.
     * @tparam P Optional ndarray specifiers
     * @param array Input array whose data is <strong>in SDHW (gmic)
     * order</strong>
     * @return A copy of the ndarray with the same data for a given set of
     * coordinates, but reordered C-style
     */
    template <class... P>
    static TNDArray<P...> copy_ndarray(const CTNDArray<P...> &array)
    {
        const T *src = array.data();
        T *dest = new T[array.size()];
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

    constexpr static auto CLASSNAME = "Image";

    template <class I, class... Args>
    struct can_native_init : false_type {};

    template <class... Args>
    struct can_native_init<
        enable_if_t<
            is_same_v<Img,
                      decltype(Img(declval<Trans::translated<Args>>()...))>,
            Img>,
        Args...> : true_type {};

    template <class... Args>
    static void new_image(Img *img, Args... args)
    {
        if constexpr (can_native_init<Img, Args...>::value) {
            new (img) Img(Trans::translate<Args>(args)...);
        }
        else {
            new (img) Img();
            assign(*img, args...);  // NOLINT(*-unnecessary-value-param)
        }
    }

    template <class... Args>
    static auto assign(Img &img, Args... args)
        -> enable_if_t<is_lvalue_reference_v<decltype(Img{}.assign(
                           declval<Trans::translated<Args>>()...))>,
                       Img &>
    {
        img.assign(Trans::translate<Args>(args)...);
        return img;
    }

    template <class... P>
    static Img &assign(Img &img, CTNDArray<P...> arr)
    {
        if (arr.ndim() == 0 || arr.ndim() > 4) {
            throw nb::value_error(
                "Invalid ndarray dimensions for image "
                "(should be 1 <= N <= 4)");
        }
        const auto N = arr.ndim();
        array<size_t, 4> dim{1, 1, 1, 1}, strides{1, 1, 1, 1};
        for (size_t i = 0; i < N; i++) {
            dim[i] = arr.shape(i);
            strides[i] = static_cast<size_t>(arr.stride(i));
        }
        return assign(img, arr, dim, strides);
    }

    template <class... P>
    static Img &assign(Img &img, CTNDArray<P...> &arr,
                       const array<size_t, 4> &shape,
                       const array<size_t, 4> &strides)
    {
        img.assign(shape[3], shape[2], shape[1], shape[0]);
        if (is_c_contig(arr)) {
            copy_n(arr.data(), arr.size(), img.data());
        }
        else {
            for (size_t c = 0; c < shape[0]; c++) {
                const size_t offc = c * strides[0];
                for (size_t d = 0; d < shape[1]; d++) {
                    const size_t offd = offc + d * strides[1];
                    for (size_t y = 0; y < shape[2]; y++) {
                        const size_t offy = offd + y * strides[2];
                        for (size_t x = 0; x < shape[3]; x++) {
                            img(x, y, d, c) = offy + x * strides[3];
                        }
                    }
                }
            }
        }
        return img;
    }

    template <class t, class... P>
        requires(!same_as<t, T>)
    static Img &assign(Img &img, nb::ndarray<t, nb::device::cpu, P...> arr)
    {
        CImg<t> img2(arr);
        img.assign(img2);
        return img;
    }

    template <class Tp = T, class... P>
    static auto as_ndarray(Img &img)
    {
        return CTNDArray<Tp, nb::ndim<4>, P...>(
            img.data(), {img._spectrum, img._depth, img._height, img._width},
            nb::handle());
    }

    template <class... P>
    static auto to_ndarray(Img &img)
    {
        return copy_ndarray(as_ndarray<const T, P...>(img));
    }

    static auto dlpack_device(Img &)
    {
        return nb::make_tuple(nb::device::cpu::value, 0);
    }

    static nb::object array_interface(Img &img)
    {
        auto arr = as_ndarray<nb::numpy>(img);
        return cast(arr).attr(ARRAY_INTERFACE);
    }

    template <class I = size_t>
    static vector<I> strides(Img &img)
    {
        constexpr I S = sizeof(T);
        return {S * img.width() * img.height() * img.depth(),
                S * img.width() * img.height(), S * img.width(), S};
    }

    static nb::tuple strides_tuple(Img &img)
    {
        const auto s = strides(img);
        return nb::make_tuple(s[0], s[1], s[2], s[3]);
    }

    template <class I = size_t>
    static vector<I> shape(Img &img)
    {
        return {static_cast<I>(img.spectrum()), static_cast<I>(img.depth()),
                static_cast<I>(img.height()), static_cast<I>(img.width())};
    }

    static nb::tuple shape_tuple(Img &img)
    {
        return nb::make_tuple(img.spectrum(), img.depth(), img.height(),
                              img.width());
    }

    [[nodiscard]] static string str(Img &img)
    {
        stringstream out;
        out << "<" << nb::type_name(nb::type<Img>()).c_str() << " at "
            << static_cast<const void *>(&img)
            << ", data at: " << static_cast<const void *>(img.data())
            << ", w×h×d×s=" << img.width() << "×" << img.height() << "×"
            << img.depth() << "×" << img.spectrum() << ">";
        return out.str();
    }

    template <class... Args>
    using assign_t = Img &(*)(Img &, Args...);
    template <class... Args>
    using new_image_t = void (*)(Img *, Args...);

    static void bind(nb::module_ &m)
    {
        auto cls =
            nb::class_<Img>(m, CLASSNAME)
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
                     nb::rv_policy::take_ownership,
                     "Returns a copy of the underlying data as a"
                     " DLPack capsule")
                .def(
                    "as_numpy", &gmic_image_py::as_ndarray<nb::numpy>,
                    nb::rv_policy::reference_internal,
                    "Returns a view of the underlying data as a Numpy NDArray")
                .def(
                    "to_numpy", &gmic_image_py::to_ndarray<nb::numpy>,
                    nb::rv_policy::take_ownership,
                    "Returns a copy of the underlying data as a Numpy NDArray")
                .def_prop_ro("shape", &gmic_image_py::shape_tuple,
                             "Tuple of dimensions of this image")
                .def_prop_ro("strides", &gmic_image_py::strides_tuple,
                             "Tuple of strides of this image")
                .def_prop_ro("width", &Img::width,
                             "Width (1st dimension) of the image")
                .def_prop_ro("height", &Img::height,
                             "Height (2nd dimension) of the image")
                .def_prop_ro("depth", &Img::depth,
                             "Depth (3rd dimension) of the image")
                .def_prop_ro(
                    "spectrum", &Img::spectrum,
                    "Spectrum (i.e. channels, 4th dimension) of the image")
                .def("__str__", &gmic_image_py::str)
                .def("__repr__", &gmic_image_py::str);

        // Bindings for CImg constructors and assign()'s
#define ARGS(...) __VA_ARGS__
#define IMAGE_ASSIGN(funcname, doc, TYPES, ...)                               \
    cls.def("__init__",                                                       \
            static_cast<new_image_t<TYPES>>(&gmic_image_py::new_image),       \
            Trans::assign_signature<TYPES>(doc_buf, doc, "CImg<T>"),          \
            ##__VA_ARGS__)                                                    \
        .def(funcname, static_cast<assign_t<TYPES>>(&gmic_image_py::assign),  \
             Trans::assign_signature<TYPES>(doc_buf, doc, "CImg<T>::assign"), \
             nb::rv_policy::none, ##__VA_ARGS__)
        char doc_buf[1024];

        IMAGE_ASSIGN("assign_empty", "Construct an empty image", ARGS());

        IMAGE_ASSIGN("assign_copy", "Copy or proxy existing image",
                     ARGS(Img &, bool), "other"_a, "is_shared"_a = false);
        IMAGE_ASSIGN(
            "assign_dims",
            "Construct image with specified size and initialize pixel "
            "values",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int),
            "width"_a, "height"_a, "depth"_a = 0, "channels"_a = 0);
        IMAGE_ASSIGN(
            "assign_dims_valstr",
            "Construct image with specified size and initialize pixel "
            "values from a value string",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int,
                 string &, bool),
            "width"_a, "height"_a, "depth"_a, "channels"_a, "value_string"_a,
            "repeat"_a);
        IMAGE_ASSIGN("assign_load_file",
                     "Construct image from reading an image file",
                     ARGS(const char *), "filename"_a);
        IMAGE_ASSIGN(
            "assign_copy_dims",
            "Construct image with dimensions borrowed from another image",
            ARGS(Img &, string &), "other"_a, "dimensions"_a);

        // gmic-py specific bindings
        IMAGE_ASSIGN("assign_ndarray",
                     "Construct an image from an array-like object",
                     ARGS(CTNDArray<>), "array"_a);
    }
#undef IMAGE_ASSIGN
#undef ARGS
};

template <class T>
class gmic_list_base {
   protected:
    CImgList<T> list;
    template <class... Args>
    explicit gmic_list_base(Args... args) : list(args...)
    {
    }

    virtual ~gmic_list_base() = default;

   public:
    static constexpr auto CLASSNAME = "ImageList";
    using Item = CImg<T> &;

    Item get(unsigned int i)
    {
        if (i >= list.size())
            throw out_of_range("Out of range or gmic_list_py object");
        return list(i);
    }

    void set(unsigned int i, Item item) { list(i).assign(item); }
};

template <>
class gmic_list_base<char> {
   protected:
    CImgList<char> list;
    template <class... Args>
    explicit gmic_list_base(Args... args) : list(args...)
    {
    }

    virtual ~gmic_list_base() = default;

   public:
    using Item = string;
    static constexpr auto CLASSNAME = "StringList";

    //    [[nodiscard]] Item get(const unsigned int i) { return {list(i)}; }
    [[nodiscard]] Item get(const unsigned int i) const { return {list(i)}; }

    void set(const unsigned int i, Item &item)
    {
        list(i).assign(CImg<char>::string(item.c_str()));
    }
};

template <class T = gmic_pixel_type>
class gmic_list_py : public gmic_list_base<T> {
   private:
    using Item = gmic_list_base<T>::Item;
    using Base = gmic_list_base<T>;

   public:
    using Base::Base;

    ~gmic_list_py() override = default;

    CImgList<T> &list() { return Base::list; }

    auto operator[](unsigned int i) { return this->get(i); }
    auto operator[](unsigned int i) const { return this->get(i); }

    class iterator
        : std::iterator<std::forward_iterator_tag, remove_reference_t<Item>> {
        gmic_list_py &list;
        unsigned int iter = 0;

       public:
        explicit iterator(gmic_list_py &list, const unsigned int start = 0)
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

    size_t size() { return Base::list._width; }

    iterator begin() { return iterator(*this); }

    iterator end() { return iterator(*this, size()); }

    [[nodiscard]] auto iter()
    {
        return nb::make_iterator(nb::type<gmic_list_py>(), "iterator",
                                 this->begin(), this->end());
    }

    string str()
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<decltype(*this)>()).c_str()
            << '[';
        bool first = true;

        for (auto item : *this) {
            if (first)
                first = false;
            else
                out << ", ";
            if constexpr (is_same_v<CImg<T>, Item>) {
                out << item.str();
            }
            else {
                out << item;
            }
        }
        out << "]>";

        return out.str();
    }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_list_py>(m, gmic_list_base<T>::CLASSNAME)
            .def("__iter__", &gmic_list_py::iter)
            .def("__len__", &gmic_list_py::size)
            .def("__str__", &gmic_list_py::str)
            .def("__getitem__",
                 (Item(gmic_list_py::*)(unsigned int)) & gmic_list_py::get,
                 "i"_a, nb::rv_policy::reference_internal)
            .def("__setitem__", &gmic_list_py::set, "i"_a, "v"_a);
    }
};

using gmic_charlist_py = gmic_list_py<char>;

template <class T = gmic_pixel_type>
class interpreter_py {
    static gmic_list_py<T> *run(gmic &gmic, const char *cmd,
                                gmic_list_py<T> *img_list,
                                gmic_charlist_py *img_names)
    {
        if (img_list == nullptr)
            img_list = new gmic_list_py<T>();

        gmic_charlist_py _names, *names = &_names;

        if (img_names)
            names = img_names;

        try {
            gmic.run(cmd, img_list->list(), names->list());
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

    template <class R, class... Args>
    static auto make_static_run(R (*)(gmic &gmic, Args... args))
        -> function<R(Args...)>
    {
        static unique_ptr<gmic> inter{};
        return [&](Args... args) {
            if (!inter)
                inter = make_unique<gmic>();
            return run(*inter, args...);
        };
    }

    static string str(const gmic inst)
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<gmic>()).c_str() << " object at "
            << static_cast<const void *>(&inst) << '>';
        return out.str();
    }

   public:
    constexpr static auto CLASSNAME = "Gmic";

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic>(m, CLASSNAME)
            .def("run", &interpreter_py::run, "cmd"_a,
                 "img_list"_a = nb::none(), "img_names"_a = nb::none(),
                 nb::rv_policy::take_ownership)
            .def("__str__", &interpreter_py::str)
            .def(nb::init());

        m.def("run", make_static_run(&interpreter_py::run), "cmd"_a,
              "img_list"_a = nb::none(), "img_names"_a = nb::none(),
              nb::rv_policy::take_ownership);
    }
};

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
#define IS_DEFINED(macro)                              \
    (strcmp(#macro, Py_STRINGIFY(macro)) != 0 ? #macro \
         "=" Py_STRINGIFY(macro)                       \
                                              : #macro "=N/A")

        static char build[256];
        stringstream build_str;
        build_str << "Built on " __DATE__ << " at " << __TIME__;
        strncpy(build, build_str.str().c_str(), size(build));
        m.attr("__build__") = build;
        static const auto flags =
            to_array({IS_DEFINED(DEBUG),  // NOLINT(*-branch-clone)
                      IS_DEFINED(__cplusplus),
                      IS_DEFINED(cimg_display),
                      IS_DEFINED(cimg_use_pthread),
                      IS_DEFINED(cimg_use_board),
                      IS_DEFINED(cimg_use_curl),
                      IS_DEFINED(cimg_use_fftw3),
                      IS_DEFINED(cimg_use_half),
                      IS_DEFINED(cimg_use_heif),
                      IS_DEFINED(cimg_use_jpeg),
                      IS_DEFINED(cimg_use_lapack),
                      IS_DEFINED(cimg_use_magick),
                      IS_DEFINED(cimg_use_minc2),
                      IS_DEFINED(cimg_use_opencv),
                      IS_DEFINED(cimg_use_openexr),
                      IS_DEFINED(cimg_use_openmp),
                      IS_DEFINED(cimg_use_png),
                      IS_DEFINED(cimg_use_tiff),
                      IS_DEFINED(cimg_use_tinyexr),
                      IS_DEFINED(cimg_use_vt100),
                      IS_DEFINED(cimg_use_xrandr),
                      IS_DEFINED(cimg_use_xshm),
                      IS_DEFINED(cimg_use_zlib)});
        m.attr("__build_flags__") = flags;
    }

    static_assert(Trans::is_translatable<string>::value);

#if DEBUG == 1
    m.def("inspect", &inspect, "array"_a, "Inspects a N-dimensional array");
#endif

    gmic_image_py<>::bind(m);

    gmic_list_py<>::bind(m);

    gmic_charlist_py::bind(m);

    interpreter_py<>::bind(m);

    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
}

}  // namespace gmicpy
