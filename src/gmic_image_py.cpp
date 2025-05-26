#include <utility>

#include "gmicpy.hpp"
#include "nb_ndarray_buffer.hpp"
#include "utils.hpp"

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

constexpr auto ARRAY_INTERFACE = "__array_interface__";
constexpr auto DLPACK_INTERFACE = "__dlpack__";
constexpr auto DLPACK_DEVICE_INTERFACE = "__dlpack_device__";

#define ARGS(...) __VA_ARGS__
#if DEBUG
#define LOG_SIG(level, func, args, ...)                            \
    LOG << Level::level                                            \
        << assign_signature<args>(Py_STRINGIFY(func)) LOG_VA_ELSE( \
               __VA_OPT__(<< ": " << __VA_ARGS__, ) << std::endl);
#else
#define LOG_SIG(...) ;
#endif

class gmic_image_py {
   public:
    using T = gmic_pixel_type;
    using Img = CImg<>;
    /// ndarray of type T on the CPU
    template <class... P>
    using TNDArray = nb::ndarray<T, nb::device::cpu, P...>;
    /// read-only ndarray of type T on the CPU
    template <class... P>
    using CTNDArray = nb::ndarray<const T, nb::device::cpu, P...>;

    constexpr static auto CLASSNAME = "Image";

    // ReSharper disable CppTemplateParameterNeverUsed
    template <class I, class... Args>
    struct can_native_init : false_type {};
    // ReSharper restore CppTemplateParameterNeverUsed

    template <class... Args>
    struct can_native_init<
        enable_if_t<is_same_v<Img, decltype(Img(declval<Args>()...))>, Img>,
        Args...> : true_type {};

    template <class... Args>
    static void new_image(Img *img, Args... args)
    {
        if constexpr (can_native_init<Img, Args...>::value) {
            new (img) Img(args...);
            LOG_SIG(Debug, new_image, ARGS(Img &, Args...),
                    img_to_string(*img) << endl);
        }
        else {
            new (img) Img();
            assign(*img, args...);  // NOLINT(*-unnecessary-value-param)
            LOG_SIG(Debug, assign, ARGS(Img &, Args...),
                    img_to_string(*img) << endl);
        }
    }

    template <class... Args>
    static auto assign(Img &img, Args... args) -> enable_if_t<
        is_lvalue_reference_v<decltype(Img{}.assign(declval<Args>()...))>,
        Img &>
    {
        LOG_SIG(Debug, assign, ARGS(Img &, Args...),
                img_to_string(img) << endl);
        img.assign(args...);
        return img;
    }

    template <class... P>
    static Img &assign(Img &img, CTNDArray<P...> arr)
    {
        LOG_SIG(Debug, assign, ARGS(Img &, CTNDArray<P...>),
                img_to_string(img) << endl);
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
        static constexpr size_t DIM_X = 0, DIM_Y = 1, DIM_Z = 2, DIM_C = 3;
        img.assign(shape[DIM_X], shape[DIM_Y], shape[DIM_Z], shape[DIM_C]);
        LOG_SIG(Trace, assign,
                ARGS(Img &, CTNDArray<P...> &, const array<size_t, 4> &,
                     const array<size_t, 4> &),
                img_to_string(img)
                    << "\nCopying data from " << arr.data() << " with shape=("
                    << shape[0] << ", " << shape[1] << ", " << shape[2] << ", "
                    << shape[3] << ") and strides=(" << strides[0] << ", "
                    << strides[1] << ", " << strides[2] << ", " << strides[3]
                    << ")");

        if (is_f_contig(arr)) {
            LOG << ", F-contig (std::copy_n)" << endl;
            copy_n(arr.data(), arr.size(), img.data());
        }
        else {
            LOG << ", Non-F-contig (loop)" << endl;
            for (size_t c = 0; c < shape[DIM_C]; c++) {
                const size_t offc = c * strides[DIM_C];
                for (size_t d = 0; d < shape[DIM_Z]; d++) {
                    const size_t offd = offc + d * strides[DIM_Z];
                    for (size_t y = 0; y < shape[DIM_Y]; y++) {
                        const size_t offy = offd + y * strides[DIM_Y];
                        for (size_t x = 0; x < shape[DIM_X]; x++) {
                            img(x, y, d, c) =
                                arr.data()[offy + x * strides[DIM_X]];
                        }
                    }
                }
            }
        }
        return img;
    }

    static Img &assign(Img &img, const std::filesystem::path &path)
    {
        LOG_DEBUG(img_to_string(img) << endl);
        return img.load(path.c_str());
    }

    template <class Ti, class... P>
        requires(!same_as<Ti, T>)
    static Img &assign(Img &img, nb::ndarray<Ti, nb::device::cpu, P...> arr)
    {
        LOG_SIG(Trace, assign,
                ARGS(Img &, nb::ndarray<Ti, nb::device::cpu, P...>),
                img_to_string(img) << endl);
        CImg<Ti> img2(arr);
        img.assign(img2);
        return img;
    }

    template <class... P>
    static auto as_ndarray(const nb::handle &imgh)
    {
        auto &img = nb::cast<Img &>(imgh);
        LOG_SIG(Trace, as_ndarray, ARGS(T, P...), img_to_string(img) << endl);
        check_has_data(img);
        auto shape_v = shape<size_t>(img);
        auto strides_v = strides<int64_t, false>(img);
        return TNDArray<T, nb::ndim<4>, P...>(img.data(), 4, shape_v.data(),
                                              imgh, strides_v.data());
    }

    static auto dlpack_device(Img &)
    {
        return nb::make_tuple(nb::device::cpu::value, 0);
    }

    static auto dlpack(
        const nb::handle_t<Img> &img, optional<nb::handle> stream,
        optional<nb::tuple> /* max_version, accepted but ignored*/,
        optional<nb::tuple> dl_device, optional<bool> copy)
    {
        if (stream)
            throw nb::value_error("Unsupported __dlpack__ argument: stream");
        if (dl_device &&
            dl_device->not_equal(nb::make_tuple(nb::device::cpu::value, 0)))
            throw nb::value_error(
                "Unsupported __dlpack__ dl_device, only CPU is supported");

        auto array = as_ndarray<>(img);
        return !copy || !*copy ? array.cast(nb::rv_policy::reference)
                               : array.cast(nb::rv_policy::copy);
    }

    static nb::object array_interface(Img &img)
    {
        LOG_TRACE(img_to_string(img) << endl);
        check_has_data(img);
        nb::dict ai{};
        ai["typestr"] = get_typestr<T>().data();
        ai["data"] =
            nb::make_tuple(reinterpret_cast<uintptr_t>(img.data()), false);
        ai["shape"] = tuple_cat(shape<size_t>(img));
        ai["strides"] = tuple_cat(strides<size_t, true>(img));
        ai["version"] = 3;
        return ai;
    }

    /// Returns the strides of the image, in xyzc order
    template <integral I = size_t, bool bytes = false>
    static array<I, 4> strides(const Img &img)
    {
        constexpr I S = bytes ? static_cast<I>(sizeof(T)) : 1;
        return {S, S * img.width(), S * img.width() * img.height(),
                S * img.width() * img.height() * img.depth()};
    }

    /// Returns the shape of the image, in xyzc order
    template <integral I = size_t>
    static array<I, 4> shape(const Img &img)
    {
        return {static_cast<I>(img.width()), static_cast<I>(img.height()),
                static_cast<I>(img.depth()), static_cast<I>(img.spectrum())};
    }

    /** Casts a python object into a valid coordinate for the given dimension
     * size */
    static unsigned int cast_coord(const nb::object &obj,
                                   const unsigned int size, const char *dim)
    {
        try {
            return cast_long(nb::cast<long>(obj), size, dim);
        }
        catch (std::bad_cast &) {
            throw invalid_argument(
                string(dim) + " coordinate could not be converted to integer");
        }
    }

    /**
     * Casts a long to an unsigned int that is a valid in-bounds coordinate,
     * wrapping negative values around the axis
     * @param val Input value
     * @param size Size of the dimension on the given axis
     * @param dim Name of the dimension, for a more informative error message
     * @return the value, casted and checked for validity
     */
    static unsigned int cast_long(long val, const unsigned int size,
                                  const char *dim = nullptr)
    {
        if (val < 0)
            val = size + val;
        if (val < 0 || val >= size) {
            throw out_of_range(
                dim ? (string(dim) + " coordinate is out-of-bound")
                    : "Coordinate is out-of-bound");
        }
        return static_cast<unsigned int>(val);
    }

    static void check_has_data(const Img &img)
    {
        if (img.data() == nullptr)
            throw runtime_error("Image has no data");
    }

    static constexpr auto get_pydoc =
        "Returns the value at the given coordinate. Takes between 2 and 4 "
        "arguments depending on image dimensions :\n"
        "- [x, y, z, c]\n"
        "- [x, y, c] if depth = 1\n"
        "- [x, y] if depth = 1 and spectrum = 1\n"
        "Value must be between -size and size-1 on the corresponding axis. "
        "Negative values are relative to the end of the axis.\n"
        "Raises a ValueError if condition is not met";
    static T get(Img &img, const nb::tuple &args)
    {
        check_has_data(img);
        unsigned int x = cast_coord(args[0], img.width(), "X"),
                     y = cast_coord(args[1], img.height(), "Y"), z = 0, c = 0;
        switch (args.size()) {
            case 2:
                if (img.depth() != 1 || img.spectrum() != 1)
                    throw invalid_argument(
                        "Can't omit coordinates unless the corresponding axis "
                        "has a dimension of 1");
                break;
            case 3:
                if (img.depth() != 1)
                    throw invalid_argument(
                        "Can't omit coordinates unless the corresponding axis "
                        "has a dimension of 1");
                c = cast_coord(args[2], img.spectrum(), "channel");
                break;
            case 4:
                z = cast_coord(args[2], img.depth(), "Z");
                c = cast_coord(args[3], img.spectrum(), "C");
                break;
            default:
                throw invalid_argument(
                    "Invalid number of arguments (must be between 2 and 4)");
        }
#if DEBUG == 1
        LOG_TRACE("\nInterpreting " << nb::repr(args).c_str()
                                    << " as (xyzc) = [" << x << ", " << y
                                    << ", " << z << ", " << c << "]" << endl);
#endif
        return img(x, y, z, c);
    }

    static constexpr auto pixel_at_doc =
        "Returns a spectrum-sized (e.g 3 for RGB, 4 for RGBA) tuple, of the "
        "values at [x, y, z]. Z may be omitted if the image depth is 1.\n"
        "Negative values are relative to the end of the axis.\n";
    static nb::tuple pixel_at(Img &img, const long xi, const long yi,
                              const optional<long> zi)
    {
        check_has_data(img);
        unsigned int x = cast_long(xi, img.width(), "X"),
                     y = cast_long(yi, img.height(), "Y"), z = 0;
        if (zi)
            z = cast_long(*zi, img.depth(), "Z");
        else if (img.depth() != 1)
            throw invalid_argument("Can't omit Z if image depth is not 1");

#if DEBUG == 1
        LOG_TRACE("\nInterpreting ("
                  << xi << ", " << yi << ", " << (zi ? to_string(*zi) : "None")
                  << ") as (xyz) = (" << x << ", " << y << ", " << z << ")"
                  << endl);
#endif
        return to_tuple_func(img.spectrum(),
                             [&](unsigned int i) { return img(x, y, z, i); });
    }

    static int get_buffer(PyObject *exporter, Py_buffer *view,
                          const int flags) noexcept
    {
        LOG_DEBUG();
        const auto handle = nb::handle(exporter);
        try {
            const auto ndarr = as_ndarray<T>(handle);
            auto ret_val = ndarray_tpbuffer(ndarr, handle, view, flags);
            LOG << ", return code = " << ret_val << endl;
            return ret_val;
        }
        catch (nb::cast_error &) {
            PyErr_SetString(PyExc_BufferError,
                            "Object is not a valid G'MIC Image");
        }
        catch (exception &ex) {
            PyErr_SetString(PyExc_BufferError, ex.what());
        }
        view->obj = nullptr;
        return -1;
    }

    template <class... Args>
    using assign_t = Img &(*)(Img &, Args...);
    template <class... Args>
    using new_image_t = void (*)(Img *, Args...);

    static auto bind(const nb::module_ &m)
    {
        LOG_DEBUG("Binding gmic.Image class" << endl);

        PyType_Slot slots[] = {
#if defined(Py_bf_getbuffer) && defined(Py_bf_releasebuffer)
            {Py_bf_getbuffer,
             reinterpret_cast<void *>(static_cast<getbufferproc>(get_buffer))},
            {Py_bf_releasebuffer,
             reinterpret_cast<void *>(
                 static_cast<releasebufferproc>(nb_ndarray_releasebuffer))},
#endif
            {0, nullptr}};

        // ReSharper disable CppIdenticalOperandsInBinaryExpression
        auto cls =
            nb::class_<Img>(m, CLASSNAME, "G'MIC Image", nb::type_slots(slots))
                .def(DLPACK_INTERFACE, &gmic_image_py::dlpack, nb::kw_only(),
                     "stream"_a = nb::none(), "max_version"_a = nb::none(),
                     "dl_device"_a = nb::none(), "copy"_a = nb::none(),
                     nb::sig("def __dlpack__(self, *, "
                             "stream: int | Any | None = None, "
                             "max_version: tuple[int, int] | None = None, "
                             "dl_device: tuple[Enum, int] | None = None, "
                             "copy: bool | None = None) → PyCapsule"))
                .def(DLPACK_DEVICE_INTERFACE, &gmic_image_py::dlpack_device)
                .def_prop_ro(ARRAY_INTERFACE, &gmic_image_py::array_interface,
                             nb::rv_policy::reference_internal)
                .def("as_numpy", &gmic_image_py::as_ndarray<nb::numpy>,
                     nb::rv_policy::reference_internal,
                     "Returns a writable view of the underlying data as a "
                     "Numpy NDArray")
                .def(
                    "to_numpy", &gmic_image_py::as_ndarray<nb::numpy>,
                    nb::rv_policy::copy,
                    "Returns a copy of the underlying data as a Numpy NDArray")
                .def("at", &pixel_at, pixel_at_doc, "x"_a, "y"_a,
                     "z"_a = nb::none())
                .def_prop_ro(
                    "shape",
                    [](const Img &img) { return tuple_cat(shape<>(img)); },
                    "Returns the shape (size along each axis) tuple of the "
                    "image in xyzc order")
                .def_prop_ro(
                    "strides",
                    [](const Img &img) { return tuple_cat(strides<>(img)); },
                    "Returns the stride tuple (step size along each axis) "
                    "of the image in xyzc order")
                .def_prop_ro("width", &Img::width,
                             "Width (1st dimension) of the image")
                .def_prop_ro("height", &Img::height,
                             "Height (2nd dimension) of the image")
                .def_prop_ro("depth", &Img::depth,
                             "Depth (3rd dimension) of the image")
                .def_prop_ro(
                    "spectrum", &Img::spectrum,
                    "Spectrum (i.e. channels, 4th dimension) of the image")
                .def_prop_ro("size", &Img::size,
                             "Total number of values in the image (product of "
                             "all dimensions)")
                .def("__repr__", &img_to_string)
                .def("__getitem__", &get, get_pydoc)
                .def(+nb::self, "Returns a copy of the image")
                .def(-nb::self)
                .def(nb::self == nb::self)
                .def(nb::self + nb::self)
                .def(nb::self + int())
                .def(nb::self + float())
                .def(nb::self += nb::self, nb::rv_policy::none)
                .def(nb::self += int(), nb::rv_policy::none)
                .def(nb::self += float(), nb::rv_policy::none)
                .def(nb::self - nb::self)
                .def(nb::self - int())
                .def(nb::self - float())
                .def(nb::self -= nb::self, nb::rv_policy::none)
                .def(nb::self -= int(), nb::rv_policy::none)
                .def(nb::self -= float(), nb::rv_policy::none)
                .def(nb::self * int())
                .def(nb::self * float())
                .def(nb::self *= int(), nb::rv_policy::none)
                .def(nb::self *= float(), nb::rv_policy::none)
                .def(nb::self / int())
                .def(nb::self / float())
                .def(nb::self /= int(), nb::rv_policy::none)
                .def(nb::self /= float(), nb::rv_policy::none);

        cls.def(
            "fill",
            static_cast<Img &(Img::*)(const char *, bool, bool, CImgList<> *)>(
                &Img::fill),
            "Fills the image with the given value string. Like "
            "assign_dims_valstr with the image's current dimensions",
            "expression"_a, "repeat_values"_a = true, "allow_formula"_a = true,
            "list_images"_a.none() = nullptr, nb::rv_policy::none);
        // ReSharper restore CppIdenticalOperandsInBinaryExpression

        // Bindings for CImg constructors and assign()'s
#define IMAGE_ASSIGN(funcname, doc, TYPES, ...)                              \
    cls.def("__init__",                                                      \
            static_cast<new_image_t<TYPES>>(&gmic_image_py::new_image),      \
            assign_signature_doc<TYPES>(doc_buf, doc, "CImg<T>"),            \
            ##__VA_ARGS__)                                                   \
        .def(funcname, static_cast<assign_t<TYPES>>(&gmic_image_py::assign), \
             assign_signature_doc<TYPES>(doc_buf, doc, "CImg<T>::assign"),   \
             nb::rv_policy::none, ##__VA_ARGS__)
        char doc_buf[1024];

        IMAGE_ASSIGN("assign_empty",
                     "Construct an empty image. Beware that any attempt at "
                     "reading the image will raise a RuntimeError",
                     ARGS());

        IMAGE_ASSIGN("assign_copy", "Copy or proxy existing image",
                     ARGS(Img &, bool), "other"_a, "is_shared"_a = false);
        IMAGE_ASSIGN(
            "assign_dims",
            "Construct image with specified size and initialize pixel values",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int,
                 const T &),
            "width"_a, "height"_a, "depth"_a = 0, "channels"_a, "value"_a = 0);
        IMAGE_ASSIGN(
            "assign_dims_valstr",
            "Construct image with specified size and initialize pixel "
            "values from a value string",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int,
                 const char *, bool),
            "width"_a, "height"_a, "depth"_a, "channels"_a, "value_string"_a,
            "repeat"_a);
        IMAGE_ASSIGN("assign_load_file",
                     "Construct image from reading an image file",
                     ARGS(const filesystem::path &), "filename"_a);
        IMAGE_ASSIGN(
            "assign_copy_dims",
            "Construct image with dimensions borrowed from another image",
            ARGS(Img &, const char *), "other"_a, "dimensions"_a);

        // gmic-py specific bindings
        IMAGE_ASSIGN("assign_ndarray",
                     "Construct an image from an array-like object. Array "
                     "are taken as xyzc, if it has less than 4, then the "
                     "missing ones are assigned a size of 1.\n"
                     "Be aware that most image processing libraries use a "
                     "different order for dimensions (yxc), so this method "
                     "will not work as expected with such libraries. Use "
                     "Image.from_yxc(array) or img.yxc = array in that case.",
                     ARGS(CTNDArray<>), "array"_a);

        return cls;
    }
#undef IMAGE_ASSIGN
};

class yxc_wrapper {
   public:
    constexpr static auto CLASSNAME = "YXCWrapper";
    constexpr static auto CASTPOLICY_CLASSNAME = "CastPolicy";
    constexpr static cast_policy DEFAULT_CAST_POLICY = CLAMP;
    template <class... P>
    using NDArrayAnyD = nb::ndarray<nb::device::cpu, P...>;
    template <size_t ndim, class... P>
    using NDArray = nb::ndarray<nb::device::cpu, nb::ndim<ndim>, P...>;
    template <size_t ndim, class t, class... P>
    using CNDArray =
        nb::ndarray<const t, nb::device::cpu, nb::ndim<ndim>, P...>;

   private:
    using T = gmic_pixel_type;
    using DefaultOut = uint8_t;
    using ImgPy = gmic_image_py;
    using Img = ImgPy::Img;

    struct data_caster {
        function<NDArray<3, nb::ro>(const CNDArray<3, T> &, cast_policy)>
            cast_to;
        function<void(Img &, size_t, const NDArray<3, nb::ro> &, cast_policy)>
            cast_from;

        nb::dlpack::dtype dtype;
        string typestr;

        template <class t>
        static data_caster make_caster()
        {
            if constexpr (is_void_v<t>)  // For input-only wrapper
                return data_caster{{}, {}, {}, {}};
            else
                return data_caster{&yxc_wrapper::cast_data<T, t>,
                                   &yxc_wrapper::assign_data<t>,
                                   nb::dtype<t>(), get_typestr<t>().data()};
        }
    };

    nb::object img_obj;  // To keep the source image from being freed
    Img &img;
    optional<NDArray<3, nb::ro>> data = {};
    optional<nb::object> data_obj = {};
    optional<nb::object> bytes = {};
    const optional<size_t> z;
    const cast_policy cast_pol;
    const data_caster caster;

    static constexpr size_t DIM_NONE = 255;
    static constexpr array<size_t, 3> GMIC_TO_YXC = {1, 0, 3};
    static constexpr array<size_t, 4> YXC_TO_GMIC = {1, 0, DIM_NONE, 2};

    void check_has_data() const { ImgPy::check_has_data(img); }

    template <bool rtrn = true>
    [[nodiscard]] conditional_t<rtrn, size_t, void> effective_z()
        const  // NOLINT(*-use-nodiscard)
    {
        check_has_data();
        if (!z && img.depth() != 1) {
            throw runtime_error(
                "Must set Z before using wrapper unless image depth is 1");
        }
        if constexpr (rtrn)
            return z ? *z : 0;
        else
            return;
    }

    template <class From, class To>
    static NDArray<3, nb::ro> cast_data(const CNDArray<3, From> &ndarray,
                                        const cast_policy cast_pol)
    {
        return NDArray<3, nb::ro>(
            copy_ndarray<3, From, To>(ndarray, cast_pol));
    }

    template <class... P>
    CNDArray<3, T, P...> reshape_to_yxc()
    {
        const auto ez = effective_z();
        auto shape_v = shape_yxc<size_t>();
        auto strides_v = strides_yxc<int64_t>(img);
        return {&img(0, 0, ez, 0), 3, shape_v.data(),
                cast(img, nb::rv_policy::none), strides_v.data()};
    }

    auto &get_data()
    {
        if (!data) {
            const auto ndarray = reshape_to_yxc();
            data = caster.cast_to(ndarray, cast_pol);
            LOG_TRACE("Allocated YXC data buffer at "
                      << &data << " (data at " << data->data() << ')' << endl);
            data_obj = data->cast(nb::rv_policy::take_ownership);
        }
        else if (!data_obj->is_valid())
            throw runtime_error("Error allocating data array");
        return *data;
    }

    auto &get_bytes()
    {
        if (!bytes) {
            auto dat = get_data();
            bytes = cast(nb::bytes(dat.data(), dat.size() * dat.itemsize()),
                         nanobind::rv_policy::reference_internal, *data_obj);
        }
        return *bytes;
    }

    static const vector<data_caster> &get_casters()
    {
        thread_local vector<data_caster> casters;
        if (casters.empty()) {
            casters.push_back(data_caster::make_caster<float>());
            casters.push_back(data_caster::make_caster<double>());
            casters.push_back(data_caster::make_caster<uint8_t>());
            casters.push_back(data_caster::make_caster<uint16_t>());
            casters.push_back(data_caster::make_caster<uint32_t>());
            casters.push_back(data_caster::make_caster<uint64_t>());
            casters.push_back(data_caster::make_caster<int8_t>());
            casters.push_back(data_caster::make_caster<int16_t>());
            casters.push_back(data_caster::make_caster<int32_t>());
            casters.push_back(data_caster::make_caster<int64_t>());
            casters.push_back(data_caster::make_caster<bool>());
        }
        return casters;
    }

    static const vector<string> &get_casters_strs()
    {
        thread_local vector<string> typestrings;
        if (typestrings.empty()) {
            ranges::transform(get_casters(), back_inserter(typestrings),
                              &data_caster::typestr);
            if (typestrings.empty())
                throw runtime_error("");
        }
        return typestrings;
    }

   public:
    explicit yxc_wrapper(const nb::object &img_obj, const optional<size_t> z,
                         const data_caster &caster, const cast_policy cast_pol)
        : yxc_wrapper(nb::cast<Img &>(img_obj, false), z, caster, cast_pol)
    {
        this->img_obj = img_obj;
    }

    explicit yxc_wrapper(Img &img, const optional<size_t> z,
                         const data_caster &caster, const cast_policy cast_pol)
        : img(img), z(z), cast_pol(cast_pol), caster(caster)
    {
        LOG_TRACE("image: " << img_to_string(img) << endl);
    }

    ~yxc_wrapper()
    {
        LOG_TRACE("image: " << img_to_string(img) << ", data: ")
        if (data)
            LOG << " array " << &*data << " with data at " << data->data()
                << endl;
        else
            LOG << "nil" << endl;
    }

    /// Creates a temporary wrapper meant to perform a one-time conversion and
    /// which should never be returned to Python
    template <class To>
    static yxc_wrapper make_tmp_wrapper(
        Img &img, const optional<size_t> z = {},
        const cast_policy cast_pol = DEFAULT_CAST_POLICY)
    {
        return yxc_wrapper(img, z, data_caster::make_caster<To>(), cast_pol);
    }

    [[nodiscard]] static optional<data_caster> get_caster(
        const string &typestr)
    {
        auto &casters = get_casters();
        const auto caster =
            ranges::find_if(casters, [&](data_caster const &c) {
                return typestr == c.typestr || typestr == c.typestr.substr(1);
            });
        if (caster == casters.end())
            return {};
        return *caster;
    }

    [[nodiscard]] yxc_wrapper with(const nb::handle args) const
    {
        nb::tuple tup;
        try {
            tup = nb::cast<nb::tuple>(args, false);
        }
        catch (nb::cast_error &) {
            tup = nb::make_tuple(args);
        }
        try {
            optional<size_t> nz;
            optional<data_caster> cast;
            optional<cast_policy> pol;
            for (const auto &a : tup) {
                if (!nz)
                    try {
                        nz = static_cast<size_t>(nb::cast<nb::int_>(a, false));
                        check_has_data();
                        if (z)
                            throw runtime_error("Depth is already set");
                        if (nz >= img.depth())
                            throw out_of_range(
                                "Z out of range for image depth");
                        continue;
                    }
                    catch (nb::cast_error &) {
                    }
                if (!cast)
                    try {
                        auto typestr = nb::cast<string>(a, false);
                        cast = get_caster(typestr);
                        if (!cast)
                            throw nb::value_error(
                                ("Unknown dtype conversion requested: " +
                                 typestr)
                                    .c_str());
                        continue;
                    }
                    catch (nb::cast_error &) {
                    }
                if (!pol)
                    pol = nb::cast<cast_policy>(a, false);
            }

            return yxc_wrapper(img_obj, nz ? nz : z, cast.value_or(caster),
                               pol.value_or(cast_pol));
        }
        catch (nb::cast_error &e) {
            LOG_DEBUG("Cast error: " << e.what() << endl);
            throw nb::type_error(
                "Unknown type or multiple argument of the same type");
        }
    }

    template <class... P>
    auto to_ndarray()
    {
        auto dat = get_data();
        if constexpr (is_same_v<NDArray<3, nb::ro>,
                                NDArray<3, nb::ro, P...>>) {
            return dat;
        }
        else {
            const auto parent = cast(dat);
            return cast(NDArray<3, nb::ro>(dat),
                        nanobind::rv_policy::reference_internal, parent);
        }
    }

    static auto dlpack_device(Img &)
    {
        return nb::make_tuple(nb::device::cpu::value, 0);
    }

    nb::object array_interface()
    {
        LOG_TRACE();
        auto dat = get_data();

        nb::dict ai{};
        ai["typestr"] = caster.typestr;
        ai["data"] = get_bytes();
        ai["shape"] = make_tuple(dat.shape(0), dat.shape(1), dat.shape(2));
        ai["strides"] = make_tuple(dat.stride(0) * dat.itemsize(),
                                   dat.stride(1) * dat.itemsize(),
                                   dat.stride(2) * dat.itemsize());
        ai["version"] = 3;
        return ai;
    }

    /// Maps shape or strides from gmic to yxc order
    template <integral I = size_t>
    static auto dims_to_xyc(auto idims)
        -> conditional_t<is_same_v<decltype(idims), tuple<I, I, I, I>>,
                         tuple<I, I, I>, array<I, 3>>
    {
        return {get<GMIC_TO_YXC[0]>(idims), get<GMIC_TO_YXC[1]>(idims),
                get<GMIC_TO_YXC[2]>(idims)};
    }

    /// Returns the strides of the image, in yxc order
    template <integral I = size_t, bool bytes = false>
    static std::array<I, 3> strides_yxc(const Img &img)
    {
        auto istrides = ImgPy::strides<I, bytes>(img);
        return dims_to_xyc<I>(istrides);
    }

    /// Returns the shape of the image, in yxc order
    template <integral I = size_t>
    [[nodiscard]] std::array<I, 3> shape_yxc() const
    {
        auto ishape = ImgPy::shape<I>(img);
        return dims_to_xyc<I>(ishape);
    }

    template <class... P>
    static NDArray<3, P...> to_3d(const NDArrayAnyD<P...> &arr)
    {
        if (arr.ndim() == 3) {
            return NDArray<3, P...>(arr);
        }
        if (arr.ndim() == 2) {
            return NDArray<3, P...>(
                arr.data(), {arr.shape(0), arr.shape(1), 1}, {},
                {arr.stride(0), arr.stride(1), 1}, arr.dtype(),
                arr.device_type(), arr.device_id());
        }
        throw nb::next_overload("Array should be 2- or 3-dimensional");
    }

    static Img *new_image(const nb::object &obj)
    {
        const NDArrayAnyD<nb::ro> ndarr = cast_to_ndarray(obj);
        const auto img = new Img();
        const auto wrp = make_tmp_wrapper<void>(*img);
        wrp.assign_ndarray(ndarr, false);
        LOG_DEBUG("Created image " << img_to_string(*img) << endl);
        return img;
    }

    void assign_ndarray(const NDArrayAnyD<nb::ro> &iarr,
                        const bool samedims) const
    {
        LOG_DEBUG("z = " << (z ? static_cast<int>(*z) : -1) << ", cast_pol = "
                         << cast_pol << ", dtype: " << caster.typestr
                         << ", samedims: " << samedims << endl);
        const auto arr = to_3d<>(iarr);
        const auto same = img.height() == arr.shape(0) &&
                          img.width() == arr.shape(1) &&
                          img.spectrum() == arr.shape(2);
        size_t ez;
        if (samedims) {
            if (!same)
                throw nb::value_error(
                    "Can't assign an array with different dimensions, "
                    "use .assign(array, same_dims=False)");
            ez = effective_z();
        }
        else {
            if (z) {
                throw nb::value_error(
                    "Can't assign new dims to array with Z set");
            }
            if (!same || img.depth() != 1) {
                img.assign(arr.shape(YXC_TO_GMIC[0]),
                           arr.shape(YXC_TO_GMIC[1]), 1,
                           arr.shape(YXC_TO_GMIC[3]));
            }
            ez = 0;
        }
        for (const auto &caster : get_casters()) {
            if (arr.dtype() == caster.dtype) {
                caster.cast_from(img, ez, arr, cast_pol);
                return;
            }
        }
        throw nb::value_error("Invalid array type");
    }

    template <class Ti>
    static void assign_data(Img &img, const size_t z,
                            const NDArray<3, nb::ro> &iarr,
                            cast_policy cast_pol)
    {
        if (iarr.dtype() != nb::dtype<Ti>())
            throw runtime_error("Invalid array dtype passed to assign");
        const CNDArray<3, Ti> arr(iarr);
        const Ti *src = arr.data();
        const auto istrides = arr.stride_ptr();
        const auto ishape = arr.shape_ptr();
        const auto ostrides = strides_yxc<int64_t>(img);

        copy_ndarray_data<3, Ti, T>(src, istrides, ishape, &img(0, 0, z, 0),
                                    ostrides.data(), cast_pol);
    }

    void assign(const nb::handle &obj, const bool samedims) const
    {
        return assign_ndarray(cast_to_ndarray(obj), samedims);
    }

    static NDArrayAnyD<nb::ro> cast_to_ndarray(const nb::handle &obj)
    {
        try {
            return nb::cast<NDArrayAnyD<nb::ro>>(obj, false);
        }
        catch (nb::cast_error &) {
        }
        const auto typ = obj.type();
        nb::tuple mro = typ.attr("__mro__");
        const auto imgcls = ranges::find_if(mro, [](const nb::handle &c) {
            return nb::cast<string>(c.attr("__module__")) == "PIL.Image" &&
                   nb::cast<string>(c.attr("__qualname__")) == "Image";
        });
        if (imgcls != mro.end())
            try {
                const nb::dict ai = obj.attr(ARRAY_INTERFACE);
                if (nb::cast<int>(ai["version"]) != 3)
                    throw invalid_argument(
                        "Unsupported array_interface version");

                if (ai.contains("strides") || ai.contains("descr") ||
                    ai.contains("mask") || ai.contains("offset"))
                    throw invalid_argument(
                        "Unsupported array interface attributes");

                auto typestr = nb::cast<string>(ai["typestr"]);
                auto casters = get_casters();
                const auto caster_it = ranges::find_if(
                    casters,
                    [&](const auto &cst) { return cst.typestr == typestr; });
                if (caster_it == casters.end())
                    throw invalid_argument(string("Unsupported datatype: ") +
                                           typestr);
                const auto &caster = *caster_it;

                const nb::tuple shape_tup = ai["shape"];
                if (shape_tup.size() < 2 || shape_tup.size() > 3)
                    throw invalid_argument(
                        "Invalid array size: should be 2 or 3");
                vector<size_t> shape;
                size_t size = (caster.dtype.bits + 7) / 8;
                for (auto i : shape_tup) {
                    const int d = nb::cast<int>(i);
                    shape.push_back(d);
                    size *= d;
                }

                const nb::bytes data = ai["data"];
                if (data.size() != size)
                    throw invalid_argument(
                        string("Bytes object length doesn't match shape: ") +
                        to_string(data.size()) + " != " + to_string(size));

                return {data.data(), shape.size(), shape.data(),
                        ai,          nullptr,      caster.dtype};
            }
            catch (invalid_argument &) {
                throw;
            }
            catch (exception &ex) {
                LOG_INFO("Error accessing PIL image data: " << ex.what()
                                                            << endl);
                throw invalid_argument(
                    "Couldn't get image data from argument");
            }
        throw nb::type_error(
            "Unsupported object type. Object should be readable through "
            "the buffer protocol, DLPack or the NumPy array interface");
    }

    [[nodiscard]] string str() const
    {
        stringstream out;

        out << "<" << nb::type_name(nb::type<yxc_wrapper>()).c_str() << " at "
            << this << ", image at " << &img << ", z=";
        if (z)
            out << *z;
        else
            out << "None";
        out << ", cast_policy=";
        switch (cast_pol) {
            case CLAMP:
                out << "CLAMP";
                break;
            case THROW:
                out << "THROW";
                break;
            case NOCHECK:
                out << "NOCHECK";
        }
        out << ">";

        return out.str();
    }

    static int get_buffer(PyObject *exporter, Py_buffer *view,
                          const int flags) noexcept
    {
        LOG_DEBUG();
        const auto handle = nb::handle(exporter);
        try {
            auto &wrp = nb::cast<yxc_wrapper &>(handle);
            const auto &ndarr = wrp.get_data();
            auto ret_val = ndarray_tpbuffer(ndarr, handle, view, flags);
            LOG << ", return code = " << ret_val << endl;
            return ret_val;
        }
        catch (nb::cast_error &) {
            PyErr_SetString(PyExc_BufferError,
                            "Object is not a valid G'MIC Image");
        }
        catch (exception &ex) {
            PyErr_SetString(PyExc_BufferError, ex.what());
        }
        view->obj = nullptr;
        return -1;
    }

    static void bind(nb::class_<Img> &imgcls)
    {
        LOG_DEBUG("Binding gmic.Image.YXCWrapper class" << endl);
        char doc[1024];

        const auto castpolcls =
            nb::enum_<cast_policy>(
                imgcls, CASTPOLICY_CLASSNAME,
                "Datatype casting policy for OOB (out-of-bounds) values")
                .value("CLAMP", CLAMP,
                       "OOB values will be clamped to nearest bound (default)")
                .value("THROW", THROW,
                       "Exception will be raised if any OOB value is found")
                .value("NOCHECK", NOCHECK,
                       "Disable checking for OOB values. Can increase "
                       "performances at the risk of running into undefined "
                       "behaviour on OOB values (see C++ rules for "
                       "Floating-integral conversion).")
                .export_values();

        PyType_Slot slots[] = {
#if defined(Py_bf_getbuffer) && defined(Py_bf_releasebuffer)
            {Py_bf_getbuffer,
             reinterpret_cast<void *>(static_cast<getbufferproc>(get_buffer))},
            {Py_bf_releasebuffer,
             reinterpret_cast<void *>(
                 static_cast<releasebufferproc>(nb_ndarray_releasebuffer))},
#endif
            {0, nullptr}};

        auto cls =
            nb::class_<yxc_wrapper>(
                imgcls, CLASSNAME,
                ssnprintf(doc,
                          "Wrapper around a gmic.%s to exchange with "
                          "libraries using YXC axe order",
                          ImgPy::CLASSNAME),
                nb::type_slots(slots))
                .def_prop_ro_static(
                    "dtypes", [](nb::handle) { return get_casters_strs(); })
                .def_prop_ro("image",
                             [](const yxc_wrapper &wrap) {
                                 return nb::handle_t<Img>(wrap.img_obj);
                             })
                .def_ro("z", &yxc_wrapper::z)
                .def_prop_ro(
                    "dtype",
                    [](const yxc_wrapper &wrp) { return wrp.caster.typestr; })
                .def_ro("cast_policy", &yxc_wrapper::cast_pol)
                .def("__getitem__", &yxc_wrapper::with,
                     "Sets the wrapper's z, target datatype and/or casting "
                     "policy",
                     "args"_a.sig(ssnprintf(doc, "Tuple[int | str | %s, ...]",
                                            type_name(castpolcls).c_str())))
                .def("__setitem__",
                     [](const yxc_wrapper &wrp, const nb::handle args,
                        const nb::handle &obj) {
                         wrp.with(args).assign(obj, true);
                     })
                .def(DLPACK_INTERFACE, &yxc_wrapper::to_ndarray<>)
                .def(DLPACK_DEVICE_INTERFACE, &yxc_wrapper::dlpack_device)
                .def_prop_ro(ARRAY_INTERFACE, &yxc_wrapper::array_interface,
                             nb::rv_policy::reference)
                .def("__repr__", &yxc_wrapper::str)
                .def("to_numpy", &yxc_wrapper::to_ndarray<nb::numpy>,
                     "Returns a copy of the underlying data as a Numpy "
                     "NDArray")
                .def(
                    "tobytes", &yxc_wrapper::get_bytes,
                    "Returns the image data converted to the wrapper dtype as "
                    "a bytes object")
                .def_prop_ro(
                    "shape",
                    [](const yxc_wrapper &wrp) {
                        return tuple_cat(wrp.shape_yxc());
                    },
                    "Returns the shape (size along each axis) tuple of the "
                    "image in xyzc order");
        cls.def("assign", &yxc_wrapper::assign,
                "Assigns the given object's data to the image. Object "
                "must be readable through either the buffer protocol, "
                "DLPack or the NumPy Array Interface",
                "image"_a, "same_dims"_a = true);

        imgcls
            .def_prop_rw(
                "yxc",
                [](const nb::handle_t<Img> img) {
                    return new yxc_wrapper(
                        nb::steal(img), {},
                        data_caster::make_caster<DefaultOut>(),
                        DEFAULT_CAST_POLICY);
                },
                [](Img &img, const nb::object &obj) {
                    make_tmp_wrapper<void>(img).assign(obj, true);
                },
                doc)
            .def_static("from_yxc", &yxc_wrapper::new_image,
                        nb::rv_policy::take_ownership,
                        "Constructs an image from the given object. Object "
                        "must be readable through either the buffer protocol, "
                        "DLPack or the NumPy Array Interface",
                        "source"_a);
        LOG_DEBUG("Attaching yxc methods to class " << nb::repr(imgcls).c_str()
                                                    << endl);
    }
};

void bind_gmic_image(const nanobind::module_ &m)
{
    auto imgcls = gmic_image_py::bind(m);

    yxc_wrapper::bind(imgcls);
}

}  // namespace gmicpy