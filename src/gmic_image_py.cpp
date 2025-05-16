#include <utility>

#include "gmicpy.hpp"
#include "utils.hpp"

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

constexpr auto ARRAY_INTERFACE = "__array_interface__";
constexpr auto DLPACK_INTERFACE = "__dlpack__";
constexpr auto DLPACK_DEVICE_INTERFACE = "__dlpack_device__";

class gmic_image_py {
   public:
    using T = gmic_pixel_type;
    using Img = CImg<T>;
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
            LOG_DEBUG(assign_signature<Args...>("new_image") << endl);
            new (img) Img(args...);
        }
        else {
            new (img) Img();
            assign(*img, args...);  // NOLINT(*-unnecessary-value-param)
        }
    }

    template <class... Args>
    static auto assign(Img &img, Args... args) -> enable_if_t<
        is_lvalue_reference_v<decltype(Img{}.assign(declval<Args>()...))>,
        Img &>
    {
        LOG_DEBUG(assign_signature<Args...>("assign") << endl);
        img.assign(args...);
        return img;
    }

    template <class... P>
    static Img &assign(Img &img, CTNDArray<P...> arr)
    {
        LOG_DEBUG();
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
        LOG_DEBUG("\nCopying data from "
                  << arr.data() << " with shape=(" << shape[0] << ", "
                  << shape[1] << ", " << shape[2] << ", " << shape[3]
                  << ") and strides=(" << strides[0] << ", " << strides[1]
                  << ", " << strides[2] << ", " << strides[3] << ")");

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
        return img.load(path.c_str());
    }

    template <class Ti, class... P>
        requires(!same_as<Ti, T>)
    static Img &assign(Img &img, nb::ndarray<Ti, nb::device::cpu, P...> arr)
    {
        CImg<Ti> img2(arr);
        img.assign(img2);
        return img;
    }

    template <class Tp = T, class... P>
    static auto as_ndarray(Img &img)
    {
        auto shape_v = shape<size_t, array<size_t, 4>>(img);
        auto strides_v = strides<int64_t, false, array<int64_t, 4>>(img);
        return TNDArray<Tp, nb::ndim<4>, P...>(img.data(), 4, shape_v.data(),
                                               nb::handle(), strides_v.data());
    }

    static auto dlpack_device(Img &)
    {
        return nb::make_tuple(nb::device::cpu::value, 0);
    }

    static auto dlpack(Img &img)
    {
        LOG_TRACE();
        return as_ndarray<>(img);
    }

    static nb::object array_interface(Img &img)
    {
        LOG_TRACE();
        nb::dict ai{};
        ai["typestr"] = get_typestr<T>().data();
        ai["data"] =
            nb::make_tuple(reinterpret_cast<uintptr_t>(img.data()), false);
        ai["shape"] = shape<size_t>(img);
        ai["strides"] = strides<size_t, true>(img);
        ai["version"] = 3;
        return ai;
    }

    /// Returns the strides of the image, in xyzc order
    template <integral I = size_t, bool bytes = false,
              class container = tuple<I, I, I, I>>
    static container strides(const Img &img)
    {
        constexpr I S = bytes ? static_cast<I>(sizeof(T)) : 1;
        return {S, S * img.width(), S * img.width() * img.height(),
                S * img.width() * img.height() * img.depth()};
    }

    /// Returns the shape of the image, in xyzc order
    template <integral I = size_t, class container = tuple<I, I, I, I>>
    static container shape(const Img &img)
    {
        return {static_cast<I>(img.width()), static_cast<I>(img.height()),
                static_cast<I>(img.depth()), static_cast<I>(img.spectrum())};
    }

    /** Casts a python object into a valid coordinate for the given dimension
     * size */
    static unsigned int cast_coord(const nb::handle &obj,
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

    [[nodiscard]] static string str(const Img &img)
    {
        stringstream out;
        out << img;
        return out.str();
    }

    template <class... Args>
    using assign_t = Img &(*)(Img &, Args...);
    template <class... Args>
    using new_image_t = void (*)(Img *, Args...);

    static auto bind(const nb::module_ &m)
    {
        LOG_DEBUG("Binding gmic.Image class" << endl);
        // ReSharper disable CppIdenticalOperandsInBinaryExpression
        auto cls =
            nb::class_<Img>(m, CLASSNAME, "GMIC Image")
                .def(DLPACK_INTERFACE, &gmic_image_py::dlpack,
                     nb::rv_policy::reference_internal)
                .def(DLPACK_DEVICE_INTERFACE, &gmic_image_py::dlpack_device)
                .def_prop_ro(ARRAY_INTERFACE, &gmic_image_py::array_interface,
                             nb::rv_policy::reference_internal)
                .def("as_dlpack", &gmic_image_py::as_ndarray<>,
                     nb::rv_policy::reference_internal,
                     "Returns a view of the underlying data as a"
                     " DLPack capsule")
                .def("to_dlpack", &gmic_image_py::as_ndarray<>,
                     nb::rv_policy::copy,
                     "Returns a copy of the underlying data as a"
                     " DLPack capsule")
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
                    "shape", &shape<>,
                    "Returns the shape (size along each axis) tuple of the "
                    "image in xyzc order")
                .def_prop_ro(
                    "strides", &strides<>,
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
                .def("__str__", &gmic_image_py::str)
                .def("__repr__", &gmic_image_py::str)
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

        cls.def("fill",
                static_cast<Img &(Img::*)(const char *, bool, bool,
                                          CImgList<T> *)>(&Img::fill),
                "Fills the image with the given value string. Like "
                "assign_dims_valstr with the image's current dimensions",
                "expression"_a, "repeat_values"_a = true,
                "allow_formula"_a = true, "list_images"_a.none() = nullptr,
                nb::rv_policy::none);
        // ReSharper restore CppIdenticalOperandsInBinaryExpression

        // Bindings for CImg constructors and assign()'s
#define ARGS(...) __VA_ARGS__
#define IMAGE_ASSIGN(funcname, doc, TYPES, ...)                              \
    cls.def("__init__",                                                      \
            static_cast<new_image_t<TYPES>>(&gmic_image_py::new_image),      \
            assign_signature_doc<TYPES>(doc_buf, doc, "CImg<T>"),            \
            ##__VA_ARGS__)                                                   \
        .def(funcname, static_cast<assign_t<TYPES>>(&gmic_image_py::assign), \
             assign_signature_doc<TYPES>(doc_buf, doc, "CImg<T>::assign"),   \
             nb::rv_policy::none, ##__VA_ARGS__)
        char doc_buf[1024];

        IMAGE_ASSIGN("assign_empty", "Construct an empty image", ARGS());

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
                     ARGS(const char *), "filename"_a);
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
                     "will not work as expected with such libraries.",
                     ARGS(CTNDArray<>), "array"_a);

        return cls;
    }
#undef IMAGE_ASSIGN
#undef ARGS
};

class yxc_wrapper {
   public:
    constexpr static auto CLASSNAME = "YXCWrapper";
    template <class... P>
    using NDArrayAnyD = nb::ndarray<nb::device::cpu, P...>;
    template <size_t ndim, class... P>
    using NDArray = nb::ndarray<nb::device::cpu, nb::ndim<ndim>, P...>;
    template <size_t ndim, class t, class... P>
    using CNDArray =
        nb::ndarray<const t, nb::device::cpu, nb::ndim<ndim>, P...>;

   private:
    using T = gmic_pixel_type;
    using TO = uint8_t;  // Default output type
    using ImgPy = gmic_image_py;
    using Img = ImgPy::Img;

    struct data_caster {
        function<NDArray<3, nb::ro>(const CNDArray<3, T> &, cast_policy)>
            cast_to;
        function<void(Img &, size_t, const NDArray<3, nb::ro> &, cast_policy)>
            cast_from;

        nb::dlpack::dtype dtype;
        string typestr;
        static constexpr const char *void_error =
            "Tried to invoke void type caster";

        template <class t>
        static data_caster make_caster()
        {
            if constexpr (is_void_v<t>)  // For input-only wrapper
                return data_caster{{}, {}, {}, {}};
            else
                return data_caster{&yxc_wrapper::cast_data<T, t>,
                                   &yxc_wrapper::assign<t>, nb::dtype<t>(),
                                   get_typestr<t>().data()};
        }
    };

    Img &img;
    optional<size_t> z;
    optional<NDArray<3, nb::ro>> data = {};
    optional<nb::object> data_handle = {};
    optional<nb::object> bytes = {};
    cast_policy cast_pol;
    data_caster caster;

    static constexpr size_t DIM_NONE = 255;
    static constexpr array<size_t, 3> GMIC_TO_YXC = {1, 0, 3};
    static constexpr array<size_t, 4> YXC_TO_GMIC = {1, 0, DIM_NONE, 2};

    template <bool rtrn = true>
    conditional_t<rtrn, size_t, void> effective_z()
        const  // NOLINT(*-use-nodiscard)
    {
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
        auto shape_v = shape_yxc<size_t, false>();
        auto strides_v = strides_yxc<int64_t, false>(img);
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
            data_handle = data->cast(nb::rv_policy::take_ownership);
        }
        else if (!data_handle->is_valid())
            throw runtime_error("");
        return *data;
    }

    auto &get_bytes()
    {
        if (!bytes) {
            auto dat = get_data();
            bytes = cast(nb::bytes(dat.data(), dat.size() * dat.itemsize()),
                         nanobind::rv_policy::reference_internal, dat.cast());
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

   public:
    explicit yxc_wrapper(Img &img, const optional<size_t> z,
                         data_caster caster, const cast_policy cast_pol)
        : img(img), z(z), cast_pol(cast_pol), caster(std::move(caster))
    {
    }

    template <class To>
    static yxc_wrapper make_wrapper(Img &img, const optional<size_t> z = {},
                                    const cast_policy cast_pol = CLAMP)
    {
        return yxc_wrapper(img, z, data_caster::make_caster<To>(), cast_pol);
    }

    [[nodiscard]] yxc_wrapper with_z(size_t z) const
    {
        if (this->z)
            throw runtime_error("Depth is already set");
        if (z >= img.depth())
            throw out_of_range("Z out of range for image depth");
        return yxc_wrapper(img, z, caster, cast_pol);
    }

    template <class To>
    [[nodiscard]] yxc_wrapper with_dtype() const
    {
        return make_wrapper<To>(img, z, cast_pol);
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
    template <integral I = size_t, bool tuple = true, bool bytes = false>
    static auto strides_yxc(const Img &img)
        -> conditional_t<tuple, std::tuple<I, I, I>, std::array<I, 3>>
    {
        auto istrides = ImgPy::strides<
            I, bytes,
            conditional_t<tuple, std::tuple<I, I, I, I>, std::array<I, 4>>>(
            img);
        return dims_to_xyc<I>(istrides);
    }

    /// Returns the shape of the image, in yxc order
    template <integral I = size_t, bool tuple = true>
    auto shape_yxc()
        -> conditional_t<tuple, std::tuple<I, I, I>, std::array<I, 3>>
    {
        effective_z<false>();
        auto ishape = ImgPy::shape<
            I, conditional_t<tuple, std::tuple<I, I, I, I>, std::array<I, 4>>>(
            img);
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

    static Img new_image(const NDArrayAnyD<nb::ro> &array)
    {
        Img img;
        return make_wrapper<void>(img).assign_try(array, false);
    }

    static Img new_image_pil(const nb::object &obj)
    {
        LOG_DEBUG("Invoking assign_pil");
        const auto arr = read_array_interface(obj);
        if (!arr)
            throw nb::next_overload();
        return new_image(*arr);
    }

    Img &assign_try(const NDArrayAnyD<nb::ro> &iarr, const bool samedims) const
    {
        const auto arr = to_3d<>(iarr);
        const auto same = img.width() == arr.shape(GMIC_TO_YXC[0]) &&
                          img.height() == arr.shape(GMIC_TO_YXC[1]) &&
                          img.spectrum() == arr.shape(GMIC_TO_YXC[2]);
        size_t ez;
        if (samedims) {
            if (!same)
                throw invalid_argument(
                    "Can't assign an array with different dimensions, "
                    "use .assign(array, same_dims=False)");
            ez = effective_z();
        }
        else {
            if (z) {
                throw invalid_argument(
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
                return img;
            }
        }
        throw invalid_argument("Invalid array type");
    }

    template <class Ti>
    static void assign(Img &img, const size_t z,
                       const NDArray<3, nb::ro> &iarr, cast_policy cast_pol)
    {
        if (iarr.dtype() != nb::dtype<Ti>())
            throw runtime_error("Invalid array dtype passed to assign");
        const CNDArray<3, Ti> arr(iarr);
        const Ti *src = arr.data();
        const auto istrides = arr.stride_ptr();
        const auto ishape = arr.shape_ptr();
        const auto ostrides = strides_yxc<int64_t, false>(img);

        copy_ndarray_data<3, Ti, T>(src, istrides, ishape, &img(0, 0, z, 0),
                                    ostrides.data(), cast_pol);
    }

    [[nodiscard]] Img &assign_pil(const nb::object &obj,
                                  const bool samedims) const
    {
        LOG_DEBUG("Invoking assign_pil");
        const auto arr = read_array_interface(obj);
        if (!arr)
            throw nb::next_overload();

        return assign_try(*arr, samedims);
    }

    static optional<NDArrayAnyD<nb::ro>> read_array_interface(
        const nb::object &obj)
    {
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
                auto caster = ranges::find_if(casters, [&](const auto &cst) {
                    return cst.typestr == typestr;
                });
                if (caster == casters.end())
                    throw invalid_argument(string("Unsupported datatype: ") +
                                           typestr);

                const nb::tuple shape_tup = ai["shape"];
                if (shape_tup.size() < 2 || shape_tup.size() > 3)
                    throw invalid_argument(
                        "Invalid array size: should be 2 or 3");
                vector<size_t> shape;
                size_t size = 1;
                for (auto i : shape_tup) {
                    const int d = nb::cast<int>(i);
                    shape.push_back(d);
                    size *= d;
                }

                const nb::bytes data = ai["data"];
                if (data.size() != size)
                    throw invalid_argument(
                        "Bytes object length doesn't match shape");

                return NDArrayAnyD<nb::ro>(data.data(), shape.size(),
                                           shape.data(), nb::handle(ai),
                                           nullptr, caster->dtype);
            }
            catch (invalid_argument) {
                throw;
            }
            catch (exception &ex) {
                LOG_INFO("Error accessing PIL image data: " << ex.what()
                                                            << endl);
                throw invalid_argument(
                    "Couldn't get image data from argument");
            }
        return {};
    }

    static void bind(nb::class_<Img> &imgcls)
    {
        LOG_DEBUG("Binding gmic.Image.YXCWrapper class" << endl);
        char doc[1024];
        snprintf(doc, size(doc),
                 "Wrapper around a gmic.%s to exchange with "
                 "libraries using YXC axe order",
                 ImgPy::CLASSNAME);
        auto cls =
            nb::class_<yxc_wrapper>(imgcls, CLASSNAME, doc)
                .def_prop_ro(
                    "image", [](const yxc_wrapper &wrap) { return wrap.img; },
                    nb::rv_policy::reference_internal)
                .def_ro("z", &yxc_wrapper::z)
                .def("__getitem__", &yxc_wrapper::with_z, "z"_a)
                .def(DLPACK_INTERFACE, &yxc_wrapper::to_ndarray<>)
                .def(DLPACK_DEVICE_INTERFACE, &yxc_wrapper::dlpack_device)
                .def_prop_ro(ARRAY_INTERFACE, &yxc_wrapper::array_interface,
                             nb::rv_policy::reference)
                .def("to_numpy", &yxc_wrapper::to_ndarray<nb::numpy>,
                     "Returns a copy of the underlying data as a Numpy "
                     "NDArray")
                .def(
                    "tobytes", &yxc_wrapper::get_bytes,
                    "Returns the image data converted to the wrapper dtype as "
                    "a bytes object")
                .def("to_numpy_raw", &yxc_wrapper::reshape_to_yxc<nb::numpy>,
                     "Returns a direct reshaped view into the image data")
                .def_prop_ro(
                    "shape", &yxc_wrapper::shape_yxc<size_t, true>,
                    "Returns the shape (size along each axis) tuple of the "
                    "image in xyzc order")
                .def("assign", &yxc_wrapper::assign_try,
                     "Assigns the given array-compatible object's data to the "
                     "image",
                     "array"_a, "same_dims"_a = true, nb::rv_policy::none)
                .def("assign", &yxc_wrapper::assign_pil,
                     "Assigns the given PIL Image's data to the image",
                     "image"_a, "same_dims"_a = true, nb::rv_policy::none);

        imgcls
            .def_prop_ro(
                "yxc", [](Img &img) { return make_wrapper<TO>(img); }, doc)
            .def_static(
                "from_yxc", &yxc_wrapper::new_image,
                "Constructs an image from the given XYC-ordered ndarray",
                "array"_a)
            .def_static("from_yxc", &yxc_wrapper::new_image_pil,
                        "Constructs an image from the given PIL Image",
                        "image"_a);
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