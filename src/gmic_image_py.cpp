// This file is a subpart of gmicpy.cpp, split for readability but part of the
// same translation unit

template <class T = gmic_pixel_type>
class gmic_image_py {
    static constexpr auto ARRAY_INTERFACE = "__array_interface__";
    static constexpr auto DLPACK_INTERFACE = "__dlpack__";
    static constexpr auto DLPACK_DEVICE_INTERFACE = "__dlpack_device__";

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
        LOG_TRACE("Allocating " << dest << endl);
        nb::capsule owner(dest, [](void *p) noexcept {
            LOG_TRACE("Releasing " << p << endl);
            delete[] static_cast<float *>(p);
        });
        int64_t strides_src[4]{0, 0, 0, 0}, strides_dst[4]{0, 0, 0, 0};
        size_t shape[4]{1, 1, 1, 1};

        for (int64_t s = 1, i = 0; i < array.ndim(); ++i) {
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

    // ReSharper disable CppTemplateParameterNeverUsed
    template <class I, class... Args>
    struct can_native_init : false_type {};
    // ReSharper restore CppTemplateParameterNeverUsed

    template <class... Args>
    struct can_native_init<
        enable_if_t<
            is_same_v<Img,
                      decltype(Img(declval<trans::translated<Args>>()...))>,
            Img>,
        Args...> : true_type {};

    template <class... Args>
    static void new_image(Img *img, Args... args)
    {
        if constexpr (can_native_init<Img, Args...>::value) {
            LOG_DEBUG(trans::assign_signature<Args...>("new_image") << endl);
            new (img) Img(trans::translate<Args>(args)...);
        }
        else {
            new (img) Img();
            assign(*img, args...);  // NOLINT(*-unnecessary-value-param)
        }
    }

    template <class... Args>
    static auto assign(Img &img, Args... args)
        -> enable_if_t<is_lvalue_reference_v<decltype(Img{}.assign(
                           declval<trans::translated<Args>>()...))>,
                       Img &>
    {
        LOG_DEBUG(trans::assign_signature<Args...>("assign") << endl);
        img.assign(trans::translate<Args>(args)...);
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
        return gmic_image_py::as_ndarray<>(img);
    }

    template <class t>
    struct dtype_typestr {};
    template <signed_integral t>
    struct dtype_typestr<t> {
        static constexpr char typestr = 'i';
    };
    template <unsigned_integral t>
    struct dtype_typestr<t> {
        static constexpr char typestr = 'u';
    };
    template <floating_point t>
    struct dtype_typestr<t> {
        static constexpr char typestr = 'f';
    };

    static nb::object array_interface(Img &img)
    {
        LOG_TRACE();
        nb::dict ai{};
        char type[3] = "??";
        type[0] = endian::native == endian::little ? '<' : '>';
        type[1] = dtype_typestr<T>::typestr;
        ai["typestr"] =
            cast(string(type) + to_string(sizeof(T)), nb::rv_policy::copy);
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
    static container strides(Img &img)
    {
        constexpr I S = bytes ? static_cast<I>(sizeof(T)) : 1;
        return {S, S * img.width(), S * img.width() * img.height(),
                S * img.width() * img.height() * img.depth()};
    }

    /// Returns the shape of the image, in xyzc order
    template <integral I = size_t, class container = tuple<I, I, I, I>>
    static container shape(Img &img)
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

    [[nodiscard]] static string str(Img &img)
    {
        char pyimgloc[128] = "";
#if DEBUG == 1
        if (const nb::object pyimg = nb::find(img); pyimg.is_valid()) {
            sprintf(pyimgloc, ", nb::object: %p", pyimg.ptr());
        }
        else {
            sprintf(pyimgloc, ", nb::object: none");
        }
#endif
        stringstream out;
        out << "<" << nb::type_name(nb::type<Img>()).c_str() << " at "
            << static_cast<const void *>(&img)
            << ", data at: " << static_cast<const void *>(img.data())
            << pyimgloc << ", w×h×d×s=" << img.width() << "×" << img.height()
            << "×" << img.depth() << "×" << img.spectrum() << ">";
        return out.str();
    }

    template <class... Args>
    using assign_t = Img &(*)(Img &, Args...);
    template <class... Args>
    using new_image_t = void (*)(Img *, Args...);

    static void bind(nb::module_ &m)
    {
        // ReSharper disable CppIdenticalOperandsInBinaryExpression
        auto cls =
            nb::class_<Img>(m, CLASSNAME)
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
            trans::assign_signature_doc<TYPES>(doc_buf, doc, "CImg<T>"),     \
            ##__VA_ARGS__)                                                   \
        .def(funcname, static_cast<assign_t<TYPES>>(&gmic_image_py::assign), \
             trans::assign_signature_doc<TYPES>(doc_buf, doc,                \
                                                "CImg<T>::assign"),          \
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
    }
#undef IMAGE_ASSIGN
#undef ARGS
};
