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
        nb::capsule owner(
            dest, [](void *p) noexcept { delete[] static_cast<float *>(p); });
        int64_t strides_src[4]{0, 0, 0, 0}, strides_dst[4]{0, 0, 0, 0};
        size_t shape[4]{1, 1, 1, 1};

        for (int64_t s = 1, i = array.ndim() - 1; i >= 0; --i) {
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
            LOG_DEBUG();
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
        LOG_DEBUG();
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
        img.assign(shape[3], shape[2], shape[1], shape[0]);
        LOG_DEBUG("Copying data from " << arr.data());

        if (is_c_contig(arr)) {
            LOG << ", C-contig (std::copy_n)" << endl;
            copy_n(arr.data(), arr.size(), img.data());
        }
        else {
            LOG << ", Non-C-contig (loop)" << endl;
            for (size_t c = 0; c < shape[0]; c++) {
                const size_t offc = c * strides[0];
                for (size_t d = 0; d < shape[1]; d++) {
                    const size_t offd = offc + d * strides[1];
                    for (size_t y = 0; y < shape[2]; y++) {
                        const size_t offy = offd + y * strides[2];
                        for (size_t x = 0; x < shape[3]; x++) {
                            img(x, y, d, c) =
                                arr.data()[offy + x * strides[3]];
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
            "Construct image with specified size and initialize pixel "
            "values",
            ARGS(unsigned int, unsigned int, unsigned int, unsigned int),
            "width"_a, "height"_a, "depth"_a = 0, "channels"_a = 0);
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
                     "Construct an image from an array-like object",
                     ARGS(CTNDArray<>), "array"_a);
    }
#undef IMAGE_ASSIGN
#undef ARGS
};
