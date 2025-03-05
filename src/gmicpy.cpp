#include "gmicpy.h"

#include <bit>
#include <functional>
#include <iostream>
#include <optional>
#include <ranges>
#include <source_location>
#include <sstream>
#include <type_traits>

#include "logging.h"

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

using Level = DebugLogger::Level;
static DebugLogger LOG;
#define LOG_VA_ELSE(arg, ...) arg
#define LOG_(level, ...)                                                    \
    {                                                                       \
        static constexpr function_name_stripped<strlen(                     \
            source_location::current().function_name())>                    \
            fname(source_location::current().function_name());              \
        LOG << level                                                        \
            << fname.str() LOG_VA_ELSE(__VA_OPT__(<< ": " << __VA_ARGS__, ) \
                                       << std::endl);                       \
    }
#define LOG_INFO(...) LOG_(Level::Info __VA_OPT__(, __VA_ARGS__))
#define LOG_DEBUG(...) LOG_(Level::Debug __VA_OPT__(, __VA_ARGS__))
#define LOG_TRACE(...) LOG_(Level::Trace __VA_OPT__(, __VA_ARGS__))

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
    const auto dtypes = {make_pair(nb::dtype<int8_t>(), "int8_t"),
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
    const auto dt = ranges::find_if(
        dtypes, [&](auto pair) { return pair.first == a.dtype(); });
    if (dt != dtypes.end()) {
        buf << dt->second << endl;
    }
    else {
        buf << "<unknown>" << endl;
    }
    return buf.str();
}

template <ranges::sized_range V>
nb::tuple to_tuple(V v, nb::rv_policy rv = nb::rv_policy::automatic)
{
    const size_t size = v.size();
    auto result =
        nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(size)));
    size_t i = 0;
    for (const auto &e : v) {
        PyTuple_SET_ITEM(result.ptr(), i++, nb::cast(e, rv).ptr());
    }

    return result;
}

template <class F, integral I, class V = decltype(declval<F>()(declval<I>()))>
    requires std::is_invocable_r_v<V, F, I>
nb::tuple to_tuple_func(I size, F get,
                        nb::rv_policy rv = nb::rv_policy::automatic)
{
    auto result =
        nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(size)));
    for (I i = 0; i < size; ++i) {
        auto ptr = nb::cast(get(i), rv);
        PyTuple_SET_ITEM(result.ptr(), i, ptr.release().ptr());
    }

    return result;
}

template <class Sh, class St>
bool is_f_contig(unsigned short ndim, Sh *shape, St *strides)
{
    decltype(*shape * *strides) acc = 1;
    for (int i = 0; i < ndim; ++i) {
        if (strides[i] != acc)
            return false;
        acc *= shape[i];
    }
    return true;
}

template <class... P>
bool is_f_contig(nb::ndarray<P...> arr)  // NOLINT(*-unnecessary-value-param)
{
    return is_f_contig(arr.ndim(), arr.shape_ptr(), arr.stride_ptr());
}

#include "gmic_image_py.cpp"  // NOLINT(*-suspicious-include)
#include "gmic_list_py.cpp"   // NOLINT(*-suspicious-include)

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

    static string str(const gmic &inst)
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<gmic>()).c_str() << " object at "
            << &inst << '>';
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
try {
#if DEBUG == 1
    LOG = DebugLogger{&cerr, Level::Nothing};
    if (auto loglevel = getenv("GMICPY_LOGLEVEL")) {
        auto lvl = atoi(loglevel);
        LOG.set_log_level(lvl);
        LOG_INFO("Setting log level to " << lvl);
    }
#endif
    {
        static char version[16];
        constexpr auto patch = gmic_version % 10,
                       minor = gmic_version / 10 % 10,
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

    static_assert(trans::is_translatable<string>::value);

#if DEBUG == 1
    m.def("inspect", &inspect, "array"_a, "Inspects a N-dimensional array");
    m.def(
        "set_debug", [](const int lvl) { LOG.set_log_level(lvl); }, "level"_a,
        "Sets the debug log level (1=info, 2=debug, 3=trace)");
#endif

    LOG_TRACE("Binding gmic.Image class" << endl);
    gmic_image_py<>::bind(m);

    LOG_TRACE("Binding gmic.ImageList class" << endl);
    gmic_list_py<>::bind(m);

    LOG_TRACE("Binding gmic.StringList class" << endl);
    gmic_charlist_py::bind(m);

    LOG_TRACE("Binding gmic.Gmic class" << endl);
    interpreter_py<>::bind(m);

    LOG_TRACE("Binding gmic.GmicException class" << endl);
    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
}
catch (const exception &ex) {
    cerr << ex.what() << endl;
    throw;
}

}  // namespace gmicpy
