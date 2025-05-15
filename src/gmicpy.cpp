#include "gmicpy.hpp"

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

DebugLogger LOG{};

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

ostream &operator<<(ostream &out, const CImg<> &img)
{
    out << "<" << nb::type_name(nb::type<CImg<>>()).c_str() << " at " << &img
        << ", data at: " << img.data();
#if DEBUG == 1
    out << ", nb::object: ";
    if (const nb::object pyimg = nb::find(img); pyimg.is_valid()) {
        out << pyimg.ptr();
    }
    else {
        out << "none";
    }
#endif
    out << ", w×h×d×s=" << img.width() << "×" << img.height() << "×"
        << img.depth() << "×" << img.spectrum() << ">";
    return out;
}

template <size_t N = 8>
constexpr array<char, N> get_gmic_version()
{
    char version[N] = "?.?.?";
    constexpr auto patch = gmic_version % 10, minor = gmic_version / 10 % 10,
                   major = gmic_version / 100;
    static_assert(major < 10);
    version[0] = '0' + major;
    version[2] = '0' + minor;
    version[4] = '0' + patch;
    return to_array(version);
}

NB_MODULE(gmic, m)
try {
#if DEBUG == 1
    LOG = DebugLogger{&cerr, Level::Nothing};
    if (auto loglevel = getenv("GMICPY_LOGLEVEL")) {
        auto lvl = atoi(loglevel);
        LOG.set_log_level(lvl);
        LOG_INFO("Setting log level to " << lvl << endl);
    }
#endif
    {
        constexpr auto version = get_gmic_version();
        m.attr("__version__") = version.data();
#if defined(GMICPY_VERSION)
        constexpr auto gmicpy_version = Py_STRINGIFY(GMICPY_VERSION);
        constexpr string_view gmicpy_view(gmicpy_version);
        static_assert(
            gmicpy_view.starts_with(version.data()),
            "GMICPY_VERSION (" Py_STRINGIFY(GMICPY_VERSION)
            ") does not match gmic_version (" Py_STRINGIFY(gmic_version) ")");
        m.attr("__pyversion__") = gmicpy_version;
#endif
    }
    {
#define IS_DEFINED(macro)                                                   \
    {                                                                       \
        #macro,                                                             \
            (strcmp(#macro, Py_STRINGIFY(macro)) != 0 ? Py_STRINGIFY(macro) \
                                                      : nullptr)            \
    }

        static char build[256];
        stringstream build_str;
        build_str << "Built on " __DATE__ << " at " << __TIME__;
        strncpy(build, build_str.str().c_str(), size(build));
        m.attr("__build__") = build;
        const map<const char *, const char *> flags{
            IS_DEFINED(DEBUG),  // NOLINT(*-branch-clone)
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
            IS_DEFINED(cimg_use_zlib)};
        m.attr("__build_flags__") = flags;
    }

#if DEBUG == 1
    m.def("inspect", &inspect, "array"_a, "Inspects a N-dimensional array");
    m.def(
        "set_debug", [](const int lvl) { LOG.set_log_level(lvl); }, "level"_a,
        "Sets the debug log level (1=info, 2=debug, 3=trace)");
#endif

    LOG_INFO("Binding gmic module" << endl);
    bind_gmic_image(m);
    bind_gmic_list(m);

    LOG_DEBUG("Binding gmic.GmicException class" << endl);
    const auto gmic_ex = nb::exception<  // NOLINT(*-throw-keyword-missing)
        gmic_exception>(m, "GmicException");
}
catch (const exception &ex) {
    cerr << ex.what() << endl;
    throw;
}

}  // namespace gmicpy
