#ifndef LOGGING_H
#define LOGGING_H
#include <algorithm>
#include <array>
#include <cstring>
#include <ostream>
#include <source_location>

namespace gmicpy {

class DebugLogger {
   public:
    enum class Level : uint8_t { Nothing = 0, Info = 1, Debug = 2, Trace = 3 };
    static constexpr const char *LEVEL_NAMES[] = {"", "INFO", "DEBUG",
                                                  "TRACE"};

   private:
    Level level;
    std::ostream *out;
    bool enabled = false;

   public:
    explicit DebugLogger(std::ostream *out = nullptr,
                         const Level level = Level::Nothing)
        : level(level), out(out) {};

    void set_log_level(const Level lvl) { level = lvl; }

    void set_log_level(const int lvl)
    {
        if (lvl < 0 || lvl > static_cast<uint8_t>(Level::Trace))
            throw std::out_of_range("Invalid log level");
        set_log_level(static_cast<Level>(lvl));
    }

    template <class T>
        requires std::is_same_v<std::ostream &,
                                decltype(*out << std::declval<T &&>())>
    DebugLogger &operator<<(T &&x)
    {
        if (enabled && out != nullptr) {
            *out << std::forward<T>(x);
        }
        return *this;
    }

    DebugLogger &operator<<(std::ostream &(*manip)(std::ostream &))
    {
        if (enabled && out != nullptr) {
            *out << manip;
        }
        return *this;
    }

    DebugLogger &operator<<(const Level lvl)
    {
        enabled = lvl <= level;
        if (enabled && out != nullptr) {
            *out << "[" << LEVEL_NAMES[static_cast<uint8_t>(lvl)] << "] ";
        }
        return *this;
    }
};

/**
 *
 * @tparam N Utility struct to shorten function names (as returned by
 * std::source_location::function_name) by stripping a few given namespaces
 */
template <size_t N>
class function_name_stripped {
    static constexpr const char *strip_ns[][2] = {
        {"gmicpy::", ""},
        {"cimg_library::", ""},
        {"std::", ""},
        {"nanobind::", "nb::"},
        {"gmic_image_py::Img", "CImg"},
        {"gmic_image_py", "Image"},
        {"gmic_list_py<T>", "ImageList"},
        {"gmic_list_py<char>", "CharList"},
        {"gmic_list_base<T>", "ImageList"},
        {"gmic_list_base<char>", "CharList"}};
    const std::array<char, N + 1> name;

   public:
    constexpr explicit function_name_stripped(const char *oname)
        : name(strip_namespace(oname))
    {
    }

    // ReSharper disable once CppDFAUnreachableFunctionCall
    static constexpr bool isalnum(const char c)
    {
        return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' ||
               c >= 'A' && c <= 'Z' || c == '_';
    }

    // ReSharper disable once CppDFAUnreachableFunctionCall
    static constexpr std::array<char, N + 1> strip_namespace(const char *oname)
    {
        char name[N + 1];
        std::copy_n(oname, N + 1, name);
        std::string_view fname(name);
        for (auto &[ns, repl] : strip_ns) {
            std::string::size_type from = 0, to = 0, pos;
            const auto nslen = std::strlen(ns), replen = std::strlen(repl);
            do {
                auto search_from = from;
                do {
                    pos = fname.find(ns, search_from);
                    search_from = pos + 1;
                } while (pos != std::string::npos && pos > 0 &&
                         isalnum(name[pos - 1]));
                if (to != from) {
                    std::copy_n(name + from,
                                pos == std::string_view::npos ? N - from + 1
                                                              : pos - from,
                                name + to);
                }
                to += pos - from;
                from = pos + nslen;
                if (pos != std::string::npos && replen > 0) {
                    std::copy_n(repl, replen, name + to);
                    to += replen;
                }
            } while (pos != std::string::npos);
        }
        return std::to_array(name);
    }

    [[nodiscard]] constexpr const char *str() const { return name.data(); }
};

using Level = DebugLogger::Level;
extern DebugLogger LOG;
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
}  // namespace gmicpy

#endif  // LOGGING_H
