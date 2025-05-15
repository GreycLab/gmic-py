#ifndef UTILS_HPP
#define UTILS_HPP
#include "gmicpy.hpp"

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

template <class T>
static constexpr array<char, 8> get_typestr()
{
    array<char, 8> type{};
    if constexpr (!is_void_v<T>) {
        type[0] = sizeof(T) == 1                     ? '|'
                  : endian::native == endian::little ? '<'
                                                     : '>';

        static_assert(signed_integral<T> || unsigned_integral<T> ||
                      floating_point<T>);
        if constexpr (signed_integral<T>)
            type[1] = 'i';
        else if constexpr (unsigned_integral<T>)
            type[1] = 'u';
        else if constexpr (floating_point<T>)
            type[1] = 'f';

        size_t pos = 2;
        const size_t size = sizeof(T);
        static_assert(size < 100);
        if (size >= 10)
            type[pos++] = '0' + static_cast<char>(size / 10);
        type[pos++] = '0' + static_cast<char>(size % 10);
        type[pos] = '\0';
    }
    return type;
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

template <class... Args>
struct assign_signature {
    const char *const func_name;
    vector<string> arg_names{};

    explicit assign_signature(const char *func_name) : func_name(func_name)
    {
#ifdef __GNUC__
        int status;
        unsigned long bufsize = 128;
        char *buf = static_cast<char *>(malloc(sizeof(char) * bufsize));
        const char *names[] = {typeid(Args).name()...};
        for (const char *name : names) {
            char *demang = abi::__cxa_demangle(name, buf, &bufsize, &status);
            if (demang == nullptr)
                throw std::runtime_error("Could not demangle function name");
            arg_names.emplace_back(demang);
            buf = demang;
        }
        free(buf);
#else
        arg_names = {typeid(translated<Args>).name()...};
#endif
    }

    const vector<string> &get_arg_names() { return arg_names; }
};

template <class... Args>
ostream &operator<<(ostream &out, assign_signature<Args...> sig)
{
    out << sig.func_name << "(";
    const auto argtypes = sig.get_arg_names();
    bool first = true;
    for (const auto &t : argtypes) {
        if (first) {
            first = false;
        }
        else {
            out << ", ";
        }
        out << t;
    }
    out << ')';
    return out;
}

/**
 * Appends the signature of a given function, with its arguments' types
 * translated, to a given docstring, and writes it to a char buffer
 * @tparam Args Types of the pre-translation arguments
 * @tparam N Buffer size
 * @param buf Char buffer to write to
 * @param doc Documentation to append the signature to
 * @param func Name to use in the signature for the documented function
 * @return buf passthrough
 */
template <class... Args, size_t N = 1024>
static const char *assign_signature_doc(char buf[N], const char *doc,
                                        const char *func)
{
    stringstream out;
    out.rdbuf()->pubsetbuf(buf, N);
    out << doc << "\n\n" << "Binds " << assign_signature<Args...>(func);
    if (out.tellp() >= N)
        throw out_of_range("Function signature is too long for buffer");
    buf[out.tellp()] = '\0';
    return buf;
}

template <ranges::sized_range V>
nb::tuple to_tuple(V v, nb::rv_policy rv = nb::rv_policy::automatic)
{
    const size_t size = v.size();
    auto result =
        nb::steal<nb::tuple>(PyTuple_New(static_cast<Py_ssize_t>(size)));
    size_t i = 0;
    for (const auto &e : v) {
        PyTuple_SetItem(result.ptr(), i++, nb::cast(e, rv).ptr());
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
        PyTuple_SetItem(result.ptr(), i, ptr.release().ptr());
    }

    return result;
}

}  // namespace gmicpy

#endif  // UTILS_HPP
