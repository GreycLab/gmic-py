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
    type[0] = sizeof(T) == 1                     ? '|'
              : endian::native == endian::little ? '<'
                                                 : '>';

    static_assert(signed_integral<T> || unsigned_integral<T> ||
                  floating_point<T>);
    if constexpr (is_same_v<bool, T>)
        type[1] = 'b';
    else if constexpr (signed_integral<T>)
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

enum cast_policy : uint8_t {
    THROW,
    CLAMP,
    NOCHECK,
};

template <size_t ndim, class Ti, class To, cast_policy policy>
void copy_ndarray_data(const Ti *src, const int64_t *istrides,
                       const int64_t *shape, To *dst, const int64_t *ostrides)
{
#define IF_THROW_OR_CLAMP(cond, throwerr, clampval) \
    if (cond) {                                     \
        if constexpr (policy == THROW) {            \
            err = (throwerr);                       \
        }                                           \
        else {                                      \
            d = (clampval);                         \
        }                                           \
    }
    for (size_t a = 0; a < shape[0]; a++) {
        const size_t ioff = a * istrides[0], ooff = a * ostrides[0];
        if constexpr (ndim > 1) {
            copy_ndarray_data<ndim - 1, Ti, To, policy>(
                src + ioff, istrides + 1, shape + 1, dst + ooff, ostrides + 1);
        }
        else {
            auto s = src[ioff];
            auto d = static_cast<To>(s);
            if constexpr (!is_same_v<Ti, To> && !is_floating_point_v<To> &&
                          policy != NOCHECK) {
                auto cs = s;
                if constexpr (is_floating_point_v<Ti>) {
                    cs = std::trunc(cs);
                }
                if (static_cast<Ti>(d) != cs) {
                    constexpr numeric_limits<To> olims;
                    const char *err = nullptr;
                    if constexpr (is_floating_point_v<Ti>) {
                        IF_THROW_OR_CLAMP(isinf(s),
                                          "Tried casting infinite value to "
                                          "an integer type",
                                          s > 0 ? olims.max() : olims.min())
                        else IF_THROW_OR_CLAMP(!isfinite(s),
                                               "Tried casting non-finite "
                                               "value to an integer type",
                                               0);
                    }
                    IF_THROW_OR_CLAMP(s > olims.max(),
                                      "Value too large for destination type",
                                      olims.max())
                    else IF_THROW_OR_CLAMP(
                        s < olims.min(),
                        olims.min() == 0
                            ? "Tried casting negative value to "
                              "unsigned type"
                            : "Value too low for destination type",
                        olims.min());
                    if constexpr (policy == THROW)
                        throw out_of_range(err);
                }
            }
            dst[ooff] = d;
        }
    }
#undef IF_THROW_OR_CLAMP
}
template <size_t ndim, class Ti, class To>
void copy_ndarray_data(const Ti *src, const int64_t *istrides,
                       const int64_t *shape, To *dst, const int64_t *ostrides,
                       const cast_policy policy)
{
    int64_t _ostrides[ndim];
    if (ostrides == nullptr) {
        for (int64_t s = 1, i = 0; i < ndim; ++i) {
            _ostrides[i] = s;
            s *= static_cast<int64_t>(shape[i]);
        }
        ostrides = _ostrides;
    }
    switch (policy) {  // Runtime-to-compiletime for performance reasons
        case THROW:
            copy_ndarray_data<ndim, Ti, To, THROW>(src, istrides, shape, dst,
                                                   ostrides);
            break;
        case CLAMP:
            copy_ndarray_data<ndim, Ti, To, CLAMP>(src, istrides, shape, dst,
                                                   ostrides);
            break;
        case NOCHECK:
            copy_ndarray_data<ndim, Ti, To, NOCHECK>(src, istrides, shape, dst,
                                                     ostrides);
            break;
        default:
            abort();
    }
}

/**
 * Copies a ndarray. Will reorder the data so that the data is
 * C-contiguous.
 * @tparam ndim number of dimensions
 * @tparam Ti Input datatype
 * @tparam To Output datatype
 * @param array Input array whose data is <strong>in SDHW (gmic)
 * order</strong>
 * @param policy Cast policy (error / clamp / ignore)
 * @param deleter Whether or not to add a capsule owner that will take care
 * of freeing memory
 * @return A copy of the ndarray with the same data for a given set of
 * coordinates, but reordered C-style
 */
template <size_t ndim, class Ti, class To = Ti, class... P>
static nb::ndarray<To, nb::device::cpu, nb::ndim<ndim>, P...> copy_ndarray(
    const nb::ndarray<const Ti, nb::device::cpu, nb::ndim<ndim>, P...> &array,
    const cast_policy policy, bool deleter = true)
{
    const Ti *src = array.data();
    To *dest = new To[array.size()];
    LOG_TRACE("Allocating " << static_cast<void *>(dest) << endl);
    nb::capsule owner(dest, deleter ? [](void *p) noexcept {
        LOG_TRACE("Releasing " << p << endl);
        delete[] static_cast<float *>(p);
    } : [](void *) noexcept {});
    size_t shape[ndim];
    for (int64_t i = 0; i < ndim; ++i) {
        shape[i] = array.shape(i);
    }

    nb::ndarray<To, nb::device::cpu, nb::ndim<ndim>, P...> outarray(
        dest, ndim, shape, owner);

    copy_ndarray_data<ndim, Ti, To>(src, array.stride_ptr(), array.shape_ptr(),
                                    dest, outarray.stride_ptr(), policy);

    return outarray;
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
