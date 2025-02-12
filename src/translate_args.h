#ifndef TRANSLATE_ARGS_H
#define TRANSLATE_ARGS_H

namespace gmicpy::trans {
/**
 * Class that manages the translation of values back and forth between the
 * python binding and the native GMIC/CImg library. Within this class,
 * "translating" describes the python binding -> libgmic direction, and
 * "untranslating" the opposite process.
 */
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

/// Get a void pointer from any pointer or reference, for storing a into a
/// registry
template <class A>
    requires is_reference_v<A> || is_pointer_v<A>
static auto get_void_p(A val)
{
    if constexpr (is_reference_v<A>) {
        return static_cast<const void *>(&val);
    }
    else {
        return static_cast<const void *>(val);
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
using registry = std::map<const void *, pair<const void *, const type_info *>>;

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
static translated<A> translate(A a, registry *reg = nullptr)
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
[[nodiscard]] static untranslated<A> untranslate(A a, registry *reg = nullptr)
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
 * @tparam N Buffer size
 * @param buf Char buffer to write to
 * @param doc Documentation to append the signature to
 * @param func Name to use in the signature for the documented function
 * @return buf passthrough
 */
template <class... Args, size_t N = 1024>
static const char *assign_signature(char buf[N], const char *doc,
                                    const char *func)
{
    const char *max = buf + N;
    buf += snprintf(buf, N, "%s\n\nBinds %s(", doc, func);
    const vector<const char *> argtypes{typeid(translated<Args>).name()...};
    bool first = true;
    for (const auto &t : argtypes) {
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

};  // namespace gmicpy::trans

#endif  // TRANSLATE_ARGS_H
