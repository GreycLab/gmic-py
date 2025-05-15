#ifndef GMIC_LIST_PY_HPP
#define GMIC_LIST_PY_HPP

#include "gmicpy.hpp"

namespace gmicpy {
namespace nb = nanobind;
using namespace nanobind::literals;
using namespace std;
using namespace cimg_library;

template <class T>
class gmic_list_base {
   protected:
    CImgList<T> list{};

    template <class... Args>
    explicit gmic_list_base(Args... args) : list(args...)
    {
        LOG_DEBUG("Data is at: " << list._data << endl);
    }

    virtual ~gmic_list_base() = default;

   public:
    static constexpr auto CLASSNAME = "ImageList";
    using Item = CImg<T> &;

    [[nodiscard]] Item get(unsigned int i)
    {
        if (i >= list.size())
            throw out_of_range("Out of range or gmic_list_py object");
        return list(i);
    }

    void set(unsigned int i, const CImg<T> &item)
    {
        if (i >= list.size())
            throw out_of_range("Out of range or gmic_list_py object");
        list(i).assign(item);
    }

    void move_set(unsigned int i, CImg<T> &&item)
    {
        if (i >= list.size())
            throw out_of_range("Out of range or gmic_list_py object");
        item.move_to(list(i));
    }
};

template <>
class gmic_list_base<char> {
   protected:
    CImgList<char> list{};
    template <class... Args>
    explicit gmic_list_base(Args... args) : list(args...)
    {
        LOG_DEBUG("Data is at: " << list._data << endl);
    }

    virtual ~gmic_list_base() = default;

   public:
    using Item = string;
    static constexpr auto CLASSNAME = "StringList";

    [[nodiscard]] Item get(const unsigned int i) const
    {
        if (i >= list.size())
            throw out_of_range("Out of range or gmic_list_py object");
        return {list(i)};
    }

    void set(const unsigned int i, const Item &item)
    {
        if (i >= list.size())
            throw out_of_range("Out of range or gmic_list_py object");
        list(i).assign(CImg<char>::string(item.c_str()));
    }

    void move_set(unsigned int i, Item &&item) { set(i, item); }
};

template <class T = gmic_pixel_type>
class gmic_list_py : public gmic_list_base<T> {
    using Base = gmic_list_base<T>;
    using Item = typename Base::Item;
    using RawItem = remove_reference_t<Item>;

   public:
    gmic_list_py() : Base() {}

    explicit gmic_list_py(const nb::sequence &seq) : Base(len(seq))
    {
        const size_t N = size();
        size_t i = 0;
        vector<RawItem> imgs(N, RawItem{});
        // First check that the sequence contains only images or compatible
        for (const auto &img : seq) {
            if (i >= size())
                throw invalid_argument(
                    "Sequence contains more items than expected");
            if (!nb::try_cast(img, imgs[i++], true))
                throw nb::type_error(
                    "Sequence contains object(s) that isn't and cannot be "
                    "made into a gmic Image");
        }
        if (i < N)
            throw invalid_argument(
                "Sequence contains less items than expected");

        i = 0;
        for (auto it = make_move_iterator(imgs.begin());
             it != make_move_iterator(imgs.end()); ++it) {
            Base::move_set(i++, *it);
        }
    }

    ~gmic_list_py() override
    {
        LOG_DEBUG("Size: " << size() << ", data at: " << list()._data << endl);
    };

    CImgList<T> &list() { return Base::list; }

    auto operator[](unsigned int i) { return this->get(i); }
    auto operator[](unsigned int i) const { return this->get(i); }

    // TODO fix deprecatied std::iterator
    class iterator : std::iterator<std::forward_iterator_tag, RawItem> {
        gmic_list_py &list;
        unsigned int iter = 0;

       public:
        explicit iterator(gmic_list_py &list, const unsigned int start = 0)
            : list(list), iter(start)
        {
        }

        iterator &operator++()
        {
            ++iter;
            return *this;
        }

        bool operator==(iterator other) const { return iter == other.iter; }
        bool operator!=(iterator other) const { return !operator==(other); }

        auto operator*() const { return list[iter]; }
    };

    size_t size() { return Base::list._width; }

    iterator begin() { return iterator(*this); }

    iterator end() { return iterator(*this, size()); }

    [[nodiscard]] auto iter()
    {
        return nb::make_iterator(nb::type<gmic_list_py>(), "iterator",
                                 this->begin(), this->end());
    }

    template <bool with_list = false>
    string str()
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<decltype(*this)>()).c_str()
            << " at " << this;
        if constexpr (with_list) {
            out << " [";
            bool first = true;
            for (auto item : *this) {
                if (first)
                    first = false;
                else
                    out << ", ";
                out << item;
            }
            out << ']';
        }
        out << '>';

        return out.str();
    }

    static void bind(nb::module_ &m)
    {
        LOG_DEBUG("Binding gmic." << gmic_list_base<T>::CLASSNAME << " class"
                                  << endl);
        nb::class_<gmic_list_py>(m, gmic_list_base<T>::CLASSNAME)
            .def(nb::init())
            .def(nb::init_implicit<nb::sequence>())
            .def("__iter__", &gmic_list_py::iter)
            .def("__len__", &gmic_list_py::size)
            .def("__str__", &gmic_list_py::str<false>)
            .def("__repr__", &gmic_list_py::str<true>)
            .def("__getitem__", &gmic_list_py::get, "i"_a,
                 nb::rv_policy::reference_internal)
            .def("__setitem__", &gmic_list_py::set, "i"_a, "v"_a);
    }
};

using gmic_charlist_py = gmic_list_py<char>;

class interpreter_py {
    using T = gmic_pixel_type;

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
        LOG_DEBUG("Binding gmic." << CLASSNAME << " class" << endl);
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

void bind_gmic_list(nanobind::module_ &m)
{
    gmic_list_py<>::bind(m);
    gmic_list_py<char>::bind(m);
    interpreter_py::bind(m);
}

}  // namespace gmicpy

#endif  // GMIC_LIST_PY_HPP