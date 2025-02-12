// This file is a subpart of gmicpy.cpp, split for readability but part of the
// same translation unit

template <class T>
class gmic_list_base {
   protected:
    CImgList<T> list{};
    template <class... Args>
    explicit gmic_list_base(Args... args) : list(args...)
    {
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

    void set(unsigned int i, Item item) { list(i).assign(item); }
};

template <>
class gmic_list_base<char> {
   protected:
    CImgList<char> list{};
    template <class... Args>
    explicit gmic_list_base(Args... args) : list(args...)
    {
    }

    virtual ~gmic_list_base() = default;

   public:
    using Item = string;
    static constexpr auto CLASSNAME = "StringList";

    [[nodiscard]] Item get(const unsigned int i) const { return {list(i)}; }

    void set(const unsigned int i, const Item &item)
    {
        list(i).assign(CImg<char>::string(item.c_str()));
    }
};

template <class T = gmic_pixel_type>
class gmic_list_py : public gmic_list_base<T> {
    using Item = typename gmic_list_base<T>::Item;
    using Base = gmic_list_base<T>;

   public:
    using Base::Base;

    ~gmic_list_py() override = default;

    CImgList<T> &list() { return Base::list; }

    auto operator[](unsigned int i) { return this->get(i); }
    auto operator[](unsigned int i) const { return this->get(i); }

    class iterator
        : std::iterator<std::forward_iterator_tag, remove_reference_t<Item>> {
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

    string str()
    {
        stringstream out;
        out << '<' << nb::type_name(nb::type<decltype(*this)>()).c_str()
            << '[';
        bool first = true;

        for (auto item : *this) {
            if (first)
                first = false;
            else
                out << ", ";
            if constexpr (is_same_v<CImg<T>, Item>) {
                out << item.str();
            }
            else {
                out << item;
            }
        }
        out << "]>";

        return out.str();
    }

    static void bind(nb::module_ &m)
    {
        nb::class_<gmic_list_py>(m, gmic_list_base<T>::CLASSNAME)
            .def("__iter__", &gmic_list_py::iter)
            .def("__len__", &gmic_list_py::size)
            .def("__str__", &gmic_list_py::str)
            .def("__getitem__", &gmic_list_py::get, "i"_a,
                 nb::rv_policy::reference_internal)
            .def("__setitem__", &gmic_list_py::set, "i"_a, "v"_a);
    }
};

using gmic_charlist_py = gmic_list_py<char>;