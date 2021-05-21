
#include <iterator>     // for std::forward_iterator_tag
#include <type_traits>  // for std::conditional_t

namespace jet {

namespace details {
static inline size_t num_pairs(size_t n) { return (n * (n + 1)) / 2; }
} // namespace details

// A symmetric matrix with values in V
// This class wraps a pointer to the data and manages its interpretation as a
// symmetric matrix, it must be treated as carefully as a raw pointer.
template <typename V> class SymmetricMatrixView {
  template <bool IsConst> class iterator_template {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = V;
    using pointer = std::conditional_t<IsConst, V const *, V *>;
    struct reference {
      size_t i;
      size_t j;
      std::conditional_t<IsConst, V const &, V &> value;
    };

    iterator_template(pointer data, size_t i, size_t j)
        : ptr{data}, m_i{i}, m_j{j} {}

    // Conversion from const to non-const iterator
    template <bool OtherIsConst,
              class = std::enable_if_t<IsConst && !OtherIsConst>>
    iterator_template(iterator_template<OtherIsConst> const &other)
        : ptr(other.ptr), m_i(other.m_i), m_j(other.m_j) {}

    size_t i() const { return m_i; }
    size_t j() const { return m_j; }

    reference operator*() const { return {m_i, m_j, *ptr}; }
    pointer operator->() const { return ptr; }

    iterator_template &operator++() {
      ++ptr;
      ++m_i;
      if (m_i > m_j) {
        m_i = 0;
        ++m_j;
      }
      return *this;
    }
    iterator_template operator++(int) {
      iterator_template temp = *this;
      ++(*this);
      return temp;
    }
    bool operator==(iterator_template const &other) const {
      return ptr ==
             other.ptr; // within a jet i and j can be calculated from ptr
    }
    bool operator!=(iterator_template const &other) const {
      return ptr != other.ptr;
    }

  private:
    pointer ptr{};
    size_t m_i{};
    size_t m_j{};
  };

public:
  SymmetricMatrixView() {}
  SymmetricMatrixView(V *data, size_t dim) : m_data{data}, m_dim{dim} {}

  using Iterator = iterator_template<false>;
  using ConstIterator = iterator_template<true>;

  ConstIterator begin() const { return ConstIterator(m_data, 0, 0); }
  ConstIterator end() const {
    return ConstIterator(m_data + details::num_pairs(m_dim), 0, m_dim);
  }
  Iterator begin() { return Iterator(m_data, 0, 0); }
  Iterator end() {
    return Iterator(m_data + details::num_pairs(m_dim), 0, m_dim);
  }
  ConstIterator cbegin() const { return begin(); }
  ConstIterator cend() const { return end(); }

  size_t dim() const { return m_dim; }

  V &operator()(size_t i, size_t j) { return m_data[symmetric_index(i, j)]; }
  V const &operator()(size_t i, size_t j) const {
    return m_data[symmetric_index(i, j)];
  }

  static inline size_t symmetric_index(size_t i, size_t j) {
    if (i > j)
      std::swap(i, j);
    return i + details::num_pairs(j);
  }
  static inline size_t symmetric_index_ordered(size_t i, size_t j) {
    return i + details::num_pairs(j);
  }

private:
  V *m_data{};
  size_t m_dim{};
};

} // namespace jet