#pragma once

#include <assert.h>
#include <vector>

#include "jet.h"
#include "symmetric_matrix_view.h"

namespace jet {

// Currently incomplete, is not yet using SymmetricMatrixView

// Defines a non-owning view to the underlying data of a Jet2 object, this view
// can change the values of parameters and of the value and derivative data, but
// cannot change the number of parameters.  As it is a wrapper around raw pointers
// it must be treated with the same care as a raw pointer.
template <typename V, typename P> class Jet2View {
public:
  using value_type = V;
  using parameter_type = P;
  using owning_jet_type = Jet2<value_type, parameter_type>;

  Jet2View(parameter_type *parameters, value_type *value, value_type *data,
           size_t n);
  Jet2View(owning_jet_type const &jet);

  // A new jet owning a copy of the view's data
  operator owning_jet_type() const;

  // Access to parameters
  size_t num_parameters() const;
  parameter_type const &parameter(size_t i) const;
  parameter_type &parameter(size_t i);

  // Access to value
  value_type &value();
  value_type const &value() const;

  // Access to derivatives
  size_t num_derivatives() const;
  value_type &derivative(size_t i);
  V const &derivative(size_t i) const;

  // Access to second derivatives
  size_t num_second_derivatives() const;
  value_type &second_derivative(size_t i, size_t j);
  value_type const &second_derivative(size_t i, size_t j) const;

  // Access to derivatives via iterators
  //value_type *begin_derivatives();
  //value_type *end_derivatives();
  //value_type const *cbegin_derivatives() const;
  //value_type const *cend_derivatives() const;

  // Access to second derivatives via iterators
  //value_type *begin_second_derivatives();
  //value_type *end_second_derivatives();
  //value_type const *cbegin_second_derivatives() const;
  //value_type const *cend_second_derivatives() const;

private:
  value_type *const m_value_p;
  value_type *const m_data_p;
  parameter_type *const m_parameters_p;
  size_t const m_num_parameters;

  static size_t data_length(size_t n) { return n + num_pairs(n); }
  static size_t num_pairs(size_t n) { return (n * (n + 1)) / 2; }
  static size_t symmetric_index(size_t i, size_t j) {
    if (i > j)
      std::swap(i, j);
    return i + num_pairs(j);
  }
  static size_t symmetric_index_ordered(size_t i, size_t j) {
    return i + num_pairs(j);
  }
};

template <typename V, typename P> struct is_2jet<Jet2View<V, P>> {
  static constexpr bool value = true;
};

template <typename V, typename P>
inline Jet2View<V, P>::operator owning_jet_type() const {
  std::vector<parameter_type> parameters{m_parameters_p,
                                         m_parameters_p + m_num_parameters};
  Jet2<value_type, parameter_type> output{std::move(parameters), value()};

  std::copy(begin_derivatives(), end_second_derivatives(),
            output.begin_derivatives());

  return output;
}

template <typename V, typename P>
inline Jet2View<V, P>::Jet2View(P *parameters, V *value, V *data, size_t n)
    : m_value_p{value}, m_data_p{data}, m_parameters_p{parameters},
      m_num_parameters{n} {}

template <typename V, typename P>
inline Jet2View<V, P>::Jet2View(Jet2<V, P> const &jet)
    : m_value_p{&jet.value()}, m_data_p{jet.cbegin_derivatives()},
      m_parameters_p{jet.parameters().data()}, m_num_parameters{
                                                   jet.num_parameters()} {}

// Access to parameters
template <typename V, typename P>
inline size_t Jet2View<V, P>::num_parameters() const {
  return m_num_parameters;
}
template <typename V, typename P>
inline P const &Jet2View<V, P>::parameter(size_t i) const {
  return m_parameters_p[i];
}
template <typename V, typename P>
inline P &Jet2View<V, P>::parameter(size_t i) {
  return m_parameters_p[i];
}

// Access to value
template <typename V, typename P> inline V &Jet2View<V, P>::value() {
  return *m_value_p;
}
template <typename V, typename P>
inline V const &Jet2View<V, P>::value() const {
  return *m_value_p;
}

// Access to derivatives
template <typename V, typename P>
inline size_t Jet2View<V, P>::num_derivatives() const {
  return m_num_parameters;
}
template <typename V, typename P>
inline V &Jet2View<V, P>::derivative(size_t i) {
  assert(i < m_num_parameters);
  return m_data_p[i];
}
template <typename V, typename P>
inline V const &Jet2View<V, P>::derivative(size_t i) const {
  assert(i < m_num_parameters);
  return m_data_p[i];
}

// Access to second derivatives
template <typename V, typename P>
inline size_t Jet2View<V, P>::num_second_derivatives() const {
  return num_pairs(num_parameters());
}
template <typename V, typename P>
inline V &Jet2View<V, P>::second_derivative(size_t i, size_t j) {
  assert(i < m_num_parameters && j < m_num_parameters);
  return m_data_p[num_derivatives() + symmetric_index(i, j)];
}
template <typename V, typename P>
inline V const &Jet2View<V, P>::second_derivative(size_t i, size_t j) const {
  assert(i < m_num_parameters && j < m_num_parameters);
  return m_data_p[num_derivatives() + symmetric_index(i, j)];
}

// Access to derivatives via iterators
//template <typename V, typename P>
//inline V *Jet2View<V, P>::begin_derivatives() {
//  return m_data_p;
//}
//template <typename V, typename P> inline V *Jet2View<V, P>::end_derivatives() {
//  return m_data_p + num_derivatives();
//}
//template <typename V, typename P>
//inline V const *Jet2View<V, P>::cbegin_derivatives() const {
//  return m_data_p;
//}
//template <typename V, typename P>
//inline V const *Jet2View<V, P>::cend_derivatives() const {
//  return m_data_p + num_derivatives();
//}

// Access to second derivatives via iterators
//template <typename V, typename P>
//inline V *Jet2View<V, P>::begin_second_derivatives() {
//  return end_derivatives();
//}
//template <typename V, typename P>
//inline V *Jet2View<V, P>::end_second_derivatives() {
//  return end_derivatives() + num_pairs(m_num_parameters);
//}
//template <typename V, typename P>
//inline V const *Jet2View<V, P>::cbegin_second_derivatives() const {
//  return cend_derivatives();
//}
//template <typename V, typename P>
//inline V const *Jet2View<V, P>::cend_second_derivatives() const {
//  return cend_derivatives() + num_pairs(m_num_parameters);
//}
} // namespace jet