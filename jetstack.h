#pragma once

#include <assert.h>
#include <vector>

#include "jetview.h"

namespace jet {

// Shared storage for a stack of Jet2's of the same type but potentially
// differing numbers of parameters, will probably become Jet2Vector when
// functionality is added
template <typename V, typename P> class Jet2Stack {
public:
  using value_type = V;
  using parameter_type = P;

  void push_back(Jet2<value_type, parameter_type> const &J);
  void pop_back();

  Jet2View<value_type, parameter_type> operator[](size_t i);
  Jet2View<value_type const, parameter_type const> operator[](size_t i) const;

  size_t size() const;

private:
  std::vector<value_type> m_data{};
  std::vector<parameter_type> m_parameters{};
  struct entry_indices_t {
    size_t data_ix, param_ix;
  };
  std::vector<entry_indices_t> m_indices{};

  size_t get_num_parameters_of_entry(size_t i) const;
};

template <typename V, typename P>
inline size_t Jet2Stack<V, P>::get_num_parameters_of_entry(size_t i) const {
  assert(i < size());
  return ((i < size() - 1) ? m_indices[i + 1].param_ix : m_parameters.size()) -
         m_indices[i].param_ix;
}

template <typename V, typename P>
inline Jet2View<V, P> Jet2Stack<V, P>::operator[](size_t i) {
  assert(i < size());
  return Jet2View<value_type, parameter_type>{
      m_parameters.data() + m_indices[i].param_ix,
      m_data.data() + m_indices[i].data_ix,
      m_data.data() + m_indices[i].data_ix + 1, get_num_parameters_of_entry(i)};
}

template <typename V, typename P>
inline Jet2View<V const, P const> Jet2Stack<V, P>::operator[](size_t i) const {
  assert(i < size());
  return Jet2View<value_type const, parameter_type const>{
      m_parameters.data() + m_indices[i].param_ix,
      m_data.data() + m_indices[i].data_ix,
      m_data.data() + m_indices[i].data_ix + 1, get_num_parameters_of_entry(i)};
}

template <typename V, typename P> inline size_t Jet2Stack<V, P>::size() const {
  return m_indices.size();
}

template <typename V, typename P> inline void Jet2Stack<V, P>::pop_back() {
  auto const indices = m_indices.back();
  m_data.resize(indices.data_ix);
  m_parameters.resize(indices.param_ix);
  m_indices.pop_back();
}

template <typename V, typename P>
inline void Jet2Stack<V, P>::push_back(Jet2<V, P> const &J) {
  entry_indices_t indices{m_data.size(), m_parameters.size()};
  try {
    m_data.push_back(J.value());
    m_data.insert(std::end(m_data), J.data(), J.data() + J.size_data());
    m_parameters.insert(std::end(m_parameters), std::begin(J.parameters()),
                        std::end(J.parameters()));
    m_indices.push_back(indices);
  } catch (...) {
    m_data.resize(indices.data_ix);
    m_parameters.resize(indices.param_ix);
    throw;
  }
}

} // namespace jet