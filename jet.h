#pragma once

#include <vector>       // for std::vector
#include <type_traits>  // for std::enable_if_t
#include <assert.h>     // for assert

#include "symmetric_matrix_view.h"  // for SymmetricMatrixView

namespace jet {

/*
A class that holds a value of type V, along with derivative and second
derivative data with respect to a list of parameters.

The value type V must be an additive type, the parameter type P serves as a
labelling so only need implement comparison, a common choice will be scalar*
where scalar is a type of scalar, i.e. double* or float*.  The pointers will
only ever be compared and will not be dereferenced.
*/
template <typename V, typename P> class Jet2 {
public:
  using value_type = V;
  using parameter_type = P;

  // Initialise with zero parameters and a default value
  Jet2() = default;

  // Initialise with 0 parameters and a single value
  explicit Jet2(value_type const &val) : m_value{val} {};

  // Initialise with a given list of parameters and default (should be zero)
  // values (pass by value intentional, allows for move or copy constructor)
  explicit Jet2(std::vector<parameter_type> parameters);

  // Initialise with a given list of parameters and default (should be zero)
  // values (pass by value intentional, allows for move or copy constructor)
  Jet2(std::vector<parameter_type> parameters, value_type const &val);

  // Default copy/move/destruct
  ~Jet2() = default;
  Jet2(Jet2 const &) = default;
  Jet2(Jet2 &&) noexcept = default;
  Jet2 &operator=(Jet2 const &) = default;
  Jet2 &operator=(Jet2 &&) = default;

  // Access to parameters
  size_t num_parameters() const;
  parameter_type const &parameter(size_t i) const;
  std::vector<parameter_type> const &parameters() const;

  // Manipulation of parameters
  void push_back_parameter(parameter_type const &p);
  std::vector<size_t> add_parameters(std::vector<parameter_type> const &params);
  void pop_back();
  // void erase_parameter(std::vector<P>::const_iterator where);
  // void erase_parameters(std::vector<P>::const_iterator first,
  // std::vector<P>::const_iterator last);
  void clear_parameters();

  // Access to value
  value_type &value();
  value_type const &value() const;

  // Access to derivatives
  size_t num_derivatives() const;
  value_type &derivative(size_t i);
  value_type const &derivative(size_t i) const;

  // Access to second derivatives
  SymmetricMatrixView<V> const &second_derivatives() const;
  SymmetricMatrixView<V> &second_derivatives();
  V &second_derivative(size_t i, size_t j);
  V const &second_derivative(size_t i, size_t j) const;

  // Access to underlying vector data via pointer
  size_t size_data() const { return m_data.size(); }
  value_type *data() { return m_data.data(); }
  value_type const *data() const { return m_data.data(); }

  Jet2 operator-() const &;
  Jet2 &operator-() &&;

private:
  value_type m_value{};
  std::vector<parameter_type> m_parameters{}; // Constructors rely on this order
  std::vector<value_type> m_data{};

  SymmetricMatrixView<V> m_second_derivatives{};

  static inline size_t data_length(size_t n) {
    return n + details::num_pairs(n);
  }
};


// Appends the elements of rhs which are not contained in lhs to lhs and returns
// the indices of each element of rhs in the resulting lhs
template <typename P>
inline std::vector<size_t>
union_inplace(std::vector<P> &lhs,
              std::vector<P> const &rhs) // lhs and rhs could be the same vector
{
  std::vector<size_t> result{};
  result.reserve(rhs.size());

  // NB if lhs and rhs are aliased the following algorithm still works as p is
  // always found in lhs

  size_t original_size = lhs.size();
  try {
    for (P p : rhs) {
      auto q = std::find(lhs.cbegin(), lhs.cend(), p);
      result.push_back(q - lhs.cbegin());
      if (q == lhs.cend())
        lhs.push_back(p);  // NB this increments result of lhs.cend()
    }
  } catch (...) {
    lhs.resize(original_size);
    throw;
  }
  return result;
}

template <typename J> struct is_2jet { static constexpr bool value = false; };

template <typename V, typename P> struct is_2jet<Jet2<V, P>> {
  static constexpr bool value = true;
};

template <typename T> constexpr bool is_2jet_v = is_2jet<T>::value;

template <typename Jet, std::enable_if_t<is_2jet_v<Jet>, int> = 0>
Jet &add_matched_jets(Jet &out, Jet const &a) {
  size_t n = std::min(out.num_parameters(), a.num_parameters());

  out.value() += a.value();

  for (int i = 0; i < n; ++i)
    out.derivative(i) += a.derivative(i);

  for (auto el : out.second_derivatives())
    el.value += a.second_derivative(el.i, el.j);
  return out;
}

// We do not assume that multiplication is commutative
template <typename Jet, typename Jrhs,
          std::enable_if_t<is_2jet_v<Jet> && is_2jet_v<Jrhs>, int> = 0>
Jet &right_multiply_matched_jets(Jet &out, Jrhs const &a) {
  size_t n = std::min(out.num_parameters(), a.num_parameters());

  // As we are computing the variable, out in-place we need to compute the
  // second, then first derivatives and only then the value
  for (auto const el : out.second_derivatives()) {
    el.value = el.value * a.value() +
               out.derivative(el.i) * a.derivative(el.j) +
               out.derivative(el.j) * a.derivative(el.i) +
               out.value() * a.second_derivative(el.i, el.j);
  }

  for (int i = 0; i < n; ++i)
    out.derivative(i) =
        out.derivative(i) * a.value() + out.value() * a.derivative(i);

  out.value() *= a.value();

  return out;
}

template <typename Jet, std::enable_if_t<is_2jet_v<Jet>, int> = 0>
Jet &operator+=(Jet &out, Jet const &a) {
  auto indices = out.add_parameters(a.parameters());

  auto n = indices.size();

  out.value() += a.value();

  for (int i = 0; i < n; ++i)
    out.derivative(indices[i]) += a.derivative(i);

  for (auto const &it : a.second_derivatives())
    out.second_derivative(indices[it.i], indices[it.j]) += it.value;

  return out;
}

template <typename Jet, typename V,
          std::enable_if_t<is_2jet_v<Jet> && !is_2jet_v<V>, int> = 0>
Jet &operator+=(Jet &out, V const &a) {
  out.value() += a;
  return out;
}

template <typename Jet, std::enable_if_t<is_2jet_v<Jet>, int> = 0>
Jet operator+(Jet out, Jet const &b) // Pass by value deals with rhs lvalue and
                                     // lhs either rvalue or lvalue
{
  out += b;
  return out;
}

template <typename Jet, std::enable_if_t<is_2jet_v<Jet>, int> = 0>
Jet operator+(
    Jet const &a,
    Jet &&out) // Deals with rhs rvalue and lhs either rvalue or lvalue
{
  out += a;
  return std::move(out);
}

template <typename Jet, typename V,
          std::enable_if_t<is_2jet_v<Jet> && !is_2jet_v<V>, int> = 0>
Jet operator+(Jet const &a, V const &b) {
  Jet out = a;
  out += b;
  return out;
}

template <typename V, typename Jet,
          std::enable_if_t<is_2jet_v<Jet> && !is_2jet_v<V>, int> = 0>
Jet operator+(V const &b, Jet const &a) {
  Jet out = a;
  out += b;
  return out;
}

// We do not assume that multiplication is commutative, we only assume that the
// two types can be multiplied
template <typename Jet, typename Jrhs,
          std::enable_if_t<is_2jet_v<Jet> && is_2jet_v<Jrhs>, int> = 0>
Jet &operator*=(Jet &out, Jrhs const &a) {

  size_t const n_out = out.num_parameters();
  size_t const n_a = a.num_parameters();
  auto const indices = out.add_parameters(a.parameters());

  // As we are computing the variable, out in-place we need to compute the
  // second, then first derivatives and only then the value
  for (auto el : out.second_derivatives())
    el.value = el.value * a.value();

  for (auto el : a.second_derivatives())
    out.second_derivative(indices[el.i], indices[el.j]) +=
        out.value() * el.value;

  for (size_t i = 0; i < n_out; ++i)
    for (size_t j = 0; j < n_a; ++j)
      out.second_derivative(i, indices[j]) +=
          ((i == indices[j]) ? 2.0f : 1.0f) * out.derivative(i) *
          a.derivative(j);

  for (int i = 0; i < n_out; ++i)
    out.derivative(indices[i]) = out.derivative(indices[i]) * a.value();

  for (int j = 0; j < n_a; ++j)
    out.derivative(indices[j]) += out.value() * a.derivative(i);

  out.value() *= a.value();

  return out;
}

template <typename Jlhs, typename Jrhs,
          std::enable_if_t<is_2jet_v<Jlhs> && is_2jet_v<Jrhs>, int> = 0>
auto operator*(Jlhs const &lhs, Jrhs const &rhs)
//    -> Jet2<decltype(lhs.value()* rhs.value())>
{
  // NB *= not available as the result may have a different type to both
  // arguments

  auto parameters{lhs.parameters()};
  auto const indices = union_inplace(parameters, rhs.parameters());

  Jet2<decltype(lhs.value() * rhs.value()), typename Jlhs::parameter_type>
      result{parameters, lhs.value() * rhs.value()};

  // Left term of product rule (lr)' = l'r + lr'
  for (int i = 0; i < lhs.num_parameters(); ++i)
    result.derivative(i) = lhs.derivative(i) * rhs.value();

  // Right term of product rule
  for (int i = 0; i < rhs.num_parameters(); ++i)
    result.derivative(indices[i]) += lhs.value() * rhs.derivative(i);

  // Left term of double product rule (lr)'' = l''r + 2l'r' + lr''
  for (auto el : lhs.second_derivatives())
    result.second_derivative(el.i, el.j) = el.value * rhs.value();

  // Right term of double product rule
  for (auto el : rhs.second_derivatives())
    result.second_derivative(indices[el.i], indices[el.j]) +=
        lhs.value() * el.value;

  // Middle term of double product rule
  for (int i = 0; i < lhs.num_parameters(); ++i)
    for (int j = 0; j < rhs.num_parameters(); ++j)
      result.second_derivatives()(i, indices[j]) +=
          ((i == indices[j]) ? 2.0f : 1.0f) * lhs.derivative(i) *
          rhs.derivative(indices[j]);

  return result;
}

template <typename J, typename V,
          std::enable_if_t<(is_2jet_v<J>)&&!(is_2jet_v<V>), int> = 0>
auto operator*(J const &lhs, V const &rhs)
//    -> Jet2<decltype(lhs.value()* rhs)>
{
  Jet2<decltype(lhs.value() * rhs), typename J::parameter_type> result{
      lhs.parameters(), lhs.value() * rhs};

  for (int i = 0; i < lhs.num_parameters(); ++i)
    result.derivative(i) = lhs.derivative(i) * rhs;

  for (auto el : result.second_derivatives())
    el.value = lhs.second_derivative(el.i, el.j) * rhs;

  return result;
}

template <typename V, typename J,
          std::enable_if_t<is_2jet_v<J> && !is_2jet_v<V>, int> = 0>
auto operator*(V const &lhs, J const &rhs)
//    -> Jet2<decltype(lhs * rhs.value())>
{
  Jet2<decltype(lhs * rhs.value()), typename J::parameter_type> result{
      rhs.parameters(), lhs * rhs.value()};

  for (int i = 0; i < rhs.num_parameters(); ++i)
    result.derivative(i) = lhs * rhs.derivative(i);

  for (auto el : result.second_derivatives())
    el.value = lhs * rhs.second_derivative(el.i, el.j);

  return result;
}

template <typename F, typename Jlhs, typename Jrhs,
          std::enable_if_t<is_2jet_v<Jlhs> && is_2jet_v<Jrhs>, int> = 0>
auto bilinear_function(Jlhs const &lhs, Jrhs const &rhs) {
  using result_value_t =
      typename std::remove_const<typename std::remove_reference<decltype(
          F::call(lhs.value(), rhs.value()))>::type>::type;
  using parameter_t = typename Jlhs::parameter_type;

  auto parameters{lhs.parameters()};
  auto const indices = union_inplace(parameters, rhs.parameters());

  Jet2<result_value_t, parameter_t> result{parameters,
                                           F::call(lhs.value(), rhs.value())};

  // Left term of rule F(l,r)' = F(l',r) + F(l,r')
  for (int i = 0; i < lhs.num_parameters(); ++i)
    result.derivative(i) = F::call(lhs.derivative(i), rhs.value());

  // Right term of rule
  for (int i = 0; i < rhs.num_parameters(); ++i)
    result.derivative(indices[i]) += F::call(lhs.value(), rhs.derivative(i));

  // Left term of double rule F(l,r)'' = F(l'',r) + 2F(l',r') + F(l,r'')
  for (auto const el : lhs.second_derivatives())
    result.second_derivative(el.i, el.j) = F::call(el.value, rhs.value());

  // Right term of double rule
  for (auto const el : rhs.second_derivatives())
    result.second_derivative(indices[el.i], indices[el.j]) +=
        F::call(lhs.value(), el.value);

  // Middle term of double rule
  for (int i = 0; i < lhs.num_parameters(); ++i)
    for (int j = 0; j < rhs.num_parameters(); ++j)
      result.second_derivatives()(i, indices[j]) +=
          2.0f * F::call(lhs.derivative(i), rhs.derivative(indices[j]));

  return result;
}

template <typename F, typename J, typename V,
          std::enable_if_t<is_2jet_v<J> && !is_2jet_v<V>, int> = 0>
auto bilinear_function(J const &lhs, V const &rhs) {
  using result_value_t = std::remove_const_t<
      std::remove_reference_t<decltype(F::call(lhs.value(), rhs))>>;
  using parameter_t = typename J::parameter_type;

  Jet2<result_value_t, parameter_t> result{lhs.parameters(),
                                           F::call(lhs.value(), rhs)};

  for (int i = 0; i < lhs.num_parameters(); ++i)
    result.derivative(i) = F::call(lhs.derivative(i), rhs);

  for (int j = 0; j < lhs.num_parameters(); ++j)
    for (int i = 0; i <= j; ++i)
      result.second_derivatives()(i, j) =
          F::call(lhs.second_derivatives()(i, j), rhs);

  return result;
}

template <typename F, typename V, typename J,
          std::enable_if_t<is_2jet_v<J> && !is_2jet_v<V>, int> = 0>
auto bilinear_function(V const &lhs, J const &rhs) {
  using result_value_t = std::remove_const_t<
      std::remove_reference_t<decltype(F::call(lhs, rhs.value()))>>;
  using parameter_t = typename J::parameter_type;

  Jet2<result_value_t, parameter_t> result{rhs.parameters(),
                                           F::call(lhs, rhs.value())};

  for (int i = 0; i < rhs.num_parameters(); ++i)
    result.derivative(i) = F::call(lhs, rhs.derivative(i));

  for (int j = 0; j < rhs.num_parameters(); ++j)
    for (int i = 0; i <= j; ++i)
      result.second_derivatives()(i, j) =
          F::call(lhs, rhs.second_derivatives()(i, j));

  return result;
}

template <typename F, typename J>
auto linear_function(J const &jet)
//-> Jet2<typename std::remove_const<typename
// std::remove_reference<decltype(linear_fn(jet.value()))>::type>::type>
{
  using result_value_t =
      typename std::remove_const<typename std::remove_reference<decltype(
          F::call(jet.value()))>::type>::type;
  using parameter_t = typename J::parameter_type;

  Jet2<result_value_t, parameter_t> result{jet.parameters(),
                                           F::call(jet.value())};

  for (int i = 0; i < jet.num_parameters(); ++i)
    result.derivative(i) = F::call(jet.derivative(i));

  for (int j = 0; j < jet.num_parameters(); ++j)
    for (int i = 0; i <= j; ++i)
      result.second_derivatives()(i, j) =
          F::call(jet.second_derivatives()(i, j));

  return result;
}

template <typename J> J reciprocal(J const &jet) {
  J result{J.parameters(), 1 / J.value()};

  auto inv_sq = result.value() * result.value();
  auto inv_cube = result.value() * inv_sq;

  for (int i = 0; i < jet.num_parameters(); ++i)
    result.derivative(i) = -1 * jet.derivative(i) * inv_sq;

  for (int j = 0; j < jet.num_parameters(); ++j)
    for (int i = 0; i <= j; ++i)
      result.second_derivatives()(i, j) =
          (2 * jet.derivative(i) * jet.derivative(j) -
           jet.value() * jet.second_derivatives()(i, j)) *
          inv_cube;

  return result;
}

template <typename F, typename J> J inverse_function(J const &jet) {
  auto inv = F::call(jet.value());
  J result{jet.parameters(), inv};

  for (int i = 0; i < jet.num_parameters(); ++i)
    result.derivative(i) = -inv * jet.derivative(i) * inv;

  for (int j = 0; j < jet.num_parameters(); ++j)
    for (int i = 0; i <= j; ++i)
      result.second_derivatives()(i, j) =
          inv * jet.derivative(i) * inv * jet.derivative(j) * inv +
          inv * jet.derivative(j) * inv * jet.derivative(i) * inv -
          inv * jet.second_derivatives()(i, j) * inv;

  return result;
}

template <typename V, typename P>
inline Jet2<V, P>::Jet2(std::vector<P> parameters)
    : m_parameters{parameters}, m_data(data_length(m_parameters.size())),
      m_second_derivatives{m_data.data() + m_parameters.size(),
                           m_parameters.size()} {}

template <typename V, typename P>
inline Jet2<V, P>::Jet2(std::vector<P> parameters, V const &val)
    : m_value{val}, m_parameters{parameters},
      m_data(data_length(m_parameters.size())),
      m_second_derivatives{m_data.data() + m_parameters.size(),
                           m_parameters.size()} {}

// Access to parameters
template <typename V, typename P>
inline size_t Jet2<V, P>::num_parameters() const {
  return m_parameters.size();
}

template <typename V, typename P>
inline P const &Jet2<V, P>::parameter(size_t i) const {
  return m_parameters[i];
}

template <typename V, typename P>
inline std::vector<P> const &Jet2<V, P>::parameters() const {
  return m_parameters;
}

template <typename V, typename P>
inline std::vector<size_t>
Jet2<V, P>::add_parameters(std::vector<parameter_type> const &params) {
  size_t original_size = num_parameters();
  std::vector<size_t> indices = union_inplace(m_parameters, params);

  try {
    m_data.reserve(data_length(num_parameters()));
    size_t new_slots = num_parameters() - original_size;
    m_data.insert(m_data.begin() + original_size, new_slots,
                  V{}); // Insert zero derivatives for new parameters
    m_data.resize(
        data_length(num_parameters())); // Add new second derivatives at end
                                        // of data, initialised as zero
    m_second_derivatives =
        SymmetricMatrixView(m_data.data() + num_parameters(), num_parameters());
  } catch (...) {
    m_parameters.resize(original_size);
    m_data.resize(data_length(original_size));
    m_second_derivatives =
        SymmetricMatrixView(m_data.data() + original_size, original_size);
    throw;
  }

  return indices;
}

template <typename V, typename P> inline void Jet2<V, P>::pop_back() {
  m_parameters.pop_back();
  m_data.resize(data_length(num_parameters()) +
                1); // Will remove the second derivatives corresponding to the
                    // final parameter
  m_data.erase(cend_derivatives()); // Remove the extra derivative
  m_second_derivatives =
      SymmetricMatrixView(m_data + num_parameters(), num_parameters());
}

template <typename V, typename P> inline void Jet2<V, P>::clear_parameters() {
  m_parameters.clear();
  m_data.clear();
  m_second_derivatives = SymmetricMatrixView{};
}

template <typename V, typename P>
inline typename Jet2<V, P>::value_type &Jet2<V, P>::value() {
  return m_value;
}

template <typename V, typename P>
inline typename Jet2<V, P>::value_type const &Jet2<V, P>::value() const {
  return m_value;
}

template <typename V, typename P>
inline size_t Jet2<V, P>::num_derivatives() const {
  return num_parameters();
}

template <typename V, typename P>
inline typename Jet2<V, P>::value_type &Jet2<V, P>::derivative(size_t i) {
  assert(i < num_derivatives());
  return m_data[i];
}

template <typename V, typename P>
inline typename Jet2<V, P>::value_type const &
Jet2<V, P>::derivative(size_t i) const {
  assert(i < num_derivatives());
  return m_data[i];
}

template <typename V, typename P>
inline typename Jet2<V, P>::value_type &
Jet2<V, P>::second_derivative(size_t i, size_t j) {
  assert(i < num_parameters());
  assert(j < num_parameters());
  return m_second_derivatives(i, j);
}

template <typename V, typename P>
inline SymmetricMatrixView<V> &Jet2<V, P>::second_derivatives() {
  return m_second_derivatives;
}

template <typename V, typename P>
inline SymmetricMatrixView<V> const &Jet2<V, P>::second_derivatives() const {
  return m_second_derivatives;
}

template <typename V, typename P>
inline typename Jet2<V, P>::value_type const &
Jet2<V, P>::second_derivative(size_t i, size_t j) const {
  assert(i < num_parameters());
  assert(j < num_parameters());
  return m_second_derivatives(i, j);
}

template <typename V, typename P>
inline Jet2<V, P> Jet2<V, P>::operator-() const & {
  Jet2 result = *this;
  return -result; // Compute using rvalue unary operator
}

template <typename V, typename P>
inline Jet2<V, P> &Jet2<V, P>::operator-() && {
  m_value *= -1;
  for (auto &v : m_data)
    v *= -1;
  return *this;
}

// Manipulation of parameters
template <typename V, typename P>
inline void Jet2<V, P>::push_back_parameter(parameter_type const &p) {
  m_parameters.push_back(p); // after this push_back the begin/end iterators
                             // will be placed for the new size
  m_data.reserve(data_length(num_parameters()));
  m_data.insert(cend_derivatives() - 1,
                V{}); // Add first derivative (zero valued), moving old second
                      // derivatives back
  m_data.resize(
      data_length(num_parameters())); // Add slots for second derivatives
                                      // which should be at end of data
  m_second_derivatives =
      SymmetricMatrixView(m_data + num_parameters(), num_parameters());
}

} // namespace jet