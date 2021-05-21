#pragma once

#include <type_traits>

#include "jet.h"

namespace jet {

/*
Provides adaptation of common functions to jets and the algorithms to easily
implement new functions.
*/

template <typename T = float> class Exponential {

public:
  using result_type = T;
  Exponential(){}; // Used as tag for dispatch, does no computation

  Exponential(T x) : m_exp_x{std::exp(x)} {}

  T result() { return m_exp_x; }

  T deriv_wrt_arg_1(T d) { return d * m_exp_x; }

  T sec_deriv_wrt_args_11(T d1, T d2) { return d1 * d2 * m_exp_x; }

private:
  T m_exp_x{};
};

template <typename T = float> class Cosine {

public:
  using result_type = T;
  Cosine(){}; // Used as tag for dispatch, does no computation

  Cosine(T x) : m_cos_x{std::cos(x)}, m_sin_x{std::sin(x)} {}

  T result() { return m_cos_x; }

  T deriv_wrt_arg_1(T d) { return -d * m_sin_x; }

  T sec_deriv_wrt_args_11(T d1, T d2) { return -d1 * d2 * m_cos_x; }

private:
  T m_cos_x{};
  T m_sin_x{};
};

template <typename T = float> class Sine {

public:
  using result_type = T;
  Sine(){}; // Used as tag for dispatch, does no computation

  Sine(T x) : m_cos_x{std::cos(x)}, m_sin_x{std::sin(x)} {}

  T result() { return m_sin_x; }

  T deriv_wrt_arg_1(T d) { return d * m_cos_x; }

  T sec_deriv_wrt_args_11(T d1, T d2) { return -d1 * d2 * m_sin_x; }

private:
  T m_cos_x{};
  T m_sin_x{};
};

template <typename FN, typename InArg>
auto apply_to_jet(FN tag, InArg const &x) {

  auto fx = FN(x.value());
  Jet2<typename FN::result_type, typename InArg::parameter_type> result(
      x.parameters(), fx.result());

  auto const n = x.num_parameters();
  for (size_t i = 0; i < n; ++i)
    result.derivative(i) = fx.deriv_wrt_arg_1(x.derivative(i));

  for (auto el : result.second_derivatives())
    el.value = fx.deriv_wrt_arg_1(x.second_derivative(el.i, el.j)) +
               fx.sec_deriv_wrt_args_11(x.derivative(el.i), x.derivative(el.j));

  return result;
}

template <typename FN, typename InArg1, typename InArg2>
auto apply_to_jet(FN tag, InArg1 const &x1, InArg2 const &x2) {

  auto fx = FN(x1.value(), x2.value());
  auto parameters = x1.parameters(); // Intended copy
  auto indices = union_inplace(parameters, x2.parameters());

  Jet2<typename FN::result_type, typename InArg1::parameter_type> result(
      parameters, fx.result());

  auto const n1 = x1.num_parameters();
  for (size_t i = 0; i < n1; ++i)
    result.derivative(i) = fx.deriv_wrt_arg_1(x1.derivative(i));

  auto const n2 = x2.num_parameters();
  for (size_t i = 0; i < n2; ++i)
    result.derivative(indices[i]) += fx.deriv_wrt_arg_2(x2.derivative(i));

  for (auto el : x1.second_derivatives())
    result.second_derivative(el.i, el.j) =
        fx.deriv_wrt_arg_1(el.value) +
        fx.sec_deriv_wrt_args_11(x1.derivative(el.i), x1.derivative(el.j));

  for (auto el : x2.second_derivatives())
    result.second_derivative(indices[el.i], indices[el.j]) +=
        fx.deriv_wrt_arg_2(el.value) +
        fx.sec_deriv_wrt_args_22(x2.derivative(indices[el.i]),
                                 x2.derivative(indices[el.j]));

  for (size_t i = 0; i < n1; ++i)
    for (size_t j = 0; j < n2; ++j)
      result.second_derivative(i, indices[j]) +=
          fx.sec_deriv_wrt_args_12(x1.derivative(i), x1.derivative(indices[j]));

  return result;
}

} // namespace jet
