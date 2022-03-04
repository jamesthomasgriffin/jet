#pragma once

namespace jet {

namespace details {

struct addition {
  template <typename InIter, typename OutIter>
  static inline void
  call_map_iterators(InIter const &lhs_begin, InIter const &lhs_end,
                     InIter const &rhs_begin, InIter const &rhs_end,
                     OutIter const &out_begin, OutIter const &out_end) {
    InIter lhs_it{lhs_begin};
    InIter rhs_it{rhs_begin};

    for (auto out_it = out_begin; out_it != out_end; ++out_it) {

      while (lhs_it != lhs_end && lhs_it->first < out_it->first)
        ++lhs_it;
      while (rhs_it != rhs_end && rhs_it->first < out_it->first)
        ++rhs_it;

      if (lhs_it->first == out_it->first)
        *(out_it->second) += *(lhs_it->second);
      if (rhs_it->first == out_it->first)
        *(out_it->second) += *(rhs_it->second);
    }
  }

  template <typename Jet>
  static inline void call_derivatives(J const &lhs, J const &rhs, J &out) {
    auto const &lhs_der = lhs.derivatives();
    auto const &rhs_der = rhs.derivatives();
    auto &out_der = out.derivatives();
    call_map_iterators(lhs_der.begin(), lhs_der.end(), rhs_der.begin(),
                       rhs_der.end(), out_der.begin(), out_der.end());
  }

  template <typename Jet>
  static inline void call_second_derivatives(J const &lhs, J const &rhs,
                                             J &out) {
    auto const &lhs_der = lhs.second_derivatives();
    auto const &rhs_der = rhs.second_derivatives();
    auto &out_der = out.second_derivatives();
    call_map_iterators(lhs_der.begin(), lhs_der.end(), rhs_der.begin(),
                       rhs_der.end(), out_der.begin(), out_der.end());
  }

  template <typename Jet>
  static inline void call_1jets(J const &lhs, J const &rhs) {
    J result{lhs, rhs};

    result.value() = lhs.value() + rhs.value();
    call_derivatives<Jet>(lhs, rhs, result);

    return result;
  }

  template <typename Jet>
  static inline void call_2jets(J const &lhs, J const &rhs) {
    J result{lhs, rhs};

    result.value() = lhs.value() + rhs.value();
    call_derivatives<Jet>(lhs, rhs, result);
    call_second_derivatives<Jet>(lhs, rhs, result);

    return result;
  }
};

} // namespace details

} // namespace jet