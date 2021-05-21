#pragma once


#include "glm/glm.hpp"
#include "jet.h"

namespace jet {

/*
The purpose of this file is to adapt glm's functions to act on Jet2's of
glm's types.  The aim is for nearly all of the algorithmic content to be
provided by jet's core functionality, this file just provides the glue.
*/

struct glm_dot {
  static inline float call(glm::vec4 const &a, glm::vec4 const &b) {
    return glm::dot(a, b);
  }
  static inline float call(glm::vec3 const &a, glm::vec3 const &b) {
    return glm::dot(a, b);
  }
  static inline float call(glm::vec2 const &a, glm::vec2 const &b) {
    return glm::dot(a, b);
  }
};

struct glm_inverse {
  static inline glm::mat4 call(glm::mat4 const &A) { return glm::inverse(A); }
  static inline glm::mat3 call(glm::mat3 const &A) { return glm::inverse(A); }
  static inline glm::mat2 call(glm::mat2 const &A) { return glm::inverse(A); }
};

struct glm_cross {
  static inline glm::vec3 call(glm::vec3 const &v, glm::vec3 const &w) {
    return glm::cross(v, w);
  }
};

struct glm_wedge {
  static inline glm::mat2x3 call(glm::vec4 const &v, glm::vec4 const &w) {
    return glm::mat2x3{v[0] * w[1] - v[1] * w[0], v[0] * w[2] - v[2] * w[0],
                       v[0] * w[3] - v[3] * w[0], v[1] * w[2] - v[2] * w[1],
                       v[1] * w[3] - v[3] * w[1], v[2] * w[3] - v[3] * w[2]};
  }
  static inline float call(glm::mat2x3 const &A, glm::mat2x3 const &B) {
    return (A[0][0] * B[1][2] + B[0][0] * A[1][2]) -
           (A[0][1] * B[1][1] + B[0][1] * A[1][1]) +
           (A[0][2] * B[1][0] + B[0][2] * A[1][0]);
  }
};

// Functor wrapper for operator[](K)
template <typename T, int K> struct choose_coord {
  static inline auto call(T const &value) /* -> decltype(value[K])*/ {
    return value[K];
  }
};

inline Jet2<float, float *> dot(Jet2<glm::vec2, float *> const &v,
                                Jet2<glm::vec2, float *> const &w) {
  return bilinear_function<glm_dot>(v, w);
}
inline Jet2<float, float *> dot(Jet2<glm::vec2, float *> const &v,
                                glm::vec2 const &w) {
  return bilinear_function<glm_dot>(v, w);
}
inline Jet2<float, float *> dot(glm::vec2 const &v,
                                Jet2<glm::vec2, float *> const &w) {
  return dot(w, v);
}

inline Jet2<float, float *> dot(Jet2<glm::vec3, float *> const &v,
                                Jet2<glm::vec3, float *> const &w) {
  return bilinear_function<glm_dot>(v, w);
}
inline Jet2<float, float *> dot(Jet2<glm::vec3, float *> const &v,
                                glm::vec3 const &w) {
  return bilinear_function<glm_dot>(v, w);
}
inline Jet2<float, float *> dot(glm::vec3 const &v,
                                Jet2<glm::vec3, float *> const &w) {
  return dot(w, v);
}

inline Jet2<float, float *> dot(Jet2<glm::vec4, float *> const &v,
                                Jet2<glm::vec4, float *> const &w) {
  return bilinear_function<glm_dot>(v, w);
}
inline Jet2<float, float *> dot(Jet2<glm::vec4, float *> const &v,
                                glm::vec4 const &w) {
  return bilinear_function<glm_dot>(v, w);
}
inline Jet2<float, float *> dot(glm::vec4 const &v,
                                Jet2<glm::vec4, float *> const &w) {
  return dot(w, v);
}

inline Jet2<glm::vec3, float *> cross(Jet2<glm::vec3, float *> const &v,
                                      Jet2<glm::vec3, float *> const &w) {
  return bilinear_function<glm_cross>(v, w);
}
inline Jet2<glm::vec3, float *> cross(Jet2<glm::vec3, float *> const &v,
                                      glm::vec3 const &w) {
  return bilinear_function<glm_cross>(v, w);
}
inline Jet2<glm::vec3, float *> cross(glm::vec3 const &v,
                                      Jet2<glm::vec3, float *> const &w) {
  return -cross(w, v);
}

Jet2<float, float *> det(Jet2<glm::mat4, float *> const &M) {
  auto col0 = linear_function<choose_coord<glm::mat4, 0>>(M);
  auto col1 = linear_function<choose_coord<glm::mat4, 1>>(M);
  auto col2 = linear_function<choose_coord<glm::mat4, 2>>(M);
  auto col3 = linear_function<choose_coord<glm::mat4, 3>>(M);
  auto A = bilinear_function<glm_wedge>(col0, col1);
  auto B = bilinear_function<glm_wedge>(col2, col3);
  return bilinear_function<glm_wedge>(A, B);
}

Jet2<float, float *> det(Jet2<glm::mat3, float *> const &M) {
  auto col0 = linear_function<choose_coord<glm::mat3, 0>>(M);
  auto col1 = linear_function<choose_coord<glm::mat3, 1>>(M);
  auto col2 = linear_function<choose_coord<glm::mat3, 2>>(M);
  return dot(cross(col0, col1), col2);
}

inline Jet2<glm::mat4, float *> inverse(Jet2<glm::mat4, float *> const &A) {
  return inverse_function<glm_inverse>(A);
}

inline Jet2<glm::mat3, float *> inverse(Jet2<glm::mat3, float *> const &A) {
  return inverse_function<glm_inverse>(A);
}

inline Jet2<glm::mat2, float *> inverse(Jet2<glm::mat2, float *> const &A) {
  return inverse_function<glm_inverse>(A);
}

} // namespace jet