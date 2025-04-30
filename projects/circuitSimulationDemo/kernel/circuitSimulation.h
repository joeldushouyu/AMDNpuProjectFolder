#ifndef __CIRCUITSIM__
#define __CIRCUITSIM__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include "circuitConfig.h"


inline constexpr uint32_t round_up_to_any(uint32_t x, uint32_t align) {
    return (x + align - 1) / align * align;
}


template<uint32_t Align, uint32_t X>
consteval uint32_t round_up_to_pow2_at_compile() {
    static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of two");
    return (X + Align - 1) & ~(Align - 1);
}
template<uint32_t Align>
inline constexpr uint32_t round_up_to_pow2(uint32_t x) {
    static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of two");
    return (x + Align - 1) & ~(Align - 1);
}

inline constexpr int div_round_up(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

bool compare_and_copy_bits(uint32_t& dst, uint32_t src, uint32_t pos, uint32_t num_bits) ;
void setBit(uint32_t &num, uint8_t bitIndex, bool value);
void update_x_cur_and_u_with_new_u(float* x_cur_and_u, float *u_input );

#endif
