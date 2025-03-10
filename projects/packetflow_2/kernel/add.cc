#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "add.h"
#include "mul.h"

extern "C" {

#ifndef DIM_S
#define DIM_S 256
#endif

void add(
        int8 *restrict In0,
        int8 *restrict y
) {
  add_aie<int8, DIM_S>(In0, y);
}

void mul(
        int8 *restrict In0,
        int8 *restrict y
) {
  mul_aie<int8, DIM_S>(In0, y);
}
} // extern "C"
