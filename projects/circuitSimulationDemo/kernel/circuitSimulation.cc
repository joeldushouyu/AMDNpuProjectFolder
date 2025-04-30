#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// #include "zero.h"

#include "circuitSimulation.h"



//TODO: define external_switch_number and diode_number as macro?

bool compare_and_copy_bits(uint32_t& dst, uint32_t src, uint32_t pos, uint32_t num_bits) {
    // Create mask for the bit range
    uint32_t mask = ((1u << num_bits) - 1) << pos;

    // Extract masked bits from both src and dst
    bool bits_equal = (dst & mask) == (src & mask);

    // Set bits in dst to match src
    dst = (dst & ~mask) | (src & mask);

    return bits_equal;
}

void setBit(uint32_t &num, uint8_t bitIndex, bool value) {
    if (value) {
        num |= (1U << bitIndex);  // Set the bit to 1
    } else {
        num &= ~(1U << bitIndex); // Clear the bit to 0
    }
}
void update_x_cur_and_u_with_new_u(float* x_cur_and_u, float *u_input ){
    #pragma clang loop unroll_count(U_SIZE)
    for( uint32_t i = 0; i < U_SIZE; i++){
        x_cur_and_u[i+STATE_SIZE] = *u_input++;

    }
}
extern "C" {



}



