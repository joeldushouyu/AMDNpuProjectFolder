#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// #include "zero.h"
#include <aie_api/aie.hpp>
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


extern "C" {

    v64uint8 extratSwitchState_fromfloat(float *external_sw){
        return extractSwitchState<float>(external_sw);
    }

}



