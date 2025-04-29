#ifndef __CIRCUITSIM__
#define __CIRCUITSIM__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>


template <typename in_T>
__attribute__((noinline)) v64uint8 extractSwitchState(in_T* external_sw){ //TODO: right now one float, so only support max of 32 switch or diode 
    // interpret float as 4 uint8 variables
    // for now, assume in LSB

    static_assert(std::is_same<in_T, float>::value);

    event0();
    // v16uint8 value_1 0;
    // v16uint8 value_2 = 0;
    // v16uint8 value_3 = 0;
    // v8uint8 value_4 = 0;


    uint8_t *ptr = (uint8_t *) external_sw;

    // value_4 = insert(value_4, 0, *ptr);
    // ptr++;
    // value_4 = insert(value_4, 1, *ptr);
    // prt++;
    // value_4 = insert(value_4, 2, *ptr);
    // ptr++;
    // value_4 = insert(value_4, 3, *ptr);
    // ptr++;


    aie::vector<uint8_t, 64> vec; //TODO: initailzie with zero
    vec.set(*ptr, 0);
    ptr++;

    vec.set(*ptr,1);
    ptr++;

    vec.set(*ptr, 2 );
    ptr++;

    vec.set(*ptr, 3);
    ptr++;


    event1();

    return vec;

}



bool compare_and_copy_bits(uint32_t& dst, uint32_t src, uint32_t pos, uint32_t num_bits) ;


#endif
