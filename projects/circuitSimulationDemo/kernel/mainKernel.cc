//===- passThrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdlib.h>
#include "common_macro.h"
#include <aie_api/aie.hpp>

#include "circuitConfig.h"
#include "circuitSimulation.h"


float* retrieveMatrixOFfsetBaseOnState(const uint32_t state, const int32_t matrix_size, float* matrix_ptr) {

    return  matrix_ptr + (state * matrix_size);
}



template<typename T>
__attribute__((noinline)) void accumValue(float *restrict in, float*restrict out,
  const int32_t in_offset,  const int32_t out_offset
){
  
  in += in_offset;
  out+= out_offset;
  // assert divide out without any remainder
  int32_t input_per_iteration_size = INPUT_SIZE_PER_ITERATION;
  int32_t output_per_iteration_size = OUTPUT_SIZE_PER_ITERATION;
  for (int32_t i = 0; i < ITERATION_STEP_PER_PING_PONG_BUFFER * PING_PONG_BUFFER_ITERATION; i++){
    float acc= 0;
    for(int32_t k = 0; k < input_per_iteration_size; k++){
      acc += *in;
      in++;
    }
    for(int32_t l = 0; l < output_per_iteration_size; l++){
      *out = acc;
      out++;
    }

  }

}



void accum_float_value(float *in, float *out, 
    const int32_t in_offset,  const int32_t out_offset
    // uint32_t *debug_input
  ){
    event0();
    accumValue<float>(in,out, in_offset, out_offset);
    event1();
  }

extern "C" {
    void CT_main(float* in, float* out,
        const int32_t buffer_size_of_in, const int32_t buffer_size_of_out,
        const int32_t iteration_ste_per_buffer,
        const int32_t buffer_in_prod_lock_id, const int32_t buffer_in_con_loc_id,
        const int32_t buffer_out_prod_lock_id, const int32_t buffer_out_con_lock_id,

        float* C1_DSW_Buffer
    ) {

        const int32_t C1_DSW_mat_size = C1_DSW_MATRIX_SIZE;
        const uint32_t externalSwitchDiodeStates = 0x0;

        float_t x_cur_and_u[ROUND_UP_TO_16(STATE_SIZE + U_SIZE)] =  {0};
        

        for (uint64_t l = 0; l < MAX_LOOP_SIZE; l++) {
            acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
            acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);
            // use buffer 0 of ping in and out
            accum_float_value(in, out, 
            0, 0
            );


            // float* C1_DSW_ptr = C1_DSW_Buffer;
            // uint32_t offset_value = *(uint32_t *)( buffer_size_of_in);

            // uint32_t* in_int32_t = (uint32_t*)(in);
            // for (uint32_t i = 0; i < 16; i++) {
            //     // float* C1_DSW_ptr = C1_DSW_Buffer + i*C1_DSW_mat_size;

            //     float* C1_DSW_ptr = retrieveMatrixOFfsetBaseOnState(*in_int32_t, C1_DSW_mat_size, C1_DSW_Buffer);
            //     for (int k = 0; k < C1_DSW_mat_size; k++) {
            //         *out++ = *C1_DSW_ptr++;
            //     }
            //     in_int32_t++;

            // }
            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);

            acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
            acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);
            // use buffer 0 of ping in and out
            accum_float_value(in, out, 
              buffer_size_of_in, buffer_size_of_out
            );

            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);




        }
    }
} // extern "C"
