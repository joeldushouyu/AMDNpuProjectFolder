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
  const int32_t in_offset,  const int32_t out_offset,
  const int32_t iteration_size, const int32_t input_size, const int32_t output_size
){
  
  in += in_offset;
  out+= out_offset;
  // assert divide out without any remainder
  int32_t input_per_iteration_size = input_size / iteration_size;
  int32_t output_per_iteration_size = output_size / iteration_size;
  event0();


  for (int32_t i = 0; i < iteration_size; i++){
    float acc= 0;
    for(int32_t k = 0; k < input_per_iteration_size; k++){
      // acc += *in;
      if(k+1 == input_per_iteration_size){
        // if(*debug_input == 100){
          acc += *in;
        // }else{
        //   acc +=0;
        // }
      }else{
        acc += *in;
      }

      // if(k +1 == input_per_iteration_size){

      //   // This means the float value is stored in Little Endian mode
      //   uint8_t *pt = (uint8_t *)in;
      //   uint8_t v1 = *pt++;
      //   uint8_t v2 = *pt++;
      //   uint8_t v3 = *pt++;
      //   uint8_t v4 = *pt++;
        
      //   // Now instead of memcpy:
      //   uint32_t bits = 
      //   ((uint32_t)v1) |
      //   ((uint32_t)v2 << 8) |
      //   ((uint32_t)v3 << 16) |
      //   ((uint32_t)v4 << 24);
      //   float f = *(float *)&bits;
      //   acc += f;
        
      // }else{

      //   acc += *in;
      // }
      in++;
    }
    for(int32_t l = 0; l < output_per_iteration_size; l++){
      *out = acc;
      out++;
    }

  }
  event1();
}



void accum_float_value(float *in, float *out, 
    const int32_t in_offset,  const int32_t out_offset,
    const int32_t iteration_size, const int32_t input_size, const int32_t output_size
    // uint32_t *debug_input
  ){
    accumValue<float>(in,out, in_offset, out_offset,
       iteration_size, input_size,output_size);
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
            0, 0,
            iteration_ste_per_buffer, buffer_size_of_in,buffer_size_of_out
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
              buffer_size_of_in, buffer_size_of_out,
            iteration_ste_per_buffer, buffer_size_of_in,buffer_size_of_out
            );

            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);




        }
    }
} // extern "C"
