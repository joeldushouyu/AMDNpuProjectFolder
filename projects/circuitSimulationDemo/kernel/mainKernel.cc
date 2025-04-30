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


  
float* retrieveMatrixOffsetBaseOnState(const uint32_t state, const int32_t matrix_size, float* matrix_ptr) {

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
  


template< typename T_in, typename T_acc,
    uint32_t X_CUR_U_SIZE, uint32_t C1_DSW_COL_DIV_16_CEIL, uint32_t C1_DSW_ROW_DIV_16_CEIL>
void  C1_DSW_mult_x_cur_with_u(T_in* C1_DSW_ptr, T_in* x_cur_and_u, aie::accum<T_acc, C1_DSW_COL_DIV_16_CEIL >& res_Acc) {


    static_assert(std::is_same<T_acc, accfloat >::value);
    static_assert(std::is_same<T_in, float >::value);
    // option 1: use scalar
    // option 2: use vector operatin
        // option 2.1 use mask ?

        // recall C1_DSW is column majored order, and we can only do 16x16 elementwise multiplication


        // for now, assume C1_DSW_COL(determine the accumulator size) <= 16
    static_assert(C1_DSW_ROW_DIV_16_CEIL == 16, "C1_DSW_ROW_DIV_16_CEIL size > 16");

    for (uint32_t i = 0; i < C1_DSW_COL_DIV_16_CEIL; i++) {

        aie::vector<T_in, 16> b_vec = aie::load_v<16>(x_cur_and_u);

        // #pragma clang loop unroll_count(X_CUR_U_SIZE) // sometimes the x_cur_and_u size is less than 16  
        for (unsigned int l = 0; l < X_CUR_U_SIZE; l++) {
            aie::vector<T_in, 16> a_col = aie::load_v<16>(C1_DSW_ptr);
            C1_DSW_ptr += C1_DSW_ROW_SIZE;

            aie::vector<T_in, 16> b_l = aie::broadcast<T_in, 16>(b_vec[l]);

            res_Acc = mac_elem_16_accuracy_safe(a_col, b_l, res_Acc, 0, 0, 0);
        }


        //TODO:  Last Loop(less than 16, or less than certain ) is less than 16, so use scalar instead?
    }



}

extern "C" {
    void CT_main(float* in, float* out,
        const int32_t buffer_size_of_in, const int32_t buffer_size_of_out,


        const int32_t buffer_in_prod_lock_id, const int32_t buffer_in_con_loc_id,
        const int32_t buffer_out_prod_lock_id, const int32_t buffer_out_con_lock_id,

        float* C1_DSW_Buffer,

        float* C1_DSW_mat_res, float* A_B_mat_Res, float* X_U_cur, float* C_D_mat_res
    ) {

        const int32_t C1_DSW_mat_size = C1_DSW_MATRIX_SIZE;
        const uint32_t externalSwitchDiodeStates = 0x0;
        const uint32_t X_U_SIZE = STATE_SIZE + U_SIZE;
        
        // first step is to initalize those values

        static_assert(BUFFER_SIZE_OF_C1_DSW_MAT_RES == 4*3*DIODE_SIZE);
        static_assert(BUFFER_SIZE_OF_X_U_CUR == 4*(U_SIZE + STATE_SIZE));
        
        for(uint32_t i = 0; i < U_SIZE+STATE_SIZE; i++){
            X_U_cur[i] = 10.0;
        }
        const uint32_t C1_DSW_COL_DIV_16_CEIL = round_up_to_pow2_at_compile<16, C1_DSW_COL_SIZE>();
        const uint32_t C1_DSW_ROW_DIV_16_CEIL = round_up_to_pow2_at_compile<16, C1_DSW_ROW_SIZE>();
        for (uint64_t l = 0; l < MAX_LOOP_SIZE; l++) {
            acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
            acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);
            // use buffer 0 of ping in and out
            accum_float_value(in, out, 
            0, 0,
            ITERATION_STEP_PER_PING_PONG_BUFFER, buffer_size_of_in,buffer_size_of_out
            );


            // float* C1_DSW_ptr = C1_DSW_Buffer;
            // uint32_t offset_value = *(uint32_t *)( buffer_size_of_in);

            // uint32_t* in_int32_t = (uint32_t*)(in);
            // for (uint32_t i = 0; i < 16; i++) {
            //     // float* C1_DSW_ptr = C1_DSW_Buffer + i*C1_DSW_mat_size;

            //     float* C1_DSW_ptr = retrieveMatrixOffsetBaseOnState(*in_int32_t, C1_DSW_mat_size, C1_DSW_Buffer);
            //     for (int k = 0; k < C1_DSW_mat_size; k++) {
            //         *out++ = *C1_DSW_ptr++;
            //     }
            //     in_int32_t++;

            // }

            // only look at top 16 input value and then do a matrix multiplication



            // for (uint32_t i = 0; i < 16; i++) {
            //     // float* C1_DSW_ptr = retrieveMatrixOffsetBaseOnState(i, C1_DSW_MATRIX_SIZE, C1_DSW_Buffer);

            //     // aie::accum<accfloat, 16> resACC = aie::zeros<accfloat, 16>();
            //     event0();
            //     aie::vector<float, 16> b_vec = aie::load_v<16>(X_U_cur);
            //     aie::vector<float, 16> b2_vec = aie::load_v<16>(X_U_cur+2);
            //     event0();   
            //     event0();
            //     b_vec.store(out);
            //     b_vec.store(out+2);
            //     event0();

            //     // C1_DSW_mult_x_cur_with_u<float, accfloat, X_U_SIZE, C1_DSW_COL_DIV_16_CEIL, C1_DSW_ROW_DIV_16_CEIL >(C1_DSW_ptr, X_U_cur,
            //     //     resACC);

            //     // // write the result out for now(the result should be of size 3*DIODE_SIZE)
            //     // aie::store_v(out, resACC.template to_vector<float>());;
            //     // out += 3*DIODE_SIZE;
            // }


            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);

            acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
            acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);
            // use buffer 0 of ping in and out
            accum_float_value(in, out, 
              buffer_size_of_in, buffer_size_of_out,
              ITERATION_STEP_PER_PING_PONG_BUFFER, buffer_size_of_in,buffer_size_of_out
            );

            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);



            event1();
        }
    }
} // extern "C"
