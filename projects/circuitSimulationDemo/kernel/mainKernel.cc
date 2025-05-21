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
#include <vector>
#include "circuitConfig.h"
#include "circuitSimulation.h"


  
float* retrieveMatrixOffsetBaseOnState(const uint32_t state, const int32_t matrix_size, float* matrix_ptr) {

    return  matrix_ptr + (state * matrix_size);
}



template<typename T>
__attribute__((noinline)) void accumValue(float* restrict in, float* restrict out,
    const int32_t in_offset, const int32_t out_offset
) {

    in += in_offset;
    out += out_offset;
    // assert divide out without any remainder
    int32_t input_per_iteration_size = INPUT_SIZE_PER_ITERATION;
    int32_t output_per_iteration_size = OUTPUT_SIZE_PER_ITERATION;
    for (int32_t i = 0; i < ITERATION_STEP_PER_PING_PONG_BUFFER * PING_PONG_BUFFER_ITERATION; i++) {
        float acc = 0;
        for (int32_t k = 0; k < input_per_iteration_size; k++) {
            acc += *in;
            in++;
        }
        for (int32_t l = 0; l < output_per_iteration_size; l++) {
            *out = acc;
            out++;
        }

    }

}



void accum_float_value(float* in, float* out,
    const int32_t in_offset, const int32_t out_offset
    // uint32_t *debug_input
) {
    event0();
    accumValue<float>(in, out, in_offset, out_offset);
    event1();
}




void mult_with_C1_DSW(float *C1_DSW_mat, aie::vector<float, 16> *x_u_cur, float*out){

    aie::vector<float, 16> zero_vec = aie::zeros<float, 16>();
    const uint32_t C1_DSW_ROW_SIZE_DIV_16 = C1_DSW_ROW_SIZE/16;


    for(uint32_t row = 0; row < C1_DSW_ROW_SIZE_DIV_16; row+=1){

        aie::accum<accfloat, 16> C1_DSW_temp = aie::from_vector<accfloat>(zero_vec);

        for(uint32_t col = 0; col < U_SIZE+STATE_SIZE; col++){
            
            aie::vector<float, 16> a = aie::load_v<16>(C1_DSW_mat);
            C1_DSW_mat +=16;

             
            aie::vector<float, 16>b= aie::broadcast<float, 16>(   (x_u_cur+row)->get(col)  );
            C1_DSW_temp = mac_elem_16_accuracy_safe(a,b, C1_DSW_temp, 0,0,0);

        }
        //for now, store back to out
        aie::store_v(out, C1_DSW_temp.template to_vector<float>());
        out = out + 16;
    }


}

extern "C" {
    void CT_main(float* in, float* out,
        const int32_t buffer_in_prod_lock_id, const int32_t buffer_in_con_loc_id,
        const int32_t buffer_out_prod_lock_id, const int32_t buffer_out_con_lock_id,

        float* C1_DSW_Buffer,

        float* C1_DSW_mat_res, float* A_B_mat_Res, float* X_U_cur, float* C_D_mat_res
    ) {

        const int32_t C1_DSW_mat_size = C1_DSW_MATRIX_SIZE;
        const uint32_t externalSwitchDiodeStates = 0x0;
        
        //TODO: check later
        const uint32_t vector_size_of_x_u_cur = BUFFER_SIZE_OF_CUR_X_U / 16;
        
        static_assert(vector_size_of_x_u_cur < 12-1 ) ; // has 18 512bit accum, reserve one for mult with C1_DSW


        // Define storage for the accumulators
        aie::vector<float, 16> x_u_cur[vector_size_of_x_u_cur];


        for (uint32_t i = 0; i < vector_size_of_x_u_cur; ++i) {
            x_u_cur[i] = aie::zeros<float, 16>();
        }




        // // //test purpose
        // float v = 10.01;
        // x_u_cur[0].set(v, 0);
        // // x_u_cur[0] = aie::add(x_u_cur[0], v);



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

            

            // only do number of switch for now
            for(uint32_t k = 0; k < TOTAL_SWITCH_DIODE_STATE*2; k++){

                #pragma clang loop unroll_count(U_SIZE)
                for(auto i = STATE_SIZE; i < U_SIZE+STATE_SIZE ; i++ ){
                    
      
                    x_u_cur[ i /16 ].set(*in, i%16);
                    in++;
                }
                
                in++; // the input switch state

                mult_with_C1_DSW( retrieveMatrixOFfsetBaseOnState(k,C1_DSW_MATRIX_SIZE  ,C1_DSW_Buffer), 
                 x_u_cur,
                 out + k*(C1_DSW_ROW_SIZE) ); // for now write 16each time

                // mult_with_C1_DSW( C1_DSW_Buffer  ,  x_u_cur,out );
            }
            // x_u_cur[0].set(*in, 6);
            // mult_with_C1_DSW( C1_DSW_Buffer  ,  x_u_cur,out );
            


            // // use buffer 0 of ping in and out
            // accum_float_value(in, out, 
            // 0, 0
            // );


            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);

            acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
            acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);
            // // use buffer 0 of ping in and out
            // accum_float_value(in, out, 
            //   BUFFER_SIZE_OF_IN_PING_POING, BUFFER_SIZE_OF_OUT_PING_PONG
            // );

            release(buffer_in_prod_lock_id + 48, 1);
            release(buffer_out_con_lock_id + 48, 1);



            event1();
        }
    }
} // extern "C"
