// //===- passThrough.cc -------------------------------------------*- C++ -*-===//
// //
// // This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// // Copyright (C) 2022, Advanced Micro Devices, Inc.
// //
// //===----------------------------------------------------------------------===//

// // #define __AIENGINE__ 1
// #define NOCPP

// #include <stdint.h>
// #include <stdlib.h>

// #include <aie_api/aie.hpp>
// #include "circuitSimulation.h"



// float* retrieveMatrixOFfsetBaseOnState(const uint32_t state, const int32_t matrix_size, float* matrix_ptr) {

//     return  matrix_ptr + (state *matrix_size);
// }
// extern "C" {


//     void CT_main(float* in, float* out,
//         const int32_t buffer_size_of_in, const int32_t buffer_size_of_out,
//         const int32_t iteration_ste_per_buffer,
//         const int32_t buffer_in_prod_lock_id, const int32_t buffer_in_con_loc_id,
//         const int32_t buffer_out_prod_lock_id, const int32_t buffer_out_con_lock_id,

//         float* C1_DSW_Buffer,
//         const int32_t C1_DSW_row_size,
//         const int32_t C1_DSW_col_size
//         ) {

//         const uint32_t C1_matrix_size = C1_DSW_row_size * C1_DSW_col_size;

//         while (true) {
//             acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
//             acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);

//             // only read first 16 input and write corresponding C1_diode_SW_Buffer back
//             for(int i = 0; i < 16; i++){
//                 float input  = *in;
//                 in++;
//                 uint32_t input_switch_state = *in;
//                 in ++;

//                 float *offset_header =  retrieveMatrixOFfsetBaseOnState( input_switch_state,
//                 C1_matrix_size, C1_DSW_Buffer  );
                
//                 // now write the matrix back to output
//                 for(int k  = 0; k <C1_matrix_size; k++ ){
//                     *out = *offset_header;
//                     out++;
//                     offset_header++;

//                 }
//             }   

            
//             release(buffer_in_prod_lock_id + 48, 1);
//             release(buffer_out_con_lock_id + 48, 1);



//             acquire_greater_equal(buffer_in_con_loc_id + 48, 1);
//             acquire_greater_equal(buffer_out_prod_lock_id + 48, 1);

//                 // only read first 16 input and write corresponding C1_diode_SW_Buffer back
//                 for(int i = 0; i < 16; i++){
//                     float input  = *in;
//                     in++;
//                     uint32_t input_switch_state = *in;
//                     in ++;

//                     float *offset_header =  retrieveMatrixOFfsetBaseOnState( input_switch_state,
//                     C1_matrix_size, C1_DSW_Buffer  );
                    
//                     // now write the matrix back to output
//                     for(int k  = 0; k <C1_matrix_size; k++ ){
//                         *out = *offset_header;
//                         out++;
//                         offset_header++;

//                     }
//                 }   


//             release(buffer_in_prod_lock_id + 48, 1);
//             release(buffer_out_con_lock_id + 48, 1);




//         }
//     }

// } // extern "C"
