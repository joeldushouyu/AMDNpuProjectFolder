//===- kernel1.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>


//TODO: multiple vent0?
template<typename T>
void loadFloatToCascade(T *__restrict buff, const int size){
    static_assert(std::is_same<T, float>::value);
    event0();

    for(unsigned int i = 0; i < size; i+=16){
        aie::vector<float, 16> data = aie::load_v<16>(buff);
        buff+=16;
        
        v16accfloat data_acc = v16accfloat(data);
        //https://docs.amd.com/r/en-US/am025-versal-aie-ml-register-reference/Accumulator_Control-CORE_MODULE-Register
        // guessing the en flag is either 0 (to south) and 1 to east for output??
        event0();
        put_mcd(  data_acc, 1 );
    } 
 
    event1();

}


extern "C" {


    void loadFloatVectorToCascade(float *data, const int size){
        loadFloatToCascade<float>(data, size);
    }


} // extern "C"