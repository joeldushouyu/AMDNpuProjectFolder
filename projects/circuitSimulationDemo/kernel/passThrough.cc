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

#include <aie_api/aie.hpp>

template <typename T, int N>
__attribute__((noinline)) void passThrough_aie(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {
  event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
    chess_prepare_for_pipelining chess_loop_range(6, ) { *outPtr++ = *inPtr++; }

  event1();
}

template<typename T>
__attribute__((noinline)) void passThrough_simple(float *restrict in, float*restrict out, const int32_t size){
  
  event0();

  for( int32_t j = 0; j < size; j++){
    *out = *in;
    out++;
    in++;
  }
  event1();
}


template<typename T>
__attribute__((noinline)) void accumValue(float *restrict in, float*restrict out,
  const int32_t in_to_out_ratio ,const int32_t  out_size  
){
  
  event0();

  for(int32_t j = 0; j < out_size; j++){
    float acc_data = 0;
    for(int32_t k = 0; k < in_to_out_ratio; k++){
      acc_data += *in;
      in++;
    }
    *out = acc_data;
    out++;
  }
  event1();
}
extern "C" {

// #if BIT_WIDTH == 8

// void passThroughLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
//   passThrough_aie<uint8_t, 64>(in, out, 1, lineWidth);
// }

// void passThroughTile(uint8_t *in, uint8_t *out, int32_t tileHeight,
//                      int32_t tileWidth) {
//   passThrough_aie<uint8_t, 64>(in, out, tileHeight, tileWidth);
// }

// #elif BIT_WIDTH == 16

// void passThroughLine(int16_t *in, int16_t *out, int32_t lineWidth) {
//   passThrough_aie<int16_t, 32>(in, out, 1, lineWidth);
// }

// void passThroughTile(int16_t *in, int16_t *out, int32_t tileHeight,
//                      int32_t tileWidth) {
//   passThrough_aie<int16_t, 32>(in, out, tileHeight, tileWidth);
// }

// #else // 32

// void passThroughLine(int32_t *in, int32_t *out, int32_t lineWidth) {
//   passThrough_aie<int32_t, 16>(in, out, 1, lineWidth);
// }

// void passThroughTile(int32_t *in, int32_t *out, int32_t tileHeight,
//                      int32_t tileWidth) {
//   passThrough_aie<int32_t, 16>(in, out, tileHeight, tileWidth);
// }

// #endif
void passThroughLine_float_0(float *in, float *out, int32_t lineWidth) {
  passThrough_simple<float>( in, out, lineWidth);
}
void passThroughLine_float_1(float *in, float *out, int32_t lineWidth) {
  passThrough_simple<float>( in, out, lineWidth);
}

void passThroughLine_float_2(float *in, float *out, int32_t lineWidth) {
  passThrough_simple<float>( in, out, lineWidth);
}

void passThroughLine_float_3(float *in, float *out, int32_t lineWidth) {
  passThrough_simple<float>( in, out, lineWidth);
}

void accum_float_value(float *in, float *out, int32_t in_to_out_ratio, int32_t out_size){
  accumValue<float>(in,out,in_to_out_ratio, out_size);
}


} // extern "C"
