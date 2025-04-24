#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "zero.h"
#include <aie_api/aie.hpp>

// A is column major in grandualtiy of
// Since vector size is 256 bit for operation
// Each time the B is 256 bit, means 8 float at max

template <typename T_in, typename T_out, typename T_acc, unsigned m, unsigned k,  unsigned t_size>
void matVec_float32(T_in *__restrict a, T_in *__restrict b, T_out *__restrict c)
{
    static_assert(t_size ==16); // grandualtiy of 256 bit or 512 bit operation, support 256 bit for now
    static_assert(std::is_same<T_in, float >::value && std::is_same<T_out, float>::value  );
    static_assert(std::is_same<T_acc, accfloat>::value); // seem default accumulator for https://xilinx.github.io/aie_api/group__group__arithmetic.html

    // for now, assume M and K is factor of R
    static_assert(m % t_size == 0 && k % t_size == 0);

    event0();

    T_in *__restrict a_ptr = a;
    T_in *__restrict b_ptr = b;
    T_out *__restrict c_ptr = c; // reset to the first row of C output on

    /*
    // v16float *__restrict a_ptr =  (v16float *)a;
    // v16float *__restrict b_ptr =(v16float *) b;
    // v16float *__restrict c_ptr = (v16float *)c;

    // r is number of element of "b" row and number of "a" column, in other word
    // this is limite to 256 or 512/float32, as required that coefficient factor is size limited to 256b or 512 b

    // t == 8 is the number of column in "c" and number of element in each column of "a"
    // This is still limited to 256 or 512 or 1024. C = A*b, where the result of c is 256/accfloat32 or 512/accfloat32 or 1024/float32

    // for (unsigned int row = 0; row < m; row += t_size)
    // {

    //     aie::accum<T_acc, t_size> c_acc_in;
    //     c_acc_in.from_vector(aie::load_v<r_size>(c_ptr));

    //     a_ptr = a + t_size * k;
    //     b_ptr = b;

    //     for (unsigned int col = 0; col < k; col += r_size)
    //     { // each time, load only 8 column of A for now

    //         aie::vector<T_in, r_size> b_vec = aie::load_v<r_size>(b_ptr); // size of 8 because of 256/32
    //         // let each load be granduality of 256 bit
    //         // thus load 8 float at most
    //         // The number of element in each A tis 8 because C_out is accumulate float
    //         // where 256/float32   = 8
    //         const aie::vector<T_in, t_size> a_col1 = aie::load_v<t_size>(a_ptr);
    //         const aie::vector<T_in, t_size> a_col2 = aie::load_v<t_size>(a_ptr + t_size);
    //         const aie::vector<T_in, t_size> a_col3 = aie::load_v<t_size>(a_ptr + 2 * t_size);
    //         const aie::vector<T_in, t_size> a_col4 = aie::load_v<t_size>(a_ptr + 3 * t_size);
    //         const aie::vector<T_in, t_size> a_col5 = aie::load_v<t_size>(a_ptr + 4 * t_size);
    //         const aie::vector<T_in, t_size> a_col6 = aie::load_v<t_size>(a_ptr + 5 * t_size);
    //         const aie::vector<T_in, t_size> a_col7 = aie::load_v<t_size>(a_ptr + 6 * t_size);
    //         const aie::vector<T_in, t_size> a_col8 = aie::load_v<t_size>(a_ptr + 7 * t_size);

    //         // don't even need to filter even or odd here

    //         c_acc_in = aie::accumulate<r_size>(c_acc_in, b_vec, 0,
    //                                            a_col1, a_col2, a_col3, a_col4,
    //                                            a_col5, a_col6, a_col7, a_col8);
    //         // now move to next 8 colum of A and next "r" row of B
    //         a_ptr += r_size * t_size; // 8 column, each colum is 8 elements
    //         b_ptr += r_size;
    //     }
    //     aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
    //     c_ptr += t_size;
    // }


    // Given there is a build in function that does mac of 16x16 float32 element wise multiplication

    // let A be order as  column major with grandualtiy of 16
    // A to be
    // 0000000000000000  888888888888888888 
    // 1111111111111111
    // 2222222222222222
    // ................

    */

    // A is order as colum major wirh granduality of 1 only, and lastest long for 16 float (512 bit retg) 
    /*
        0,1,2,3
        0,1,2,3
        .....  repeat uitl 16 
        4,5,6,7 // the 17 row(start counting from row 1)
    */

    // B is just a normal row-majored vector lf kx1 in size
    // v16float *__restrict a_ptr =  (v16float *)a;
    // v16float *__restrict b_ptr =(v16float *) b;
    // v16float *__restrict c_ptr = (v16float *)c;

    //TODO: check k, m is dividsible by 16

    ///home/shouyud/PROJECT/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include/aie_api/detail/aie2p/emulated_mmul_intrinsics.hpp
    ///home/shouyud/PROJECT/mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/lib/clang/19/include/aie2p_vmult.h
    
    for(unsigned int col = 0; col< k; col +=16){

        aie::accum<T_acc, 16> c_acc_in;
        c_acc_in.from_vector(aie::load_v<16>(c_ptr));
        b_ptr = b;
        for(unsigned int row = 0; row < m; row += 16){
            
            aie::vector<T_in, 16> b_vec = aie::load_v<16>(b_ptr);
            
            #pragma clang loop unroll_count(16)
            for (unsigned int l = 0; l < 16; l++){
                // load each A1... An and scatter each 
                aie::vector<T_in, 16> a_vec_0 = aie::load_v<16>(a_ptr);
                a_ptr += 16;
                aie::vector<T_in, 16> b0 = aie::broadcast<T_in, 16>(b_vec[l]);
                c_acc_in = mac_elem_16_accuracy_safe(a_vec_0, b0, c_acc_in,0,0,0);
                // c_acc_in = mac_elem_16(a_vec_0, b0, c_acc_in);

                // load second A
            }

            b_ptr +=16; // next 16 in the b
        }
        aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
        c_ptr += 16;     // Move to next r rows of the same columns in A.

        
        
    }


    
    event1();
}

extern "C"
{

#ifndef MV_M_fp32
#define MV_M_fp32 64
#endif

#ifndef MV_K_fp32
#define MV_K_fp32 64
#endif

    float test_float_operation(float x, float y, float z){
        // #pragma clang fp reassociate(on)
        float t = x+y;
        float v = t+z+100.23;
        return v;
    }
    void mv_float32(
        float *restrict a,
        float *restrict b,
        float *restrict c)
    {
        matVec_float32<float, float, accfloat, MV_M_fp32, MV_K_fp32, 16>(a, b, c);
    }

    void zero_m_float32(
        float *restrict c)
    {
        zero_vectorized<float, MV_M_fp32>(c);
    }
} // extern "C"


// // from /home/shouyud/PROJECT/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include/aie_api/detail/aie2p/emulated_mmul_intrinsics.hpp
// inline v16float extract_v4float_broadcast_to_v16float( v16float v, int idx ) {
//     return (v16float) broadcast_elem_128( (v16int32)v, idx );
// }

// inline aie::accum<accfloat, 16> mul_4x8_8x4_fp32(v32float x, v32float y)
// {
//     aie::vector<float, 16> xl = ::extract_v16float(x, 0);
//     aie::vector<float, 16> xh = ::extract_v16float(x, 1);
//     aie::vector<float, 16> xa = ::shuffle(xl, xh, T32_4x8_lo);
//     aie::vector<float, 16> xb = ::shuffle(xl, xh, T32_4x8_hi);

//     aie::vector<float, 16> x0 = ::shuffle(::extract_v4float_broadcast_to_v16float(xa, 0), T32_4x4);
//     aie::vector<float, 16> x1 = ::shuffle(::extract_v4float_broadcast_to_v16float(xa, 1), T32_4x4);
//     aie::vector<float, 16> x2 = ::shuffle(::extract_v4float_broadcast_to_v16float(xa, 2), T32_4x4);
//     aie::vector<float, 16> x3 = ::shuffle(::extract_v4float_broadcast_to_v16float(xa, 3), T32_4x4);
//     aie::vector<float, 16> x4 = ::shuffle(::extract_v4float_broadcast_to_v16float(xb, 0), T32_4x4);
//     aie::vector<float, 16> x5 = ::shuffle(::extract_v4float_broadcast_to_v16float(xb, 1), T32_4x4);
//     aie::vector<float, 16> x6 = ::shuffle(::extract_v4float_broadcast_to_v16float(xb, 2), T32_4x4);
//     aie::vector<float, 16> x7 = ::shuffle(::extract_v4float_broadcast_to_v16float(xb, 3), T32_4x4);

//     aie::vector<float, 16> y0 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 0), 0); // basically get each y row?
//     aie::vector<float, 16> y1 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 0), 1);
//     aie::vector<float, 16> y2 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 0), 2);
//     aie::vector<float, 16> y3 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 0), 3);
//     aie::vector<float, 16> y4 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 1), 4);
//     aie::vector<float, 16> y5 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 1), 5);
//     aie::vector<float, 16> y6 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 1), 6);
//     aie::vector<float, 16> y7 = ::extract_v4float_broadcast_to_v16float(::extract_v16float(y, 1), 7);

//     v16accfloat acc = ::mul_elem_16(x0, y0);
//     acc             = ::mac_elem_16(x1, y1, acc);
//     acc             = ::mac_elem_16(x2, y2, acc);
//     acc             = ::mac_elem_16(x3, y3, acc);
//     acc             = ::mac_elem_16(x4, y4, acc);
//     acc             = ::mac_elem_16(x5, y5, acc);
//     acc             = ::mac_elem_16(x6, y6, acc);
//     acc             = ::mac_elem_16(x7, y7, acc);

//     return acc;
// }
