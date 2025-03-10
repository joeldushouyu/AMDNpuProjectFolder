#ifndef __ADD_H__
#define __ADD_H__

template <typename T, int W = 128>
void add_aie(T *restrict in0, T *restrict out) {
    // // one possible method is use both unaligned store and load
    // // assume the input is 260 size and output is 256 in size
    // // However, it will take much longer time
    // event0();
    
    // const int vec_factor = 32;
 
    // aie::vector<T, vec_factor> In0;
    // aie::vector<T, vec_factor> Out;

    // // T *restrict n_in0 = in0 + 4;
    // // in0+= 4;
    // // but this is unaligned, since alignment requre in unit of 32 byte
    // const int F = W / vec_factor;
    // for (int i = 0; i < F; i++)
    //     chess_prepare_for_pipelining chess_loop_range(6, ) 
    //     { 
    //         // if(i == 0){
    //             // In0 = aie::load_unaligned_v<vec_factor>(in0);
    //             In0 = aie::load_unaligned_v<vec_factor>(in0+4);
    //         // }else{
    //         //     In0 = aie::load_v<vec_factor>(in0);
    //         // }

    //         Out = aie::add(In0, (T)2);
    //         // if (i != 0){
    //         //     aie::store_v(out, Out);
    //         //     in0 += vec_factor;
    //         //     out += vec_factor;
    //         // }else{
    //             // do a unaligned store
    //             aie::store_unaligned_v(out, Out, 1);
    //             // aie::store_v(out, Out);
    //             in0 +=vec_factor;
    //             out += vec_factor;
    //         // }

    //     }
    // event1();
    ///method 2: skip first "vect_factor" in datqa
    // pro: keep align load/store
    // con: waster "vect_factor"-4(packet header size) of byte in this case
    event0();
    const int vec_factor = 16;

    aie::vector<T, vec_factor> In0;
    aie::vector<T, vec_factor> Out;
    in0+= vec_factor; // skip first 16 byte
    const int F = W / vec_factor;
    for (int i = 0; i < F; i++)
        chess_prepare_for_pipelining chess_loop_range(6, ) { 
            In0 = aie::load_v<vec_factor>(in0);
            Out = aie::add(In0, (T)2);
            aie::store_v(out, Out);
            in0 += vec_factor;
            out += vec_factor;
        }
    event1();


}
#endif
