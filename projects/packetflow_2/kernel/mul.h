#ifndef __MUL_H__
#define __MUL_H__

template <typename T, int W = 128>
void mul_aie(T *restrict in0, T *restrict out) {
    event0();
    const int vec_factor = 16;

    aie::vector<T, vec_factor> In0;
    aie::accum<acc32, vec_factor> Out;

    const int F = W / vec_factor;
    for (int i = 0; i < F; i++)
        chess_prepare_for_pipelining chess_loop_range(6, ) { 
            In0 = aie::load_v<vec_factor>(in0);
            Out = aie::mul(In0, (T)3);
            aie::store_v(out, Out.template to_vector<T>());
            in0 += vec_factor;
            out += vec_factor;
        } 
    event1();
}
#endif
