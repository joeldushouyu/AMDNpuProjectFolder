

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

template<typename T>
void vectorElementWiseMultplication(T *__restrict in_vector, T*__restrict out_vector,  const int size){
    // 2 vector dot product, the other vector came from cascade stream from neighbort(west)


    static_assert(std::is_same<T, float>::value);

    event0();

    for(unsigned int i = 0; i < size; i+=16){
        aie::vector<float, 16> vec1 = aie::load_v<16>(in_vector);
        v16accfloat vec2 = get_scd_v16accfloat(1 ); // read from west?

        aie::accum<accfloat, 16> c_acc_res;
        // now do a dot multiplication
        
        c_acc_res =  mul_elem_16_accuracy_safe(vec1, vec2);
        
        aie::store_v(out_vector, c_acc_res.template to_vector<float>() );

        in_vector += 16;
        out_vector +=16;
        
    }


    event1();


}

extern "C"
{

    void vectorElementWiseMultiplication_from_cascade(float * in, float*out, int size){
        vectorElementWiseMultplication<float>(in, out, size);
    }
} // extern "C"