#include <cstdint>
#include <bitset>
#include <iostream>
#include <cmath>


#include <cassert>

#include "kernel/circuitSimulation.h"

bool approximatelyEqual(float a, float b, float epsilon = 0.0001f) {
    return std::fabs(a - b) < epsilon;
  }

void test_compare_and_copy_bits(){
    uint32_t src = 0b00001100;  // bits 2 and 3 = 1
    uint32_t dst = 0b10101110;  // bits 2 and 3 = 0

    bool same = compare_and_copy_bits(dst, src, 2, 3);
    assert (same == true);
    assert(dst == 0b10101110);


    src = 0b00001100;  
    dst = 0b10101110;  
    same = compare_and_copy_bits(dst, src, 1, 3);
    assert (same == false); // indicate a change
    assert(dst == 0b10101100);
}

void test_setBit(){
    uint32_t value =  0b00001100; 
    
    setBit(value, 1, 1);
    assert(value == 0b00001110 );
    setBit(value, 1, 0);
    assert(value == 0b00001100 );
}

void test_ROUND_UP_TO_16(){
    assert(16 == ROUND_UP_TO_16(1));
    assert(32 == ROUND_UP_TO_16(30));
    assert(16 == ROUND_UP_TO_16(16));
}

void test_update_x_cur_and_u_with_new_u(){

    float x_cur_with_u [16] = {0};
    
    float u_updates [U_SIZE] = {10.123};
    assert(U_SIZE == 1);
    assert(STATE_SIZE == 6);
    update_x_cur_and_u_with_new_u(x_cur_with_u, u_updates);

    // for(auto i = 0; i < 16; i++){
    //     std::cout << x_cur_with_u[i] <<std::endl ;
    // }
    assert(approximatelyEqual(x_cur_with_u[STATE_SIZE], 10.123));

    
}

int main() {
    test_compare_and_copy_bits();
    std::cout << "pass compare_and_copy_bits()" << std::endl;
    test_setBit();
    std::cout << "pass setBit()" << std::endl;
    test_ROUND_UP_TO_16();
    std::cout << "passed test_ROUND_UP_To_16" << std::endl;
    test_update_x_cur_and_u_with_new_u();
    std::cout << "passed test_update_x_cur_and_u_with_new_u"<< std::endl;
    return 0;
}    