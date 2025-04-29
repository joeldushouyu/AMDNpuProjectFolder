#include <cstdint>
#include <bitset>
#include <iostream>

bool compare_and_copy_bits(uint32_t& dst, uint32_t src, uint32_t pos, uint32_t num_bits) {
    // Create mask for the bit range
    uint32_t mask = ((1u << num_bits) - 1) << pos;

    // Extract masked bits from both src and dst
    bool bits_equal = (dst & mask) == (src & mask);

    // Set bits in dst to match src
    dst = (dst & ~mask) | (src & mask);

    return bits_equal;
}


int main() {
    uint32_t src = 0b00001100;  // bits 2 and 3 = 1
    uint32_t dst = 0b10101110;  // bits 2 and 3 = 0

    std::cout << "Before:\n";
    std::cout << "  src: " << std::bitset<32>(src) << '\n';
    std::cout << "  dst: " << std::bitset<32>(dst) << '\n';

    bool same = compare_and_copy_bits(dst, src, 2, 3);

    std::cout << "\nAfter:\n";
    std::cout << "  dst: " << std::bitset<32>(dst) << '\n';
    std::cout << "  Were bits the same? " << std::boolalpha << same << '\n';

    return 0;
}