#ifndef __MVM_SEQUENCE_HPP__
#define __MVM_SEQUENCE_HPP__

#include "npu_utils.hpp"

///@brief generate the mvm sequence
///@param M the number of rows of the matrix A
///@param K the number of columns of the matrix A
///@param m the row tile size
///@param k the column tile size
///@param rows the number of CT rows
///@param cols the number of CT columns
///@return the npu sequence
///@note The function will generate the mvm sequence for the matrix multiplication
///@warning The function is only used for the npu2
///@warning This function will write the sequence to a file named "generated.txt"
///@warning The sequence name is "mvm_i8"
npu_sequence generate_mvm_sequence(uint32_t M, uint32_t K, uint32_t m, uint32_t k, uint32_t rows, uint32_t cols){
    npu_sequence seq;
    int cores = rows * cols;
    const int Arg_A = 0;
    const int Arg_B = 1;
    const int Arg_C = 2;
    seq.setup_device(device_npu2);

    seq.name_instr("mvm_i8");
    std::vector<npu_tiles> shim_tiles;
    for (int i = 0; i < cols; i++){
        shim_tiles.push_back(get_tile(0, i));
    }
    /*
    npu_dma_memcpy_nd(
        metadata=memB_fifo,
        bd_id=2,
        mem=B,
        offsets=[0, 0, 0, 0],
        sizes=[M // m // n_cores, 1, 1, K],
        strides=[0, 0, 0, 1],
    )
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][4, 1, 1, 512][0, 0, 0, 1]) {id = 2 : i64, metadata = @memB} : memref<512xi8>
    */
    seq.npu_dma_memcpy_nd(
        sizeof(int8_t),
        Arg_B, 
        MM2S,
        shim_tiles[0],
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {M / m / cores, 1, 1, K},
        {0, 0, 0, 1},
        -1, 0, false
    );

    for (int i = 0; i < cols; i++){
        uint32_t A_offset = i * M * K / cols;
        uint32_t C_offset = i * M / cols;
        // npu_dma_memcpy_nd(
        //     metadata=memA_fifos[i],
        //     bd_id=1,
        //     mem=A,
        //     offsets=[0, 0, 0, A_offset],
        //     sizes=[M // mvm_cols // (m * mvm_rows), K_div_k, mvm_rows * m, k],
        //     strides=[m_x_K * mvm_rows, k, K, 1],
        // )
                    
        seq.npu_dma_memcpy_nd(
            sizeof(int8_t),
            Arg_A, 
            MM2S,
            shim_tiles[i],
            bd_1,
            it_channel_0,
            {0, 0, 0, A_offset},
            {M / cores / m, K / k, rows * m, k},
            {m * K * rows, k, K, 1},
            -1, 0, false
        );
        // npu_dma_memcpy_nd(
        //     metadata=memC_fifos[i],
        //     bd_id=0,
        //     mem=C,
        //     offsets=[0, 0, 0, C_offset],
        //     sizes=[1, 1, M // m // mvm_cols // mvm_rows, mvm_rows * m],
        //     strides=[0, 0, mvm_rows * m, 1],
        // )
        seq.npu_dma_memcpy_nd(
            sizeof(int32_t),
            Arg_C,
            S2MM,
            shim_tiles[i],
            bd_0,
            it_channel_0,
            {0, 0, 0, C_offset},
            {1, 1, M / cores / m, rows * m},
            {0, 0, rows * m, 1},
            -1, 0, true
        );
        
    }

    // DMA wait

    for (int i = 0; i < cols; i++){
        seq.npu_dma_wait(
            shim_tiles[i],
            S2MM,
            it_channel_0
        );
    }

    seq.to_npu();
    seq.write_out_sequence("generated.txt");

    return seq;
}

#endif
