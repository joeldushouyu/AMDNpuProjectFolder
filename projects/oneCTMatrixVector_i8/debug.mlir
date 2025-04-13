module {
  aie.device(npu2) {
    memref.global "public" @C_MT_SHM_cons : memref<128xi32>
    memref.global "public" @C_MT_SHM : memref<128xi32>
    memref.global "public" @C_CT_MT_cons : memref<128xi32>
    memref.global "public" @C_CT_MT : memref<128xi32>
    memref.global "public" @B_MT_CT_cons : memref<128xi8>
    memref.global "public" @B_MT_CT : memref<128xi8>
    memref.global "public" @B_SHM_MT_cons : memref<128xi8>
    memref.global "public" @B_SHM_MT : memref<128xi8>
    memref.global "public" @A_MT_CT_cons : memref<128x128xi8>
    memref.global "public" @A_MT_CT : memref<128x128xi8>
    memref.global "public" @A_SHM_MT_cons : memref<16384xi8>
    memref.global "public" @A_SHM_MT : memref<16384xi8>
    func.func private @zero_m_int8(memref<128xi32>)
    func.func private @mv_int8(memref<128x128xi8>, memref<128xi8>, memref<128xi32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %C_MT_SHM_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 1 : i32, sym_name = "C_MT_SHM_cons_prod_lock_0"}
    %C_MT_SHM_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_MT_SHM_cons_cons_lock_0"}
    %C_CT_MT_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "C_CT_MT_cons_buff_0"} : memref<128xi32> 
    %C_CT_MT_cons_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "C_CT_MT_cons_buff_1"} : memref<128xi32> 
    %C_CT_MT_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "C_CT_MT_cons_prod_lock_0"}
    %C_CT_MT_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_CT_MT_cons_cons_lock_0"}
    %C_CT_MT_buff_0 = aie.buffer(%tile_0_2) {sym_name = "C_CT_MT_buff_0"} : memref<128xi32> 
    %C_CT_MT_buff_1 = aie.buffer(%tile_0_2) {sym_name = "C_CT_MT_buff_1"} : memref<128xi32> 
    %C_CT_MT_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "C_CT_MT_prod_lock_0"}
    %C_CT_MT_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_CT_MT_cons_lock_0"}
    %B_MT_CT_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "B_MT_CT_cons_buff_0"} : memref<128xi8> 
    %B_MT_CT_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "B_MT_CT_cons_buff_1"} : memref<128xi8> 
    %B_MT_CT_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "B_MT_CT_cons_prod_lock_0"}
    %B_MT_CT_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_MT_CT_cons_cons_lock_0"}
    %B_SHM_MT_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "B_SHM_MT_cons_buff_0"} : memref<128xi8> 
    %B_SHM_MT_cons_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "B_SHM_MT_cons_buff_1"} : memref<128xi8> 
    %B_SHM_MT_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_SHM_MT_cons_prod_lock_0"}
    %B_SHM_MT_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_SHM_MT_cons_cons_lock_0"}
    %B_SHM_MT_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "B_SHM_MT_prod_lock_0"}
    %B_SHM_MT_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_SHM_MT_cons_lock_0"}
    %A_MT_CT_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "A_MT_CT_cons_buff_0"} : memref<128x128xi8> 
    %A_MT_CT_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "A_MT_CT_cons_buff_1"} : memref<128x128xi8> 
    %A_MT_CT_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_MT_CT_cons_prod_lock_0"}
    %A_MT_CT_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_MT_CT_cons_cons_lock_0"}
    %A_SHM_MT_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "A_SHM_MT_cons_buff_0"} : memref<16384xi8> 
    %A_SHM_MT_cons_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "A_SHM_MT_cons_buff_1"} : memref<16384xi8> 
    %A_SHM_MT_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_SHM_MT_cons_prod_lock_0"}
    %A_SHM_MT_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_SHM_MT_cons_cons_lock_0"}
    %A_SHM_MT_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "A_SHM_MT_prod_lock_0"}
    %A_SHM_MT_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_SHM_MT_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    %rtp_buffer = aie.buffer(%tile_0_2) {sym_name = "rtp_buffer"} : memref<16xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967294 step %c2 {
        aie.use_lock(%C_CT_MT_prod_lock_0, AcquireGreaterEqual, 1)
        func.call @zero_m_int8(%C_CT_MT_buff_0) : (memref<128xi32>) -> ()
        %c0_3 = arith.constant 0 : index
        %c4_4 = arith.constant 4 : index
        %c1_5 = arith.constant 1 : index
        %c2_6 = arith.constant 2 : index
        scf.for %arg1 = %c0_3 to %c4_4 step %c2_6 {
          aie.use_lock(%A_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%B_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          func.call @mv_int8(%A_MT_CT_cons_buff_0, %B_MT_CT_cons_buff_0, %C_CT_MT_buff_0) : (memref<128x128xi8>, memref<128xi8>, memref<128xi32>) -> ()
          aie.use_lock(%A_MT_CT_cons_prod_lock_0, Release, 1)
          aie.use_lock(%B_MT_CT_cons_prod_lock_0, Release, 1)
          aie.use_lock(%A_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%B_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          func.call @mv_int8(%A_MT_CT_cons_buff_1, %B_MT_CT_cons_buff_1, %C_CT_MT_buff_0) : (memref<128x128xi8>, memref<128xi8>, memref<128xi32>) -> ()
          aie.use_lock(%A_MT_CT_cons_prod_lock_0, Release, 1)
          aie.use_lock(%B_MT_CT_cons_prod_lock_0, Release, 1)
        }
        aie.use_lock(%C_CT_MT_cons_lock_0, Release, 1)
        aie.use_lock(%C_CT_MT_prod_lock_0, AcquireGreaterEqual, 1)
        func.call @zero_m_int8(%C_CT_MT_buff_1) : (memref<128xi32>) -> ()
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %c1_9 = arith.constant 1 : index
        %c2_10 = arith.constant 2 : index
        scf.for %arg1 = %c0_7 to %c4_8 step %c2_10 {
          aie.use_lock(%A_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%B_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          func.call @mv_int8(%A_MT_CT_cons_buff_0, %B_MT_CT_cons_buff_0, %C_CT_MT_buff_1) : (memref<128x128xi8>, memref<128xi8>, memref<128xi32>) -> ()
          aie.use_lock(%A_MT_CT_cons_prod_lock_0, Release, 1)
          aie.use_lock(%B_MT_CT_cons_prod_lock_0, Release, 1)
          aie.use_lock(%A_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%B_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
          func.call @mv_int8(%A_MT_CT_cons_buff_1, %B_MT_CT_cons_buff_1, %C_CT_MT_buff_1) : (memref<128x128xi8>, memref<128xi8>, memref<128xi32>) -> ()
          aie.use_lock(%A_MT_CT_cons_prod_lock_0, Release, 1)
          aie.use_lock(%B_MT_CT_cons_prod_lock_0, Release, 1)
        }
        aie.use_lock(%C_CT_MT_cons_lock_0, Release, 1)
      }
      aie.use_lock(%C_CT_MT_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_m_int8(%C_CT_MT_buff_0) : (memref<128xi32>) -> ()
      %c0_0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_1 = arith.constant 1 : index
      %c2_2 = arith.constant 2 : index
      scf.for %arg0 = %c0_0 to %c4 step %c2_2 {
        aie.use_lock(%A_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
        aie.use_lock(%B_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
        func.call @mv_int8(%A_MT_CT_cons_buff_0, %B_MT_CT_cons_buff_0, %C_CT_MT_buff_0) : (memref<128x128xi8>, memref<128xi8>, memref<128xi32>) -> ()
        aie.use_lock(%A_MT_CT_cons_prod_lock_0, Release, 1)
        aie.use_lock(%B_MT_CT_cons_prod_lock_0, Release, 1)
        aie.use_lock(%A_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
        aie.use_lock(%B_MT_CT_cons_cons_lock_0, AcquireGreaterEqual, 1)
        func.call @mv_int8(%A_MT_CT_cons_buff_1, %B_MT_CT_cons_buff_1, %C_CT_MT_buff_0) : (memref<128x128xi8>, memref<128xi8>, memref<128xi32>) -> ()
        aie.use_lock(%A_MT_CT_cons_prod_lock_0, Release, 1)
        aie.use_lock(%B_MT_CT_cons_prod_lock_0, Release, 1)
      }
      aie.use_lock(%C_CT_MT_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "mvm_i8.o"}
    aiex.runtime_sequence @sequenc(%arg0: memref<262144xi8>, %arg1: memref<512xi8>, %arg2: memref<512xi32>) {
      aiex.npu.rtp_write(@rtp_buffer, 0, 1)
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][4, 1, 1, 512][0, 0, 0, 1]) {id = 2 : i64, metadata = @B_SHM_MT} : memref<512xi8>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][4, 4, 128, 128][65536, 128, 512, 1]) {id = 1 : i64, metadata = @A_SHM_MT} : memref<262144xi8>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 4, 128][0, 0, 128, 1]) {id = 0 : i64, metadata = @C_MT_SHM} : memref<512xi32>
      aiex.npu.dma_wait {symbol = @C_MT_SHM}
    }
    aie.shim_dma_allocation @A_SHM_MT(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_SHM_MT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_SHM_MT_cons_buff_0 : memref<16384xi8>, 0, 16384)
      aie.use_lock(%A_SHM_MT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_SHM_MT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_SHM_MT_cons_buff_1 : memref<16384xi8>, 0, 16384)
      aie.use_lock(%A_SHM_MT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_SHM_MT_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_SHM_MT_cons_buff_0 : memref<16384xi8>, 0, 16384, [<size = 32, stride = 4>, <size = 128, stride = 128>, <size = 4, stride = 1>])
      aie.use_lock(%A_SHM_MT_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_SHM_MT_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_SHM_MT_cons_buff_1 : memref<16384xi8>, 0, 16384, [<size = 32, stride = 4>, <size = 128, stride = 128>, <size = 4, stride = 1>])
      aie.use_lock(%A_SHM_MT_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_SHM_MT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_SHM_MT_cons_buff_0 : memref<128xi8>, 0, 128)
      aie.use_lock(%B_SHM_MT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_SHM_MT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_SHM_MT_cons_buff_1 : memref<128xi8>, 0, 128)
      aie.use_lock(%B_SHM_MT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_SHM_MT_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_SHM_MT_cons_buff_0 : memref<128xi8>, 0, 128)
      aie.use_lock(%B_SHM_MT_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_SHM_MT_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_SHM_MT_cons_buff_1 : memref<128xi8>, 0, 128)
      aie.use_lock(%B_SHM_MT_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_CT_MT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_CT_MT_cons_buff_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%C_CT_MT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_CT_MT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_CT_MT_cons_buff_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%C_CT_MT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_CT_MT_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_CT_MT_cons_buff_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%C_CT_MT_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_CT_MT_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_CT_MT_cons_buff_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%C_CT_MT_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_MT_CT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_MT_CT_cons_buff_0 : memref<128x128xi8>, 0, 16384)
      aie.use_lock(%A_MT_CT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_MT_CT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_MT_CT_cons_buff_1 : memref<128x128xi8>, 0, 16384)
      aie.use_lock(%A_MT_CT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_MT_CT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_MT_CT_cons_buff_0 : memref<128xi8>, 0, 128)
      aie.use_lock(%B_MT_CT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_MT_CT_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_MT_CT_cons_buff_1 : memref<128xi8>, 0, 128)
      aie.use_lock(%B_MT_CT_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%C_CT_MT_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_CT_MT_buff_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%C_CT_MT_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%C_CT_MT_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_CT_MT_buff_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%C_CT_MT_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @B_SHM_MT(MM2S, 1, 0)
    aie.shim_dma_allocation @C_MT_SHM(S2MM, 0, 0)
  }
}

