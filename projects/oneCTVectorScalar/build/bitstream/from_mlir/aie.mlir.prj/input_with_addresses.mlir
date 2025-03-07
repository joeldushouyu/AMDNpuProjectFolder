module {
  aie.device(npu1_1col) {
    func.func private @vector_scalar_mul_int16_vector(memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "_anonymous0", sys_name = "out_buff_0"} : memref<1024xi16> 
    %_anonymous1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "_anonymous1", sys_name = "out_buff_1"} : memref<1024xi16> 
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sys_name = "out_prod_lock"}
    %lock_0_2_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sys_name = "out_con_lock"}
    %_anonymous2 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "_anonymous2", sys_name = "input_vector_buffer_0"} : memref<1024xi16> 
    %_anonymous3 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "_anonymous3", sys_name = "input_vector_buffer_1"} : memref<1024xi16> 
    %lock_0_2_1 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sys_name = "input_vector_prod_lock"}
    %lock_0_2_2 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sys_name = "input_vector_con_lock"}
    %_anonymous4 = aie.buffer(%tile_0_2) {address = 3072 : i32, mem_bank = 0 : i32, sym_name = "_anonymous4", sys_name = "input_scalar_buffer_0"} : memref<1xi32> 
    %_anonymous5 = aie.buffer(%tile_0_2) {address = 18432 : i32, mem_bank = 1 : i32, sym_name = "_anonymous5", sys_name = "input_scalar_buffer_1"} : memref<1xi32> 
    %lock_0_2_3 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sys_name = "input_scalar_prod_lock"}
    %lock_0_2_4 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sys_name = "input_scalar_con_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1024_i32 = arith.constant 1024 : i32
      %c1024_i32_5 = arith.constant 1024 : i32
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c4 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      func.call @vector_scalar_mul_int16_vector(%_anonymous2, %_anonymous0, %_anonymous4, %c1024_i32) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      func.call @vector_scalar_mul_int16_vector(%_anonymous3, %_anonymous1, %_anonymous4, %c1024_i32_5) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2_0, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c4 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      func.call @vector_scalar_mul_int16_vector(%_anonymous2, %_anonymous0, %_anonymous5, %c1024_i32) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      func.call @vector_scalar_mul_int16_vector(%_anonymous3, %_anonymous1, %_anonymous5, %c1024_i32_5) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
      aie.use_lock(%lock_0_2_1, Release, 1)
      aie.use_lock(%lock_0_2_0, Release, 1)
      %7 = arith.addi %5, %c2 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%lock_0_2_3, Release, 1)
      %8 = arith.addi %0, %c2 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    } {link_with = "scale.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<4096xi16>, %arg1: memref<1xi32>, %arg2: memref<4096xi16>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      } {issue_token = true}
      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>])
        aie.end
      } {issue_token = true}
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
    }
    aie.shim_dma_allocation @in(MM2S, 0, 0)
    aie.shim_dma_allocation @infactor(MM2S, 1, 0)
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%_anonymous2 : memref<1024xi16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%_anonymous3 : memref<1024xi16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_2_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%_anonymous4 : memref<1xi32>, 0, 1) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%_anonymous5 : memref<1xi32>, 0, 1) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%_anonymous0 : memref<1024xi16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%_anonymous1 : memref<1024xi16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
