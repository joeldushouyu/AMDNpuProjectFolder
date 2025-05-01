module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2) {allocation_scheme = "basic-sequential"}
    %tile_1_2 = aie.tile(1, 2)
    %in_buffer_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "in_buffer_0"} : memref<1120xf32> 
    %in_buffer_p_lock = aie.lock(%tile_0_2, 8) {init = 2 : i32, sym_name = "in_buffer_p_lock"}
    %in_buffer_c_lock = aie.lock(%tile_0_2, 9) {init = 0 : i32, sym_name = "in_buffer_c_lock"}
    %out_buffer_0 = aie.buffer(%tile_0_2) {address = 34176 : i32, sym_name = "out_buffer_0"} : memref<7840xf32> 
    %lock_0_2 = aie.lock(%tile_0_2, 10) {init = 2 : i32}
    %lock_0_2_0 = aie.lock(%tile_0_2, 11) {init = 0 : i32}
    %switch_diode_buffer = aie.buffer(%tile_0_2) {sym_name = "switch_diode_buffer"} : memref<1792xf32> 
    %switch_diode_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "switch_diode_prod_lock"}
    %switch_diode_con_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "switch_diode_con_lock"}
    func.func private @passThroughLine_float_0(memref<1792xf32>, memref<1792xf32>, i32)
    %A_B_C_D_buffer = aie.buffer(%tile_0_2) {sym_name = "A_B_C_D_buffer"} : memref<5376xf32> 
    %A_B_C_D_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "A_B_C_D_prod_lock"}
    %A_B_C_D_con_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "A_B_C_D_con_lock"}
    func.func private @passThroughLine_float_1(memref<5376xf32>, memref<5376xf32>, i32)
    %switch_diode_buffer_debug = aie.buffer(%tile_1_2) {sym_name = "switch_diode_buffer_debug"} : memref<1792xf32> 
    %switch_diode_debug_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "switch_diode_debug_prod_lock"}
    %switch_diode_debug_con_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "switch_diode_debug_con_lock"}
    %A_B_C_D_debug_buffer = aie.buffer(%tile_1_2) {sym_name = "A_B_C_D_debug_buffer"} : memref<5376xf32> 
    %lock_1_2 = aie.lock(%tile_1_2, 2) {init = 1 : i32}
    %lock_1_2_1 = aie.lock(%tile_1_2, 3) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%switch_diode_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%switch_diode_buffer : memref<1792xf32>, 0, 1792)
      aie.use_lock(%switch_diode_con_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%A_B_C_D_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_B_C_D_buffer : memref<5376xf32>, 0, 1680)
      aie.use_lock(%A_B_C_D_con_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4:  // pred: ^bb2
      aie.use_lock(%A_B_C_D_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_B_C_D_buffer : memref<5376xf32>, 1680, 3696)
      aie.use_lock(%A_B_C_D_con_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb6:  // 2 preds: ^bb4, ^bb7
      aie.use_lock(%in_buffer_p_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_buffer_0 : memref<1120xf32>, 0, 560)
      aie.use_lock(%in_buffer_c_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%in_buffer_p_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_buffer_0 : memref<1120xf32>, 560, 560)
      aie.use_lock(%in_buffer_c_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // 2 preds: ^bb5, ^bb10
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buffer_0 : memref<7840xf32>, 0, 3920) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb10
    ^bb9:  // pred: ^bb5
      aie.end
    ^bb10:  // pred: ^bb8
      aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buffer_0 : memref<7840xf32>, 3920, 3920) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb8
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%switch_diode_debug_con_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%switch_diode_buffer_debug : memref<1792xf32>, 0, 1792) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
      aie.use_lock(%switch_diode_debug_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      aie.use_lock(%lock_1_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_B_C_D_debug_buffer : memref<5376xf32>, 0, 5376) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
      aie.use_lock(%lock_1_2, Release, 1)
      aie.next_bd ^bb1
    }
    func.func private @CT_main(memref<1120xf32>, memref<7840xf32>, i32, i32, i32, i32, i32, i32, i32, memref<1792xf32>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c560_i32 = arith.constant 560 : i32
      %c3920_i32 = arith.constant 3920 : i32
      %c280_i32 = arith.constant 280 : i32
      %c8_i32 = arith.constant 8 : i32
      %c9_i32 = arith.constant 9 : i32
      %c10_i32 = arith.constant 10 : i32
      %c11_i32 = arith.constant 11 : i32
      func.call @CT_main(%in_buffer_0, %out_buffer_0, %c560_i32, %c3920_i32, %c280_i32, %c8_i32, %c9_i32, %c10_i32, %c11_i32, %switch_diode_buffer) : (memref<1120xf32>, memref<7840xf32>, i32, i32, i32, i32, i32, i32, i32, memref<1792xf32>) -> ()
      aie.end
    } {link_with = "mainKernel.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%switch_diode_debug_prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%switch_diode_con_lock, AcquireGreaterEqual, 1)
        %c1792_i32 = arith.constant 1792 : i32
        func.call @passThroughLine_float_0(%switch_diode_buffer, %switch_diode_buffer_debug, %c1792_i32) : (memref<1792xf32>, memref<1792xf32>, i32) -> ()
        aie.use_lock(%switch_diode_debug_con_lock, Release, 1)
        aie.use_lock(%switch_diode_prod_lock, Release, 1)
        aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
        aie.use_lock(%A_B_C_D_con_lock, AcquireGreaterEqual, 2)
        %c5376_i32 = arith.constant 5376 : i32
        func.call @passThroughLine_float_1(%A_B_C_D_buffer, %A_B_C_D_debug_buffer, %c5376_i32) : (memref<5376xf32>, memref<5376xf32>, i32) -> ()
        aie.use_lock(%A_B_C_D_prod_lock, Release, 2)
        aie.use_lock(%lock_1_2_1, Release, 1)
      }
      aie.end
    } {link_with = "passThrough.o"}
    aie.packet_flow(6) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
    aie.packet_flow(7) {
      aie.packet_source<%tile_1_2, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 0>
    }
    aie.packet_flow(8) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 1>
      aie.packet_dest<%tile_0_2, DMA : 1>
    }
    aie.packet_flow(9) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    }
    memref.global "public" @in_SHM_CT_0_2_0 : memref<3472xf32>
    memref.global "public" @in_SHM_CT_0_2_1 : memref<5936xf32>
    memref.global "public" @B_CT_1_2_SHM : memref<7168xf32>
    memref.global "public" @out_CT_0_2_SHM : memref<15680xf32>
    aie.shim_dma_allocation @B_CT_1_2_SHM(S2MM, 0, 0)
    aie.shim_dma_allocation @in_SHM_CT_0_2_0(MM2S, 0, 0)
    aie.shim_dma_allocation @out_CT_0_2_SHM(S2MM, 1, 0)
    aie.shim_dma_allocation @in_SHM_CT_0_2_1(MM2S, 1, 0)
    aiex.runtime_sequence @sequence(%arg0: memref<7168xf32>, %arg1: memref<7168xf32>, %arg2: memref<2240xf32>, %arg3: memref<15680xf32>) {
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 7168][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @B_CT_1_2_SHM} : memref<7168xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 16, 7, 16][0, 112, 1, 7], packet = <pkt_type = 0, pkt_id = 6>) {id = 1 : i64, metadata = @in_SHM_CT_0_2_0} : memref<7168xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 5, 7, 48][0, 336, 1, 7], packet = <pkt_type = 0, pkt_id = 6>) {id = 2 : i64, metadata = @in_SHM_CT_0_2_0} : memref<7168xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 496][1, 11, 7, 48][0, 336, 1, 7], packet = <pkt_type = 0, pkt_id = 8>) {id = 3 : i64, metadata = @in_SHM_CT_0_2_1} : memref<7168xf32>
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 0][1, 1, 1, 15680][0, 0, 0, 1]) {id = 4 : i64, issue_token = true, metadata = @out_CT_0_2_SHM} : memref<15680xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 2240][0, 0, 0, 1], packet = <pkt_type = 0, pkt_id = 8>) {id = 5 : i64, metadata = @in_SHM_CT_0_2_1} : memref<2240xf32>
      aiex.npu.dma_wait {symbol = @B_CT_1_2_SHM}
      aiex.npu.dma_wait {symbol = @out_CT_0_2_SHM}
    }
  }
}

