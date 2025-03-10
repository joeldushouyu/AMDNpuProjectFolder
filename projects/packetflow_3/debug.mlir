module {
  aie.device(npu1_1col) {
    memref.global "public" @objFifo_in1 : memref<260xi8>
    memref.global "public" @objFifo_out1 : memref<256xi8>
    func.func private @add(memref<260xi8>, memref<256xi8>)
    func.func private @mul(memref<256xi8>, memref<256xi8>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 2, %1) {keep_pkt_header = true}
      %3 = aie.masterset(North : 4, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.masterset(DMA : 0, %0) {keep_pkt_header = true}
      %5 = aie.masterset(DMA : 2, %3) {keep_pkt_header = true}
      %6 = aie.masterset(South : 2, %2)
      %7 = aie.masterset(North : 1, %1)
      aie.packet_rules(North : 0) {
        aie.rule(31, 4, %3)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %1)
      }
      aie.packet_rules(DMA : 2) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0) {keep_pkt_header = true}
      %3 = aie.masterset(South : 0, %1)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 4, %1)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %objFifo_core02_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_core02_cons_buff_0"} : memref<260xi8> 
    %objFifo_core02_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_core02_buff_0"} : memref<256xi8> 
    %objFifo_core02_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "objFifo_core02_cons_prod_lock"}
    %objFifo_core02_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_core02_cons_cons_lock"}
    %objFifo_core02_prod_lock = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "objFifo_core02_prod_lock"}
    %objFifo_core02_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "objFifo_core02_cons_lock"}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_0_1, DMA : 2>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%mem_tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(4) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 2>
    } {keep_pkt_header = true}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%objFifo_core02_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%objFifo_core02_prod_lock, AcquireGreaterEqual, 1)
        func.call @add(%objFifo_core02_cons_buff_0, %objFifo_core02_buff_0) : (memref<260xi8>, memref<256xi8>) -> ()
        aie.use_lock(%objFifo_core02_cons_prod_lock, Release, 1)
        aie.use_lock(%objFifo_core02_cons_lock, Release, 1)
      }
      aie.end
    } {link_with = "add.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_core02_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_core02_cons_buff_0 : memref<260xi8>, 0, 260)
        aie.use_lock(%objFifo_core02_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_core02_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_core02_buff_0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
        aie.use_lock(%objFifo_core02_prod_lock, Release, 1)
      }]
      aie.end
    }
    aie.shim_dma_allocation @objFifo_in1(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<260xi8>, %arg1: memref<264xi8>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 260][0, 0, 0, 1]) {id = 0 : i64, metadata = @objFifo_in1} : memref<260xi8>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 264][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @objFifo_out1} : memref<264xi8>
      aiex.npu.dma_wait {symbol = @objFifo_out1}
    }
    aie.shim_dma_allocation @objFifo_out1(S2MM, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %objFifo_in0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<260xi8> 
      %objFifo_out_from_customer_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "objFifo_out_from_customer_buff_0"} : memref<260xi8> 
      %objFifo_in0_cons_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
      %objFifo_in0_cons_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
      %objFifo_out0_prod_lock = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
      %objFifo_out0_cons_lock = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<260xi8>)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<260xi8>)
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      %2 = aie.dma(MM2S, 2) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out_from_customer_buff_0 : memref<260xi8>, 0, 260) {packet = #aie.packet_info<pkt_type = 4, pkt_id = 2>}
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      %3 = aie.dma(S2MM, 2) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out_from_customer_buff_0 : memref<260xi8>, 0, 260)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}

