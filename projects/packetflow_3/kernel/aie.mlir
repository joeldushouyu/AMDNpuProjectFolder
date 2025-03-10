module {
    aie.device(npu1_1col) {
        memref.global "public" @objFifo_in1 : memref<260xi8>
        memref.global "public" @objFifo_out1 : memref<256xi8>

        // func.func private @add(memref<260xi8>, memref<256xi8>) //option 1
        func.func private @add(memref<276xi8>, memref<256xi8>) //option 2
        func.func private @mul(memref<276xi8>, memref<256xi8>)

        %ShimTile_0_0 = aie.tile(0, 0)
        %MemTile_0_1 = aie.tile(0, 1)
        %CT_0_2 = aie.tile(0, 2)


        // core_0_2:
        %objFifo_core02_cons_buff_0 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_cons_buff_0"} : memref<276xi8>
        %objFifo_core02_buff_0 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_buff_0"} : memref<256xi8>
        
        // /// allocating wasted buffer for seeing how CT assembly will change with it
        // %objFifo_core02_waste = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_waste"} : memref<16x1024xi8>
        // %objFifo_core02_waste2 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_waste2"} : memref<16x1024xi8>
        // %objFifo_core02_waste3 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_waste3"} : memref<16x1024xi8>
        // %objFifo_core02_waste4 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_waste4"} : memref<14x1024xi8>
        // %objFifo_core02_waste5 = aie.buffer(%CT_0_2) {sym_name = "objFifo_core02_waste5"} : memref<492xi8>
        %objFifo_core02_cons_prod_lock = aie.lock(%CT_0_2, 0) {init = 1 : i32, sym_name = "objFifo_core02_cons_prod_lock"}
        %objFifo_core02_cons_cons_lock = aie.lock(%CT_0_2, 1) {init = 0 : i32, sym_name = "objFifo_core02_cons_cons_lock"}
        %objFifo_core02_prod_lock = aie.lock(%CT_0_2, 4) {init = 1 : i32, sym_name = "objFifo_core02_prod_lock"}
        %objFifo_core02_cons_lock = aie.lock(%CT_0_2, 5) {init = 0 : i32, sym_name = "objFifo_core02_cons_lock"}



        // Add
        aie.packet_flow(0) {
            aie.packet_source<%ShimTile_0_0, DMA : 0>
            aie.packet_dest<%MemTile_0_1, DMA : 0>
        } {keep_pkt_header = true} 


        aie.packet_flow(2) {
            aie.packet_source<%MemTile_0_1, DMA : 2>
            aie.packet_dest<%ShimTile_0_0, DMA : 0>
        }{keep_pkt_header = true}  // note: I purposefully keep the transmision header when transmit back to host to inspect

        aie.packet_flow(0) {
            aie.packet_source<%MemTile_0_1, DMA : 0>
            aie.packet_dest<%CT_0_2, DMA : 0>
        }{keep_pkt_header  = true} // still want to keep the original header from host through shmtile->Memory tile
            // use the information inside the package header to decide the kernel operation in CT
        aie.packet_flow(4) {
            aie.packet_source<%CT_0_2, DMA : 0>
            aie.packet_dest<%MemTile_0_1, DMA : 2>
        }{keep_pkt_header = true} // want to still keep the packet header at MT side

        %core_0_2 = aie.core(%CT_0_2) {
            %c0 = arith.constant 0 : index
            %c4294967295 = arith.constant 4294967295 : index
            %c1 = arith.constant 1 : index
            scf.for %arg0 = %c0 to %c4294967295 step %c1 {
                aie.use_lock(%objFifo_core02_cons_cons_lock, AcquireGreaterEqual, 1)
                aie.use_lock(%objFifo_core02_prod_lock, AcquireGreaterEqual, 1)


                //retrieve the 16 byte( last 4 byte has the packet header information)
                
                // recall package is in LSB ordering
                // look at the 11:8 (4 bit)
                %idx_13 = arith.constant 13 : index
                %second_byte =memref.load %objFifo_core02_cons_buff_0[%idx_13] : memref<276xi8>
                %c15_i8= arith.constant 15 : i8
                %c0_i8= arith.constant 0 : i8
                %bottom_4 = arith.andi %second_byte, %c15_i8 : i8

                %cmp = arith.cmpi eq, %bottom_4, %c0_i8 : i8
                scf.if %cmp {
                    // Handle case where bottom 4 bits are 0
                    // do add operation

                    // the packet header are in LSB order
                    // subview help to ignore the 4 byte packet header
                    // // option 1:
                    // func.call @add(%objFifo_core02_cons_buff_0,  %objFifo_core02_buff_0) : (memref<260xi8>, memref<256xi8>) -> ()
                    
                    //option 2: since the add is doing a 16 8byte per lane, allocate extra 16 byte in byffer
                    // // to a recast here?
                    // %sub= memref.subview %objFifo_core02_cons_buff_0[16][256][1] : memref<276xi8> to memref<256xi8>

                    func.call @add(%objFifo_core02_cons_buff_0,  %objFifo_core02_buff_0) : (memref<276xi8>, memref<256xi8>) -> ()
                } else {
                    func.call @mul(%objFifo_core02_cons_buff_0,  %objFifo_core02_buff_0) : (memref<276xi8>, memref<256xi8>) -> ()
                }

                aie.use_lock(%objFifo_core02_cons_prod_lock, Release, 1)
                aie.use_lock(%objFifo_core02_cons_lock, Release, 1)
            }
            aie.end
        } {link_with = "add.o"}

        %mem_0_2 = aie.mem(%CT_0_2) {
            %0 = aie.dma(S2MM, 0) [
                {
                    aie.use_lock(%objFifo_core02_cons_prod_lock, AcquireGreaterEqual, 1)
                    // //option 1:
                    // aie.dma_bd(%objFifo_core02_cons_buff_0 : memref<260xi8>, 0,260)
                    //option 2:
                    aie.dma_bd(%objFifo_core02_cons_buff_0 : memref<276xi8>, 12,260) // waster 12 byte of free space in this option
                    aie.use_lock(%objFifo_core02_cons_cons_lock, Release, 1)
                }
            ]
            %1 = aie.dma(MM2S, 0) [{
                aie.use_lock(%objFifo_core02_cons_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_core02_buff_0 : memref<256xi8>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
                aie.use_lock(%objFifo_core02_prod_lock, Release, 1)
            }]
            aie.end
        }


        aie.shim_dma_allocation @objFifo_in1(MM2S, 0, 0)

        aiex.runtime_sequence(%arg0: memref<260xi8>, %arg2: memref<264xi8>) {
            aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 260][0, 0, 0, 1]) {id = 0 : i64, metadata = @objFifo_in1} : memref<260xi8>
            aiex.npu.dma_memcpy_nd (0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 264][0, 0, 0, 1]) {id = 2 : i64, metadata = @objFifo_out1, issue_token = true} : memref<264xi8>
            aiex.npu.dma_wait { symbol = @objFifo_out1 }
        }        
        aie.shim_dma_allocation @objFifo_out1(S2MM, 0, 0)

        %memtile_dma_0_1 = aie.memtile_dma(%MemTile_0_1) {
            // add
            %objFifo_in0_cons_buff_0 = aie.buffer(%MemTile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<260xi8>
            %objFifo_out_from_customer_buff_0 = aie.buffer(%MemTile_0_1) {sym_name = "objFifo_out_from_customer_buff_0"} : memref<260xi8>
            // %objFifo_out_to_ddr = aie.buffer(%MemTile_0_1) {sym_name = "objFifo_out_to_ddr"} : memref<260xi8>

            %objFifo_in0_cons_prod_lock = aie.lock(%MemTile_0_1, 0) {init = 1 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
            %objFifo_in0_cons_cons_lock = aie.lock(%MemTile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
            %objFifo_out0_prod_lock = aie.lock(%MemTile_0_1, 4) {init = 1 : i32, sym_name = "objFifo_out0_prod_lock"}
            %objFifo_out0_cons_lock = aie.lock(%MemTile_0_1, 5) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}

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
                aie.dma_bd(%objFifo_out_from_customer_buff_0 : memref<260xi8>,0,260) {packet = #aie.packet_info<pkt_type = 4, pkt_id = 2>}
                aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
            }]
            %3 = aie.dma(S2MM, 2) [{
                aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%objFifo_out_from_customer_buff_0 : memref<260xi8>, 0, 260) // 4 byte for the header
                aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
            }]

            aie.end
        }
    }
}
