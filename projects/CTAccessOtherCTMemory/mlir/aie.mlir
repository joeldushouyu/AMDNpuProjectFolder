// this is the code for testing CT acccess other CT

// CT 1,3 (column 2, row 2 of CT) sould be able to access its top, bottom and left neighbor CT
// example this CT reads 256xi8 values from neighbor CT, sum them together and return the result back to host
module {
  aie.device(npu1_4col) {



        memref.global "public" @objFifo_in1 : memref<8xi8>
        memref.global "public" @objFifo_out1 : memref<8xi8>
        // setup the cities of it 
        // %shim_noc_tile_0_0 = aie.tile(0, 0)
        %shim_noc_tile_1_0 = aie.tile(1, 0)
        // %shim_noc_tile_2_0 = aie.tile(2, 0)
        // %shim_noc_tile_3_0 = aie.tile(3, 0)

        
        %tile_1_3 = aie.tile(0,3)
        // %tile_0_3 = aie.tile(0,3) // the left neighbor


        // %tile_0_3_buffer_out_0 = aie.buffer(%tile_0_3){ sys_name = "tile_0_3_buffer_out_0"} : memref<8xi8>
        // %tile_0_3_buffer_out_1 = aie.buffer(%tile_0_3){ sys_name = "tile_0_3_buffer_out_1"} : memref<8xi8>
        // %tile_0_3_buffer_prod_lock = aie.lock(%tile_0_3,0) {init=2: i32, sys_name="tile_0_3_buffer_prod_lock"}
        // %tile_0_3_buffer_con_lock = aie.lock(%tile_0_3, 1) {init=0: i32, sys_name="tile_0_3_buffer_con_lock"}

        %tile_1_3_in_buffer_0 = aie.buffer(%tile_1_3){ sys_name= "tile_1_3_in_buffer_0"}: memref<8xi8>
        %tile_1_3_in_buffer_1 = aie.buffer(%tile_1_3) {sys_name = "tile_1_3_in_buffer_1"} : memref<8xi8>
        %tile_1_3_out_buffer_0 = aie.buffer(%tile_1_3){sys_name = "tile_1_3_out_buffer_0"}: memref<8xi8>
        %tile_1_3_out_buffer_1 = aie.buffer(%tile_1_3){sys_name = "tile_1_3_out_buffer_1"}: memref<8xi8>
        %tile_1_3_in_buffer_prod_lock = aie.lock(%tile_1_3, 0){ init=1: i32, sys_name = "tile_1_3_in_buffer_prod_lock" }
        %tile_1_3_in_buffer_con_lock = aie.lock(%tile_1_3, 1){init=0: i32, sys_name= "tile_1_3_in_buffer_con_lock"}
        %tile_1_3_out_buffer_prod_lock = aie.lock(%tile_1_3, 2){init=1: i32, sys_name= "tile_1_3_out_buffer_prod_lock"}
        %tile_1_3_out_buffer_con_lock = aie.lock(%tile_1_3, 3){init = 0:i32, sys_name = "tile_1_3_out_buffer_con_lock"}

        // flow of DMA get vector to and back form tile 1,3 
        aie.flow(%shim_noc_tile_1_0, DMA:0, %tile_1_3, DMA:0)
        aie.flow(%tile_1_3, DMA:0, %shim_noc_tile_1_0, DMA:0)

         
        // %core_0_3 = aie.core(tile_0_3){
        //     %c0 = arith.constant 0 : index
        //     %c4294967295 = arith.constant 4294967295 : index
        //     %c8 = arith.constant 8: index
        //     %c1 = arith.constant 1 : index

        //     %constant_1 = arith.constant 1: i8
        //     %value = arith.constant 1: i8
        //     scf.for %arg0 = %c0 to %c4294967295 step %c1 {
                
        //         aie.use_lock(%tile_0_3_buffer_prod_lock, AcquireGreaterEqual, 1)
        //         scf.for %arg_id_1 = %c0 to %c8 step %c1 {
        //             // write whatever %value into it 
        //             memref.store %value, %tile_0_3_buffer_out_0[%arg_id_1]: memref<1xi8>
        //         }
        //         aie.use_lock(tile_0_3_buffer_con_lock, Release, 1)


        //         %value  = arith.addi %value, %constant_1 : i8
        //         aie.use_lock(%tile_0_3_buffer_prod_lock, AcquireGreaterEqual, 1)
        //         // do it twice for ping ping buffer
        //         scf.for %arg_1 = %c0 to %c8 step c1{
        //             memref.store %value, %tile_0_3_buffer_out_1[%arg_id_1]: memref<1xi8>
        //         }
        //         aie.use_lock(tile_0_3_buffer_con_lock, Release, 1)
        //     }

        // }

        %core_1_3 = aie.core(%tile_1_3){
            // for not, just doing add ing by 1 for first step test
            %c0 = arith.constant 0 :index
            %c9223372036854775806 = arith.constant 9223372036854775806 : index
            %c1 = arith.constant 1: index            
            %c8 = arith.constant 8: index

            %c1_i8 = arith.constant 1: i8
            // scf.for %arg0=%c0 to %c9223372036854775806 step %c1{
                // for now, just do add 1 to 
                aie.use_lock(%tile_1_3_in_buffer_con_lock, AcquireGreaterEqual, 1)
                aie.use_lock(%tile_1_3_out_buffer_prod_lock, AcquireGreaterEqual, 1)
                scf.for %arg_1=%c0 to %c8 step %c1{
                    %temp_val = memref.load %tile_1_3_in_buffer_0[%arg_1] : memref<8xi8>
                    %cal_val = arith.addi %temp_val, %c1_i8 : i8
                    memref.store %cal_val, %tile_1_3_out_buffer_0[%arg_1] :memref<8xi8>
                }
                aie.use_lock(%tile_1_3_in_buffer_prod_lock, Release, 1) 
                aie.use_lock(%tile_1_3_out_buffer_con_lock, Release, 1)

                // //pong buffer
                // aie.use_lock(%tile_1_3_in_buffer_con_lock, AcquireGreaterEqual, 1)
                // aie.use_lock(%tile_1_3_out_buffer_prod_lock, AcquireGreaterEqual, 1)
                // scf.for %arg_1=%c0 to %c8 step %c1{
                //     %temp_val = memref.load %tile_1_3_in_buffer_1[%arg_1] : memref<8xi8>
                //     %cal_val = arith.addi %temp_val, %c1_i8 : i8
                //     memref.store %cal_val, %tile_1_3_out_buffer_1[%arg_1] :memref<8xi8>
                // }C_L1L2_3_3_prod_lock
            // } 
            aie.end    
        }
        %mem_1_3 = aie.mem(%tile_1_3){
            %value_in = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
            ^bb1:
                aie.use_lock(%tile_1_3_in_buffer_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd( %tile_1_3_in_buffer_0: memref<8xi8>, 0, 8 )
                aie.use_lock(%tile_1_3_in_buffer_con_lock, Release, 1)
                aie.next_bd ^bb1
            // ^bb2:
            //     aie.use_lock(%tile_1_3_in_buffer_prod_lock, AcquireGreaterEqual, 1)
            //     aie.dma_bd( %tile_1_3_in_buffer_1: memref<8xi8>, 0, 8 )
            //     aie.use_lock(%tile_1_3_in_buffer_con_lock, Release, 1)
            //     aie.next_bd ^bb1
            ^bb3:
                %value_out = aie.dma_start( MM2S, 0, ^bb4, ^bb6)
            ^bb4:
                aie.use_lock(%tile_1_3_out_buffer_con_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%tile_1_3_out_buffer_0: memref<8xi8>, 0,8 )
                aie.use_lock(%tile_1_3_out_buffer_prod_lock, Release, 1)
                aie.next_bd ^bb4
            // ^bb5:
            //     aie.use_lock(%tile_1_3_out_buffer_con_lock, AcquireGreaterEqual, 1)
            //     aie.dma_bd(%tile_1_3_out_buffer_1: memref<8xi8>, 0,8 )
            //     aie.use_lock(%tile_1_3_out_buffer_prod_lock, Release, 1)
            //     aie.next_bd ^bb4
            ^bb6:
                aie.end
        }




        aiex.runtime_sequence(%arg0: memref<8xi8>, %arg2: memref<8xi8>) {
            aiex.npu.dma_memcpy_nd (0, 1, %arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @objFifo_in1} : memref<8xi8>
            aiex.npu.dma_memcpy_nd (0, 1, %arg2[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 2 : i64, metadata = @objFifo_out1, issue_token = true} : memref<8xi8>
            // %0 = aiex.dma_configure_task_for @objFifo_in1{
            //     aie.dma_bd(%arg0: memref<8xi8>, 0, 8, [<size = 1, stride = 0>, <size =1, stride = 0>, <size = 1, stride = 0>, <size= 8, stride = 1>])
            //     aie.end
            // }{issue_token = true}
            // %1 = aiex.dma_configure_task_for @objFifo_out1{
            //     aie.dma_bd(%arg2: memref<8xi8>, 0, 8, [<size =1, stride = 0>, <size = 1, stride = 0>, <size =1, stride = 0>, <size = 8, stride = 1>])
            //     aie.end
            // }{issue_token = true}
            // aiex.dma_start_task(%0)
            // aiex.dma_start_task(%1)
            // aiex.dma_await_task(%0)
            // aiex.dma_await_task(%1)            
        }


        aie.shim_dma_allocation @objFifo_in1(MM2S, 0,1)
        aie.shim_dma_allocation @objFifo_out1(S2MM, 0, 1)
            
  }
}        // %shim_noc_tile_0_0 = aie.tile(0, 0)