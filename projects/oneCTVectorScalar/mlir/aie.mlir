// This is the aie code for doing vector scalar mutliplcation
// vector is int16 of 4096 size
// scalar is int32 of 1 size


module{
    aie.device(npu1_1col){
        func.func private @vector_scalar_mul_int16_vector(memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32)

        // setup the cities of it 
        %tile_0_2 = aie.tile(0,2)
        %shim_noc_tile_0_0 = aie.tile(0,0)

        // 2 input channel and 1 output channel
        // thus needs 6 locks for it


        %out_buff_0 = aie.buffer(%tile_0_2) {sys_name = "out_buff_0"} : memref<1024xi16>
        %out_buff_1 = aie.buffer(%tile_0_2) {sys_name = "out_buff_1"} : memref<1024xi16>
        %out_prod_lock = aie.lock(%tile_0_2,0){init=2: i32, sys_name="out_prod_lock"}
        %out_con_lock = aie.lock(%tile_0_2, 1){init=0: i32, sys_name="out_con_lock"}

        %input_vector_buffer_0 = aie.buffer(%tile_0_2) {sys_name = "input_vector_buffer_0"}: memref<1024xi16>
        %input_vector_buffer_1 = aie.buffer(%tile_0_2) {sys_name = "input_vector_buffer_1"}: memref<1024xi16>
        %input_vector_prod_lock = aie.lock(%tile_0_2,2) {init=2: i32, sys_name = "input_vector_prod_lock"}
        %input_vector_con_lock = aie.lock(%tile_0_2,3) {init=0:i32, sys_name = "input_vector_con_lock"}


        %input_scalar_buffer_0 = aie.buffer(%tile_0_2) {sys_name = "input_scalar_buffer_0"} : memref< 1xi32>
        %input_scalar_buffer_1 = aie.buffer(%tile_0_2) {sys_name = "input_scalar_buffer_1"} : memref< 1xi32>
        %input_scalar_prod_lock = aie.lock(%tile_0_2, 4){init=2:i32, sys_name="input_scalar_prod_lock"}
        %input_scalar_con_lock = aie.lock(%tile_0_2, 5){init=0:i32, sys_name="input_scalar_con_lock"}

        // circuit flow between DMA
        aie.flow(%shim_noc_tile_0_0, DMA:0, %tile_0_2, DMA: 0)
        aie.flow(%shim_noc_tile_0_0, DMA:1, %tile_0_2, DMA:1)
        aie.flow(%tile_0_2, DMA:0, %shim_noc_tile_0_0, DMA:0)


        //TODO: simplify the following using if statement? 

        // setup core logic
        %core_0_2 = aie.core(%tile_0_2){
            %c0 = arith.constant 0 :index
            %constant_2 = arith.constant 2: index
            %constant_4 = arith.constant 4: index
            %c1024_i32 = arith.constant 1024 : i32
            %c1024_i32_11 = arith.constant 1024 : i32
            %c9223372036854775806 = arith.constant 9223372036854775806 : index
            // essentially an forever loop
            scf.for %arg0= %c0 to %c9223372036854775806 step %constant_2{
                

                // use scalar 0 version
                aie.use_lock(%input_scalar_con_lock, AcquireGreaterEqual, 1)
                // do it in step of 1024, since total buffer is 4096, so need to repeat 4 times
                // but since ping pong buffer, so in step of 2
                scf.for %arg1=%c0 to %constant_4 step %constant_2{
                    
                    aie.use_lock(%input_vector_con_lock, AcquireGreaterEqual, 1)
                    aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)

                    func.call @vector_scalar_mul_int16_vector(%input_vector_buffer_0, %out_buff_0, %input_scalar_buffer_0, %c1024_i32) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
                    aie.use_lock(%input_vector_prod_lock, Release, 1)
                    aie.use_lock(%out_con_lock, Release, 1)

                    
                    aie.use_lock(%input_vector_con_lock, AcquireGreaterEqual, 1)
                    aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
                    func.call @vector_scalar_mul_int16_vector(%input_vector_buffer_1, %out_buff_1, %input_scalar_buffer_0, %c1024_i32_11): (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
                    aie.use_lock(%input_vector_prod_lock, Release, 1)
                    aie.use_lock(%out_con_lock, Release, 1)


                }

                aie.use_lock(%input_scalar_prod_lock, Release, 1)


                // use scalar 1 version
                aie.use_lock(%input_scalar_con_lock, AcquireGreaterEqual, 1)
                // do it in step of 1024, since total buffer is 4096, so need to repeat 4 times
                // but since ping pong buffer, so in step of 2
                scf.for %arg1=%c0 to %constant_4 step %constant_2{
                    
                    aie.use_lock(%input_vector_con_lock, AcquireGreaterEqual, 1)
                    aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
                    func.call @vector_scalar_mul_int16_vector(%input_vector_buffer_0, %out_buff_0, %input_scalar_buffer_1, %c1024_i32): (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
                    aie.use_lock(%input_vector_prod_lock, Release, 1)
                    aie.use_lock(%out_con_lock, Release, 1)

                    
                    aie.use_lock(%input_vector_con_lock, AcquireGreaterEqual, 1)
                    aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
                    func.call @vector_scalar_mul_int16_vector(%input_vector_buffer_1, %out_buff_1, %input_scalar_buffer_1, %c1024_i32_11): (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
                    aie.use_lock(%input_vector_prod_lock, Release, 1)
                    aie.use_lock(%out_con_lock, Release, 1)


                }

                aie.use_lock(%input_scalar_prod_lock, Release, 1)

            }
            aie.end

        } {link_with = "scale.o"}
        // set memory logic
        

        // setting up the runtime stuff, interfacing with host
        aiex.runtime_sequence @sequence(%arg0: memref<4096xi16>, %arg1 : memref<1xi32>, %arg2 : memref<4096xi16>){
            %0 = aiex.dma_configure_task_for @in{
                aie.dma_bd(%arg0: memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size =1, stride = 0>, <size = 1, stride = 0>, <size= 4096, stride = 1>])
                aie.end
            }{issue_token = true}
            %1 = aiex.dma_configure_task_for @infactor{
                aie.dma_bd(%arg1: memref<1xi32>, 0, 1, [<size =1, stride = 0>, <size = 1, stride = 0>, <size =1, stride = 0>, <size = 1, stride = 1>])
                aie.end
            }{issue_token = true}

            %2 = aiex.dma_configure_task_for @out{
                aie.dma_bd(%arg2: memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>,  <size = 4096, stride = 1>])
                aie.end
            }{issue_token = true}

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
        %mem_0_2 = aie.mem(%tile_0_2){
                %vector_in = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
            ^bb1:
                aie.use_lock(%input_vector_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%input_vector_buffer_0: memref<1024xi16>, 0, 1024)
                aie.use_lock(%input_vector_con_lock, Release, 1)
                aie.next_bd ^bb2
            ^bb2:
                aie.use_lock(%input_vector_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%input_vector_buffer_1: memref<1024xi16>, 0, 1024)
                aie.use_lock(%input_vector_con_lock, Release, 1)
                aie.next_bd ^bb1
            ^bb3:
                %scalar_in = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
            ^bb4:
                aie.use_lock(%input_scalar_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%input_scalar_buffer_0: memref<1xi32>, 0, 1)
                aie.use_lock(%input_scalar_con_lock, Release, 1)
                aie.next_bd ^bb5 
            ^bb5:
                aie.use_lock(%input_scalar_prod_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%input_scalar_buffer_1: memref<1xi32>, 0, 1)
                aie.use_lock(%input_scalar_con_lock, Release, 1)            
                aie.next_bd ^bb4
            ^bb6:
                %vector_out = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
            ^bb7:
                aie.use_lock(%out_con_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%out_buff_0: memref<1024xi16>, 0,1024)
                aie.use_lock(%out_prod_lock, Release, 1)
                aie.next_bd ^bb8
            ^bb8:
                aie.use_lock(%out_con_lock, AcquireGreaterEqual, 1)
                aie.dma_bd(%out_buff_1: memref<1024xi16>, 0,1024)
                aie.use_lock(%out_prod_lock, Release, 1)            
                aie.next_bd ^bb7
            ^bb9:
                aie.end  



        }

    }




}