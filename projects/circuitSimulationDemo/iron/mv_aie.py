import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

from aie.dialects import memref
#given a 512x512 matrix size and 512x`1 vector


def round_to_nearest_multiple(n, multiple):
  """Rounds an integer to the nearest multiple of a given number"""
  if multiple == 0:
      return n  # Avoid division by zero
  return ((n + multiple - 1) // multiple) * multiple









def single_mat_vect_mult():
    dev = AIEDevice.npu2
    

    
    dtype_in = np.dtype[np.float32]
    dtype_out = np.dtype[np.float32]
    
    
    @device(AIEDevice.npu2)
    def device_body():
        #TODO: pass as parameters
        simulate_end_time = 4e-3
        simulate_frequency = (100e3)*20
        switch_size = 2
        diode_size = 2
        u_size = 1
        state_size = 6
        output_size = 14
        total_switch_size = 2**(switch_size+diode_size)
        
        
   
        
        
        # For each diode, the matrix is      3*(diode_number) x (state_size+u_size)
        #NOTE: the host should already add (u_size) of column concat with C_diode_impulse_sw, if not memory reordering is not possible with constant stride in dma
        _len_of_switch_diode_determine_matrix =  3* diode_size* (state_size+u_size) 
        # Iteration matrix consist of A_with_dep, B_with_dep, C_imp, C_natual, D_imp, D_natual

        # A_with_dep and B_with_dep can be combined into one matrix of size (state_size x (state_size+u_size))
        # And C_impulse_mat, D_impulse_mat can be combine into one matrix with of size (Y_size x(state_size + u_size))
        # and C_non_impulse_matrix, D_non_impulse_matrix can be combined into another matrix of size (Y_size x (state_size+u_size))
        _len_of_A_B_matrix = state_size *( state_size+u_size)
        _len_of_C_imp_D_imp = (output_size * (state_size+u_size) )
        _len_of_C_non_imp_D_non_imp = _len_of_C_imp_D_imp

        # The idea is to store ALL switch_diode_determine_matrix in one buffer?
        # Store all A_B matrix in another Buffer
        # Store all C_non_imp_D_non_imp in another buffer
        
        # Then use rest of buffer to store the U, and External switch 
        
        _len_of_switch_size = (round_to_nearest_multiple(switch_size,32)//32 )   # number of 4byte(float) use for sending external switch for each iteration
        len_of_input_for_each_iteration = u_size+ _len_of_switch_size
        
        # Tile declarations
        ShimTile = tile(0,0)
        ComputeTile_0_2 = tile(0,2)        
        # ComputeTile_0_2 = tile(0,2, allocation_scheme="bank-aware")        
        # ComputeTile_0_2 = tile(0,2, allocation_scheme="basic-sequential")
        ComputeTile_1_2 = tile(1,2)
        #TODO: profiling for if it makes a different of putting diode_matrix, A,B, C, D matrix together or not
        # allocate buffer for switch_diode_matrix
        buffer_size_of_each_switch_diode_matrix = _len_of_switch_diode_determine_matrix# round_to_nearest_multiple(_len_of_switch_diode_determine_matrix,16) # round to nearest 16 multipler(256 bit width)
        switch_diode_matrix_ty = np.ndarray[ (buffer_size_of_each_switch_diode_matrix, ), dtype_in]
        switch_diode_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=switch_diode_matrix_ty, name=f"switch_diode_buffer_{id}") for id in range(total_switch_size)
        ]
        switch_diode_prod_lock =  lock(ComputeTile_0_2, lock_id=0, init=total_switch_size, sym_name="switch_diode_prod_lock")
        switch_diode_con_lock = lock(ComputeTile_0_2, lock_id=1, init=0, sym_name="switch_diode_con_lock")


        # Debug Buffer
        switch_diode_buffer_debug_out = [
          buffer(tile=ComputeTile_1_2, datatype=switch_diode_matrix_ty, name=f"switch_diode_buffer_debug_{id}") for id in range(total_switch_size)
        ]
        switch_diode_debug_prod_lock = lock(ComputeTile_1_2, lock_id= 0,init=total_switch_size, sym_name="switch_diode_debug_prod_lock" )
        switch_diode_debug_con_lock = lock(ComputeTile_1_2, lock_id=1, init=0, sym_name="switch_diode_debug_con_lock")


        pass_through_func = external_func( "passThroughLine_float", inputs=[
          switch_diode_matrix_ty, switch_diode_matrix_ty, np.int32
        ] )

        """buffer_size_of_each_A_B_matrix = _len_of_A_B_matrix# round_to_nearest_multiple(_len_of_A_B_matrix,16)   # 4 byte each float, 8 bit each byte. Round to nearest 256 bit
        A_B_matrix_ty = np.ndarray[ (buffer_size_of_each_A_B_matrix, ), dtype_in]
        A_B_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=A_B_matrix_ty, name=f"A_B_buffer_{id}") for id in range(total_switch_size)
        ]
        A_B_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=2, init=total_switch_size)
        A_B_buffer_con_lock = lock(ComputeTile_0_2, lock_id=3, init=0)
        
        buffer_size_of_each_C_D_imp_matrix = _len_of_C_imp_D_imp#round_to_nearest_multiple(_len_of_C_imp_D_imp, 16)
        C_D_matrix_ty = np.ndarray[ (buffer_size_of_each_C_D_imp_matrix,), dtype_in ]
        C_D_imp_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=C_D_matrix_ty, name=f"C_D_imp_buffer_{id}") for id in range(total_switch_size)
        ]
        C_D_imp_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=4, init=total_switch_size)
        C_D_imp_buffer_con_lock = lock(ComputeTile_0_2, lock_id=5, init=0)
        
        C_D_non_imp_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=C_D_matrix_ty, name=f"C_D_non_imp_buffer_{id}") for id in range(total_switch_size)
        ]
        C_D_non_imp_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=6, init=total_switch_size)
        C_D_non_imp_buffer_con_lock = lock(ComputeTile_0_2, lock_id=7, init=0)
                
        # now see what are the size of it for us doing double buffer on both input and output
        # size remainig
        buffer_size_for_in_out = (15+16+16)*(1024/4) - (buffer_size_of_each_switch_diode_matrix +buffer_size_of_each_A_B_matrix+2*_len_of_C_imp_D_imp )*total_switch_size
        # define a ping pong for it?
        
        max_iteration_step =  int(round_to_nearest_multiple( buffer_size_for_in_out//(len_of_input_for_each_iteration + output_size),2)) 
        # print(max_iteration_step)

        in_data_ty = np.ndarray[ (len_of_input_for_each_iteration*(max_iteration_step//2), ), dtype_in]
        out_data_ty = np.ndarray[ (output_size*(max_iteration_step//2), ), dtype_out]
        in_buffer = [
          buffer(tile=ComputeTile_0_2, datatype=in_data_ty, name=f"in_buffer_{i}") for i in range(2)
        ]
        in_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=8, init=1)
        in_buffer_con_lock = lock(ComputeTile_0_2, lock_id=9, init=0)
        out_buffer = [
          buffer(tile=ComputeTile_0_2, datatype=out_data_ty, name=f"out_buffer_{i}") for i in range(2)
        ]
        out_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=10, init=1)
        out_buffer_con_lock = lock(ComputeTile_0_2, lock_id=11, init=0)
        """
        
        """
        #TODO: later step
        # packetflow(
        #     pkt_id=0x0,
        #     source=ShimTile,
        #     source_port=WireBundle.DMA,
        #     source_channel=0,
        #     dest=ComputeTile_0_2,
        #     dest_port=WireBundle.DMA,
        #     dest_channel=0,
        #     #keep_pkt_header=True,
        # )
        # packetflow(
        #   pkt_id=0x1,
        #   source=ShimTile,
        #   source_port=WireBundle.DMA,
        #   source_channel=1,
        #   dest=ComputeTile_0_2,
        #   dest_port=WireBundle.DMA,
        #   dest_channel=1
        # )
        # # Debug result
        # packetflow(
        #   pkt_id=0x2,
        #   source = ComputeTile_1_2,
        #   source_port=WireBundle.DMA,
        #   source_channel = 0,
        #   dest = ShimTile,
        #   dest_port= WireBundle.DMA,
        #   dest_channel=0
          
        # )
        """
        flow(ShimTile, WireBundle.DMA, 0, ComputeTile_0_2, WireBundle.DMA, 0)
        flow(ComputeTile_1_2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA,0)
        
        
        memref.global_("B_CT_1_2_SHM", T.memref( buffer_size_of_each_switch_diode_matrix*total_switch_size, T.f32() ), sym_visibility="public")
        memref.global_("A_SHM_CT_0_2", T.memref( buffer_size_of_each_switch_diode_matrix*total_switch_size, T.f32() ), sym_visibility="public")
        shim_dma_allocation("B_CT_1_2_SHM", DMAChannelDir.S2MM, 0, 0)
        shim_dma_allocation("A_SHM_CT_0_2", DMAChannelDir.MM2S, 0, 0)
        
        #TODO: CT has 2 s2mm, so how do evenly divide it for transmitting those data?        

      
        # test_buffers = [ x for x in switch_diode_buffer] + [y for y in A_B_buffer]
        @mem(ComputeTile_0_2)
        def m(block):
          
            start_block=1
            end_block =total_switch_size+start_block
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[start_block], chain=block[end_block])  
            for i in range(end_block-start_block):
                with block[i + 1]:
                    use_lock(switch_diode_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(switch_diode_buffer[i], offset=0, len=buffer_size_of_each_switch_diode_matrix)
                    use_lock(switch_diode_con_lock, LockAction.Release, value=1)
                    next_index = i + 2 if (i + 2) <= end_block else start_block
                    next_bd(block[next_index])
            with block[end_block]:
                EndOp()
                
                
            # start_block=1
            # end_block =1+start_block
            # s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[start_block], chain=block[end_block])  

            # for i in range(end_block-start_block):
            #     with block[i + 1]:
            #         use_lock(switch_diode_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            #         for k in range(total_switch_size):
            #             dma_bd(switch_diode_buffer[k], offset=0, len=buffer_size_of_each_switch_diode_matrix)
            #         use_lock(switch_diode_con_lock, LockAction.Release, value=1)
            #         next_index = i + 2 if (i + 2) <= end_block else start_block
            #         next_bd(block[next_index])
            # with block[end_block]:
            #     EndOp()
                
                
                
        @mem(ComputeTile_1_2)
        def m(block):
          start_block=1
          end_block = total_switch_size+start_block
          s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[start_block], chain=block[end_block])  
          for i in range(end_block-start_block):
              with block[i + 1]:
                  use_lock(switch_diode_debug_con_lock, LockAction.AcquireGreaterEqual, value=1)
                  dma_bd(switch_diode_buffer_debug_out[i], offset=0, len=buffer_size_of_each_switch_diode_matrix)
                  use_lock(switch_diode_debug_prod_lock, LockAction.Release, value=1)
                  next_index = i + 2 if (i + 2) <= end_block else start_block
                  next_bd(block[next_index])
          with block[end_block]:
              EndOp()
            # start_block=1
            # end_block =1+start_block
            # s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[start_block], chain=block[end_block])  

            # for i in range(end_block-start_block):
            #     with block[i + 1]:
            #         use_lock(switch_diode_debug_con_lock, LockAction.AcquireGreaterEqual, value=1)
            #         for k in range(total_switch_size):
            #             dma_bd(switch_diode_buffer_debug_out[k], offset=0, len=buffer_size_of_each_switch_diode_matrix)
            #         use_lock(switch_diode_debug_prod_lock, LockAction.Release, value=1)
            #         next_index = i + 2 if (i + 2) <= end_block else start_block
            #         next_bd(block[next_index])
            # with block[end_block]:
            #     EndOp()
                
        @core(ComputeTile_1_2, "passThrough.o")
        def core_body():
          for _ in range_(sys.maxsize):
            for k in range(total_switch_size):
                use_lock(switch_diode_debug_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                use_lock(switch_diode_con_lock, LockAction.AcquireGreaterEqual, value=1)
                pass_through_func( switch_diode_buffer[k], switch_diode_buffer_debug_out[k], constant(buffer_size_of_each_switch_diode_matrix)    )
                use_lock(switch_diode_debug_con_lock, LockAction.Release, value=1)
                use_lock(switch_diode_prod_lock, LockAction.Release, value=1)
            
            # use_lock(switch_diode_debug_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            # use_lock(switch_diode_con_lock, LockAction.AcquireGreaterEqual, value=1)
            # for k in range(total_switch_size):
            #     pass_through_func( switch_diode_buffer[k], switch_diode_buffer_debug_out[k], constant(buffer_size_of_each_switch_diode_matrix)    )
            # use_lock(switch_diode_debug_con_lock, LockAction.Release, value=1)
            # use_lock(switch_diode_prod_lock, LockAction.Release, value=1)
        input_size = buffer_size_of_each_switch_diode_matrix*total_switch_size
        @runtime_sequence(np.ndarray[(input_size, ), dtype_in], np.ndarray[(input_size, ), dtype_out]  )
        def sequence(A,B):
            npu_dma_memcpy_nd(
                metadata="A_SHM_CT_0_2",
                bd_id=1,
                mem=A, offsets=[0,0,0,0], sizes= [1,1,1,buffer_size_of_each_switch_diode_matrix*total_switch_size],
                strides=[0,0,0,1]
            )

            npu_dma_memcpy_nd(metadata="B_CT_1_2_SHM", bd_id=0, mem=B, offsets=[0,0,0,0],sizes= [1,1,1,buffer_size_of_each_switch_diode_matrix*total_switch_size],strides=[0,0,0,1])

            npu_dma_wait("B_CT_1_2_SHM")
with mlir_mod_ctx() as ctx:
    single_mat_vect_mult()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
