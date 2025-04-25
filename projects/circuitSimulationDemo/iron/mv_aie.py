import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
import math

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
from aie.dialects._aie_ops_gen import buffer as buffer_raw
from aie.helpers.util import try_convert_np_type_to_mlir_type

# def round_to_nearest_multiple(n, multiple):
#   """Rounds an integer to the nearest multiple of a given number"""
#   if multiple == 0:
#       return n  # Avoid division by zero
#   return ((n + multiple - 1) // multiple) * multiple




def custom_floor(x, multiplier):
  return math.floor(x / multiplier) * multiplier

def custom_ceil(x, multiplier):
  return math.ceil(x / multiplier) * multiplier





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
        
        _len_of_switch_size = (custom_ceil(switch_size,32)//32 )   # number of 4byte(float) use for sending external switch for each iteration
        len_of_input_for_each_iteration = u_size+ _len_of_switch_size
        
        # Tile declarations
        ShimTile = tile(0,0)
        # ComputeTile_0_2 = tile(0,2)        
        # ComputeTile_0_2 = tile(0,2, allocation_scheme="bank-aware")        
        ComputeTile_0_2 = tile(0,2, allocation_scheme="basic-sequential")
        ComputeTile_1_2 = tile(1,2)
        #TODO: profiling for if it makes a different of putting diode_matrix, A,B, C, D matrix together or not
        # allocate buffer for switch_diode_matrix
        
        
        buffer_size_of_A_B_matrix = _len_of_A_B_matrix* total_switch_size# round_to_nearest_multiple(_len_of_A_B_matrix,16)   # 4 byte each float, 8 bit each byte. Round to nearest 256 bit
        buffer_size_of_switch_diode = _len_of_switch_diode_determine_matrix*total_switch_size
        buffer_size_of_C_D_imp_non = _len_of_C_imp_D_imp*total_switch_size#round_to_nearest_multiple(_len_of_C_imp_D_imp, 16)
        
        
        buffer_size_for_in_out = ((63)*(1024))//4 - (buffer_size_of_switch_diode + buffer_size_of_A_B_matrix + buffer_size_of_C_D_imp_non +buffer_size_of_C_D_imp_non )
        # define a ping pong for it?
        
        _max_iteration_step = int(custom_floor( buffer_size_for_in_out//(len_of_input_for_each_iteration + output_size),2)) #TODO: round down instead?
        iteration_step_per_buffer = _max_iteration_step //2
        buffer_size_of_in_ping_pong = len_of_input_for_each_iteration*(iteration_step_per_buffer)
        buffer_size_of_out_ping_pong = output_size*(iteration_step_per_buffer)
        
        in_data_ty = np.ndarray[ (buffer_size_of_in_ping_pong*2, ), dtype_in]
        out_data_ty = np.ndarray[ (buffer_size_of_out_ping_pong*2, ), dtype_out]

        
        #NOTE: mem_bank flag seem not working anymore after Tile() is configure to basic-sequential address mode
        in_buffer = [
          buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(in_data_ty), sym_name=f"in_buffer_{0}", address=1024), # 1024 offset, reserve for stack

        ]
        in_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=8, init=2)
        in_buffer_con_lock = lock(ComputeTile_0_2, lock_id=9, init=0)
        
        out_buffer_address = (64*1024) - (buffer_size_of_out_ping_pong*2*4)
        out_buffer = [
            buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(out_data_ty), sym_name=f"out_buffer_{0}", address=out_buffer_address ), # 
        ]        
        out_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=10, init=2)
        out_buffer_con_lock = lock(ComputeTile_0_2, lock_id=11, init=0)
                
        
        
        accum_float_value_func = external_func("accum_float_value", inputs=[
            in_data_ty, out_data_ty,  
            np.int32, np.int32,
            np.int32, np.int32, np.int32
        ])
        

        switch_diode_matrix_ty = np.ndarray[ (buffer_size_of_switch_diode, ), dtype_in]
        switch_diode_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=switch_diode_matrix_ty, name=f"switch_diode_buffer") 
        ]
        switch_diode_prod_lock =  lock(ComputeTile_0_2, lock_id=0, init=1, sym_name="switch_diode_prod_lock")
        switch_diode_con_lock = lock(ComputeTile_0_2, lock_id=1, init=0, sym_name="switch_diode_con_lock")


        pass_through_float_diode_matrix = external_func( "passThroughLine_float_0", inputs=[
          switch_diode_matrix_ty, switch_diode_matrix_ty, np.int32
        ] )

 
        A_B_matrix_ty = np.ndarray[ (buffer_size_of_A_B_matrix, ), dtype_in]
        A_B_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=A_B_matrix_ty, name=f"A_B_buffer") 
        ]
        A_B_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=2, init=1)
        A_B_buffer_con_lock = lock(ComputeTile_0_2, lock_id=3, init=0)

        pass_through_float_A_B_matrix = external_func( "passThroughLine_float_1", inputs=[
          A_B_matrix_ty, A_B_matrix_ty, np.int32
        ] )


        C_D_matrix_ty = np.ndarray[ (buffer_size_of_C_D_imp_non,), dtype_in ]
        C_D_imp_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=C_D_matrix_ty, name=f"C_D_imp_buffer")
        ]
        C_D_imp_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=4, init=1)
        C_D_imp_buffer_con_lock = lock(ComputeTile_0_2, lock_id=5, init=0)

        pass_through_float_C_D_matrix = external_func( "passThroughLine_float_2", inputs=[
          C_D_matrix_ty, C_D_matrix_ty, np.int32
        ] )
        
        
        C_D_non_imp_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=C_D_matrix_ty, name=f"C_D_non_imp_buffer") 
        ]
        C_D_non_imp_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=6, init=1)
        C_D_non_imp_buffer_con_lock = lock(ComputeTile_0_2, lock_id=7, init=0)
                

        # Debug Buffer
        switch_diode_buffer_debug_out = [
          buffer(tile=ComputeTile_1_2, datatype=switch_diode_matrix_ty, name=f"switch_diode_buffer_debug") 
        ]
        switch_diode_debug_prod_lock = lock(ComputeTile_1_2, lock_id= 0,init=1, sym_name="switch_diode_debug_prod_lock" )
        switch_diode_debug_con_lock = lock(ComputeTile_1_2, lock_id=1, init=0, sym_name="switch_diode_debug_con_lock")

        A_B_debug_buffer = [
            buffer(tile=ComputeTile_1_2, datatype=A_B_matrix_ty, name="A_B_debug_Buffer")
        ]
        A_B_debug_buffer_prod_lock = lock(ComputeTile_1_2, lock_id=2, init=1)
        A_B_debug_buffer_con_lock  = lock(ComputeTile_1_2, lock_id=3, init=0)
        
        C_D_debug_imp_buffer = [
            buffer(tile=ComputeTile_1_2, datatype=C_D_matrix_ty, name="C_D_debug_imp_buffer")
        ]
        C_D_debug_imp_buffer_prod_lock = lock(ComputeTile_1_2, lock_id=4, init=1)
        C_D_debug_imp_buffer_con_lock = lock(ComputeTile_1_2, lock_id=5, init=0)
        
        C_D_debug_non_imp_buffer = [
            buffer(tile=ComputeTile_1_2, datatype=C_D_matrix_ty, name="C_D_debug_non_imp_buffer")
        ]
        C_D_debug_non_imp_buffer_prod_lock = lock(ComputeTile_1_2, lock_id=6, init=1)
        C_D_debug_non_imp_buffer_con_lock = lock(ComputeTile_1_2, lock_id=7, init=0)
        
        
        
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

        
        #TODO: Balance workload for to port?
        
        @mem(ComputeTile_0_2)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[5])  
            with block[1]:
                use_lock(switch_diode_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(switch_diode_buffer[0], offset=0, len=buffer_size_of_switch_diode)
                use_lock(switch_diode_con_lock, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(A_B_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)    
                dma_bd(A_B_buffer[0], offset=0, len=buffer_size_of_A_B_matrix)
                use_lock(A_B_buffer_con_lock, LockAction.Release, value=1)
                next_bd(block[3])
            with block[3]:
                use_lock(C_D_imp_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(C_D_imp_buffer[0], offset=0, len=buffer_size_of_C_D_imp_non)
                use_lock(C_D_imp_buffer_con_lock, LockAction.Release, value=1)
                next_bd(block[4])
            with block[4]:
                use_lock(C_D_non_imp_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(C_D_non_imp_buffer[0], offset=0, len=buffer_size_of_C_D_imp_non)
                use_lock(C_D_non_imp_buffer_con_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[5]:
                s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[8])
            with block[6]:
                use_lock(in_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(in_buffer[0], offset=0, len=buffer_size_of_in_ping_pong)
                use_lock(in_buffer_con_lock, LockAction.Release, value=1)
                next_bd(block[7])
            with block[7]:
                use_lock(in_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(in_buffer[0], offset=buffer_size_of_in_ping_pong, len=buffer_size_of_in_ping_pong)
                use_lock(in_buffer_con_lock, LockAction.Release, value=1)
                next_bd(block[6])
            with block[8]: 
                s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[9], chain=block[11])
            with block[9]:
                use_lock(out_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(out_buffer[0], offset=0, len= buffer_size_of_out_ping_pong)
                use_lock(out_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[10])
            with block[10]:
                use_lock(out_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(out_buffer[0], offset=buffer_size_of_out_ping_pong, len = buffer_size_of_out_ping_pong)
                use_lock(out_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[9])
            with block[11]:
                EndOp()
        """

        
        def with_block_unroll(block, chain):
            for idx, prod_locks, buf, off, len, con_locks, nxt_idx in chain:
                with block[idx]:
                    for p_lock in prod_locks:
                        use_lock(p_lock, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd( buf, offset=off, len=len) # only allow one dma_buffers in each with block
                    for c_lock in con_locks:
                        use_lock(c_lock, LockAction.Release, value=1)
                    next_bd(block[nxt_idx])
        def handle_dma_sequences(block):

            # block_idx, acqire_locks, buffer, buffer_offset, buffer_len, release_locks, next_idx      
            chain0 = [
                (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, buffer_size_of_switch_diode, [switch_diode_con_lock], 2 ),
                (2, [A_B_buffer_prod_lock],   A_B_buffer[0], 0, buffer_size_of_A_B_matrix, [A_B_buffer_con_lock], 3   ),
                (3, [C_D_imp_buffer_prod_lock], C_D_imp_buffer[0], 0, buffer_size_of_C_D_imp_non, [C_D_imp_buffer_con_lock], 4  ),
                (4, [C_D_non_imp_buffer_prod_lock], C_D_non_imp_buffer[0], 0, buffer_size_of_C_D_imp_non,  [C_D_non_imp_buffer_con_lock], 1 )
            ]
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[5])
            with_block_unroll(block=block, chain=chain0)
            
            chain1 = [
                (6, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock], 7),
                (7, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock], 6),
            ]
            with block[5]:
                s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[8])
            with_block_unroll(block=block, chain=chain1)


            chain2 = [
                (9,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 10),
                (10, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock],  9),
            ]
            with block[8]:
                s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[9], chain=block[11])
            with_block_unroll(block=block, chain=chain2)
            with block[11]:
                EndOp()
        @mem(ComputeTile_0_2)
        def m(block):
            handle_dma_sequences(block) 
                
        @mem(ComputeTile_1_2)
        def m(block):
        #   start_block=1
        #   end_block = total_switch_size+start_block
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[5])  

            with block[1]:
                use_lock(switch_diode_debug_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(switch_diode_buffer_debug_out[0], offset=0, len=buffer_size_of_switch_diode)
                use_lock(switch_diode_debug_prod_lock, LockAction.Release, value=1)
    
                next_bd(block[2])
            with block[2]:
                use_lock(A_B_debug_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(A_B_debug_buffer[0], offset=0, len=buffer_size_of_A_B_matrix)
                use_lock(A_B_debug_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[3])
            with block[3]:
                use_lock(C_D_debug_imp_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(C_D_debug_imp_buffer[0], offset=0, len=buffer_size_of_C_D_imp_non)
                use_lock(C_D_debug_imp_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[4])
            with block[4]:
                use_lock(C_D_debug_non_imp_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(C_D_debug_non_imp_buffer[0], offset=0, len=buffer_size_of_C_D_imp_non)
                use_lock(C_D_debug_non_imp_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[5]:
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
        @core(ComputeTile_0_2, "passThrough.o")
        def core_body():
            for _ in range_(sys.maxsize):
                use_lock(in_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                use_lock(out_buffer_prod_lock,LockAction.AcquireGreaterEqual, value=1)
                accum_float_value_func( in_buffer[0], out_buffer[0],
                                       constant(0), constant(0),   # ping pong 0, no offset need
                                       constant(iteration_step_per_buffer), constant(buffer_size_of_in_ping_pong), constant(buffer_size_of_out_ping_pong)
                                       )
                use_lock(out_buffer_con_lock, LockAction.Release, value=1)
                use_lock(in_buffer_prod_lock, LockAction.Release, value=1)
                
                use_lock(in_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                use_lock(out_buffer_prod_lock,LockAction.AcquireGreaterEqual, value=1)
                accum_float_value_func( in_buffer[0], out_buffer[0],
                                       constant(buffer_size_of_in_ping_pong), constant(buffer_size_of_out_ping_pong),
                                       constant(iteration_step_per_buffer), constant(buffer_size_of_in_ping_pong), constant(buffer_size_of_out_ping_pong)
                                       )
                use_lock(out_buffer_con_lock, LockAction.Release, value=1)
                use_lock(in_buffer_prod_lock, LockAction.Release, value=1)                
                
        @core(ComputeTile_1_2, "passThrough.o")
        def core_body():
          for _ in range_(sys.maxsize):

            use_lock(switch_diode_debug_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(switch_diode_con_lock, LockAction.AcquireGreaterEqual, value=1)
            pass_through_float_diode_matrix( switch_diode_buffer[0], switch_diode_buffer_debug_out[0], constant(buffer_size_of_switch_diode)    )
            use_lock(switch_diode_debug_con_lock, LockAction.Release, value=1)
            use_lock(switch_diode_prod_lock, LockAction.Release, value=1)
            
            use_lock(A_B_debug_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(A_B_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
            pass_through_float_A_B_matrix( A_B_buffer[0], A_B_debug_buffer[0], constant(buffer_size_of_A_B_matrix ))
            use_lock(A_B_buffer_prod_lock, LockAction.Release, value=1)
            use_lock(A_B_debug_buffer_con_lock, LockAction.Release, value=1)
            
            use_lock(C_D_debug_imp_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(C_D_imp_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
            pass_through_float_C_D_matrix( C_D_imp_buffer[0], C_D_debug_imp_buffer[0], constant(buffer_size_of_C_D_imp_non) )
            use_lock(C_D_imp_buffer_prod_lock, LockAction.Release, value=1)
            use_lock(C_D_debug_imp_buffer_con_lock, LockAction.Release, value=1)
            
            use_lock(C_D_debug_non_imp_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(C_D_non_imp_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
            pass_through_float_C_D_matrix( C_D_non_imp_buffer[0], C_D_debug_non_imp_buffer[0], constant(buffer_size_of_C_D_imp_non))
            use_lock(C_D_non_imp_buffer_prod_lock, LockAction.Release, value=1)
            use_lock(C_D_debug_non_imp_buffer_con_lock, LockAction.Release, value=1)
            
            # use_lock(switch_diode_debug_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            # use_lock(switch_diode_con_lock, LockAction.AcquireGreaterEqual, value=1)
            # for k in range(total_switch_size):
            #     pass_through_func( switch_diode_buffer[k], switch_diode_buffer_debug_out[k], constant(buffer_size_of_each_switch_diode_matrix)    )
            # use_lock(switch_diode_debug_con_lock, LockAction.Release, value=1)
            # use_lock(switch_diode_prod_lock, LockAction.Release, value=1)
            
            
        matrix_size =buffer_size_of_switch_diode+buffer_size_of_A_B_matrix + buffer_size_of_C_D_imp_non*2

        data_flow_out_size = buffer_size_of_out_ping_pong *4   # lest do 4 multple o f ping-pong size
        data_flow_in_size =  buffer_size_of_in_ping_pong*4
        
        
        
        #TODO: need balance in port transfer in future
        flow(ShimTile, WireBundle.DMA, 0, ComputeTile_0_2, WireBundle.DMA, 0)
        flow(ComputeTile_1_2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA,0)
        flow(ShimTile, WireBundle.DMA, 1, ComputeTile_0_2, WireBundle.DMA,1 )
        flow(ComputeTile_0_2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA,1 )
        
        
        memref.global_("B_CT_1_2_SHM", T.memref( matrix_size, T.f32() ), sym_visibility="public")
        memref.global_("A_SHM_CT_0_2", T.memref( matrix_size, T.f32() ), sym_visibility="public")
        memref.global_("out_CT_0_2_SHM", T.memref( data_flow_out_size, T.f32()), sym_visibility="public" )
        memref.global_("in_SHM_CT_0_2", T.memref(data_flow_in_size, T.f32()), sym_visibility="public")
        
        shim_dma_allocation("B_CT_1_2_SHM", DMAChannelDir.S2MM, 0, 0)
        shim_dma_allocation("A_SHM_CT_0_2", DMAChannelDir.MM2S, 0, 0)        
        shim_dma_allocation("out_CT_0_2_SHM", DMAChannelDir.S2MM, 1,0)
        shim_dma_allocation("in_SHM_CT_0_2", DMAChannelDir.MM2S, 1, 0 )
        
        @runtime_sequence(np.ndarray[(matrix_size, ), dtype_in], np.ndarray[(matrix_size, ), dtype_out], np.ndarray[(data_flow_in_size,), dtype_in], np.ndarray[(data_flow_out_size,), dtype_out]  )
        def sequence(A,B, in_buf, out_buf):
            npu_dma_memcpy_nd(
                metadata="A_SHM_CT_0_2",
                bd_id=1,
                mem=A, offsets=[0,0,0,0], sizes= [1,1,1,matrix_size],
                strides=[0,0,0,1]
            )

            npu_dma_memcpy_nd(metadata="B_CT_1_2_SHM", bd_id=0, mem=B, offsets=[0,0,0,0],sizes= [1,1,1,matrix_size],strides=[0,0,0,1])

            npu_dma_memcpy_nd(metadata="in_SHM_CT_0_2", bd_id=2, mem=in_buf, offsets=[0,0,0,0], sizes=[1,1,1, data_flow_in_size], strides=[0,0,0,1])
            npu_dma_memcpy_nd(metadata="out_CT_0_2_SHM", bd_id=3, mem=out_buf, offsets=[0,0,0,0], sizes=[1,1,1, data_flow_out_size], strides=[0,0,0,1])

            npu_dma_wait("B_CT_1_2_SHM")
            npu_dma_wait("out_CT_0_2_SHM")
with mlir_mod_ctx() as ctx:
    single_mat_vect_mult()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
