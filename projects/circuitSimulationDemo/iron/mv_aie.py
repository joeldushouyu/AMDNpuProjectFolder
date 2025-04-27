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

from aie.dialects._aie_ops_gen import buffer as buffer_raw
from aie.helpers.util import try_convert_np_type_to_mlir_type
import numpy as np
import sys
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from enum import IntEnum

from custom_npu_dma_memcpy import NpuDmaMemcpyNd as custom_npu_dma_memcpy_nd
from aie.dialects.aiex import control_packet
from CT_0_2_helper import *
from custom_npu_dma_memcpy import generate_packet_attribute
# def round_to_nearest_multiple(n, multiple):
#   """Rounds an integer to the nearest multiple of a given number"""
#   if multiple == 0:
#       return n  # Avoid division by zero
#   return ((n + multiple - 1) // multiple) * multiple


# npu_dma_memcpy_nd
def balance_matrix_transfer_case(switch_diode_matrix_size, A_B_matrix_size, C_D_imp_matrix_size, C_D_non_imp_matrix_size):
    mid_point = (switch_diode_matrix_size+ A_B_matrix_size+ C_D_imp_matrix_size+C_D_non_imp_matrix_size)//2
    
    # possible only two case
    if(mid_point > switch_diode_matrix_size and mid_point < (switch_diode_matrix_size+A_B_matrix_size)):
        return 1, mid_point-switch_diode_matrix_size  # midpoint in A_B_matrix
    elif(mid_point   >  (switch_diode_matrix_size+A_B_matrix_size) and mid_point < (switch_diode_matrix_size+A_B_matrix_size+C_D_imp_matrix_size)):
        return 2, mid_point-switch_diode_matrix_size-A_B_matrix_size
    else:
        raise ValueError("Unexpected scenario")
    

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
        trace_size = 2048 #TODO:
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
        ShimTile_0 = tile(0,0)
        ShimTile_1 = tile(1, 0)
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
        in_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=8, init=2, sym_name="in_buffer_p_lock")
        in_buffer_con_lock = lock(ComputeTile_0_2, lock_id=9, init=0, sym_name="in_buffer_c_lock")
        
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
        
        
        
        strategy to balance out the S2MM workload on two port of CT_0_2
        S2MM_stratgey, offset = balance_matrix_transfer_case(buffer_size_of_switch_diode, buffer_size_of_A_B_matrix,
                                                     buffer_size_of_C_D_imp_non, buffer_size_of_C_D_imp_non)
        # S2MM_stratgey = 2

        @mem(ComputeTile_0_2)
        def m(block):

            if S2MM_stratgey == 1:
                # DIVISION point at  A_B_ buffer
                #block_idx, acqire_locks, buffer, buffer_offset, buffer_len, release_locks, next_idx, [packet_id, packet_type]                 
                chain0 = [
                    (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, buffer_size_of_switch_diode, [switch_diode_con_lock], 2, [] ),
                    (2, [A_B_buffer_prod_lock],   A_B_buffer[0], 0, buffer_size_of_A_B_matrix, [A_B_buffer_con_lock], 1, []   ),
                ]
                chain0_s_e = (1, 1+len(chain0))
                chain1 = [
                    (4, [C_D_imp_buffer_prod_lock], C_D_imp_buffer[0], 0, buffer_size_of_C_D_imp_non, [C_D_imp_buffer_con_lock],5, [] ),
                    (5, [C_D_non_imp_buffer_prod_lock], C_D_non_imp_buffer[0], 0, buffer_size_of_C_D_imp_non,  [C_D_non_imp_buffer_con_lock], 6, []),
                    (6, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock],7, [] ),
                    (7, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock],6, [] ), # becase matrix only transfer once
                ]
                chain1_s_e = (chain0_s_e[1]+1,chain0_s_e[1]+1+len(chain1))
                chain2 = [
                    (9,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 10, [9,0]),
                    (10, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock],  9, [9,0]),
                ]  
                chain2_s_e = (chain1_s_e[1]+1, chain1_s_e[1]+1+len(chain2))     
            elif S2MM_stratgey == 2:
                # during middle of C_D_impulse
                chain0 = [
                    (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, buffer_size_of_switch_diode, [switch_diode_con_lock], 2, [] ),
                    (2, [A_B_buffer_prod_lock],   A_B_buffer[0], 0, buffer_size_of_A_B_matrix, [A_B_buffer_con_lock], 3, []   ),
                    (3, [C_D_imp_buffer_prod_lock], C_D_imp_buffer[0], 0, buffer_size_of_C_D_imp_non, [C_D_imp_buffer_con_lock], 1, [] ),
                ]
                chain0_s_e = (1, 1+len(chain0))
                chain1 = [
                    (5, [C_D_non_imp_buffer_prod_lock], C_D_non_imp_buffer[0], 0, buffer_size_of_C_D_imp_non,  [C_D_non_imp_buffer_con_lock], 6, [] ),
                    (6, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock], 7, []),
                    (7, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock], 6, []) # because C_D_non one transfer once
                ]
                chain1_s_e = (chain0_s_e[1]+1,chain0_s_e[1]+1+len(chain1))
                chain2 = [
                    (9,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 10, [9,0]),
                    (10, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 9, [9,0]),
                ]  
                chain2_s_e = (chain1_s_e[1]+1, chain1_s_e[1]+1+len(chain2))                      
            else :
                #Does not consider any workload balance case
                chain0 = [
                    (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, buffer_size_of_switch_diode, [switch_diode_con_lock], 2, [] ),
                    (2, [A_B_buffer_prod_lock],   A_B_buffer[0], 0, buffer_size_of_A_B_matrix, [A_B_buffer_con_lock], 3, []   ),
                    (3, [C_D_imp_buffer_prod_lock], C_D_imp_buffer[0], 0, buffer_size_of_C_D_imp_non, [C_D_imp_buffer_con_lock], 4, []  ),
                    (4, [C_D_non_imp_buffer_prod_lock], C_D_non_imp_buffer[0], 0, buffer_size_of_C_D_imp_non,  [C_D_non_imp_buffer_con_lock], 1, [] )
                ]
                chain0_s_e = (1, 1+len(chain0))
                chain1 = [
                    (6, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock], 7, []),
                    (7, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock], 6, []),
                ]
                chain1_s_e = (chain0_s_e[1]+1,chain0_s_e[1]+1+len(chain1))
                chain2 = [
                    (9,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 10, [9,0]),
                    (10, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock],  9, [9,0]),
                ]  
                chain2_s_e = (chain1_s_e[1]+1, chain1_s_e[1]+1+len(chain2))                
            

            handle_dma_sequences(block, chain0=chain0, chain1=chain1, chain2=chain2, chain0_start_end=chain0_s_e, chain1_start_end=chain1_s_e, chain2_start_end=chain2_s_e) 
                
        @mem(ComputeTile_1_2)
        def m(block):
        #   start_block=1
        #   end_block = total_switch_size+start_block
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[5])  

            with block[1]:
                use_lock(switch_diode_debug_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(switch_diode_buffer_debug_out[0], offset=0, len=buffer_size_of_switch_diode, packet=generate_packet_attribute(packet_id=7, packet_type=0))
                use_lock(switch_diode_debug_prod_lock, LockAction.Release, value=1)
    
                next_bd(block[2])
            with block[2]:
                use_lock(A_B_debug_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # dma_bd_packet(packet_id=1, packet_type=0)                
                dma_bd(A_B_debug_buffer[0], offset=0, len=buffer_size_of_A_B_matrix, packet=generate_packet_attribute(packet_id=7, packet_type=0))
                use_lock(A_B_debug_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[3])
            with block[3]:
                use_lock(C_D_debug_imp_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # dma_bd_packet(packet_id=1, packet_type=0)                
                dma_bd(C_D_debug_imp_buffer[0], offset=0, len=buffer_size_of_C_D_imp_non, packet=generate_packet_attribute(packet_id=7, packet_type=0))
                use_lock(C_D_debug_imp_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[4])
            with block[4]:
                use_lock(C_D_debug_non_imp_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # dma_bd_packet(packet_id=1, packet_type=0)                
                dma_bd(C_D_debug_non_imp_buffer[0], offset=0, len=buffer_size_of_C_D_imp_non, packet=generate_packet_attribute(packet_id=7, packet_type=0))
                use_lock(C_D_debug_non_imp_buffer_prod_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[5]:
                EndOp()
                
            
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
            

            
            
        matrix_size =buffer_size_of_switch_diode+buffer_size_of_A_B_matrix + buffer_size_of_C_D_imp_non*2

        data_flow_out_size = buffer_size_of_out_ping_pong *4   # lest do 4 multple o f ping-pong size
        data_flow_in_size =  buffer_size_of_in_ping_pong*4
        
        if S2MM_stratgey == 2:
            in_0_size = buffer_size_of_switch_diode + buffer_size_of_A_B_matrix +buffer_size_of_C_D_imp_non
            in_1_size =  buffer_size_of_C_D_imp_non + data_flow_in_size
        elif S2MM_stratgey == 1:
            in_0_size = buffer_size_of_switch_diode+ buffer_size_of_A_B_matrix
            in_1_size = 2*buffer_size_of_C_D_imp_non + data_flow_in_size
        else:
            in_0_size = matrix_size
            in_1_size = data_flow_in_size
            
        if(trace_size > 0):
            tiles_to_trace = [ComputeTile_0_2, ComputeTile_1_2] #TODO: also shimtile?
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile_1)
            
        #flow(ShimTile, WireBundle.DMA, 0, ComputeTile_0_2, WireBundle.DMA, 0)
        #flow(ComputeTile_1_2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA,0)
        #flow(ShimTile, WireBundle.DMA, 1, ComputeTile_0_2, WireBundle.DMA,1 )
        # flow(ComputeTile_0_2, WireBundle.DMA, 0, ShimTile, WireBundle.DMA,1 )
        
        # Change to packet flow
        # leave first 6(0-5) packet id for tracing
        packetflow( 6, source=ShimTile_0, source_port=WireBundle.DMA, source_channel=0, 
                   dest = ComputeTile_0_2, dest_port=WireBundle.DMA, dest_channel=0
                   )
        packetflow(7, source=ComputeTile_1_2, source_port=WireBundle.DMA, source_channel=0,
                   dest = ShimTile_0, dest_port=WireBundle.DMA, dest_channel=0
                   )
        packetflow(8, source=ShimTile_0, source_port=WireBundle.DMA, source_channel=1,
                   dest=ComputeTile_0_2, dest_port=WireBundle.DMA, dest_channel=1
                   )
        packetflow(9, source=ComputeTile_0_2, source_port=WireBundle.DMA, source_channel=0,
                    dest = ShimTile_0, dest_port= WireBundle.DMA, dest_channel=1
                   )
        
        memref.global_("in_SHM_CT_0_2_0", T.memref( in_0_size, T.f32() ), sym_visibility="public")            
        memref.global_("in_SHM_CT_0_2_1", T.memref(in_1_size, T.f32()), sym_visibility="public")
        memref.global_("B_CT_1_2_SHM", T.memref( matrix_size, T.f32() ), sym_visibility="public") #DEBUG out
        memref.global_("out_CT_0_2_SHM", T.memref( data_flow_out_size, T.f32()), sym_visibility="public" ) # result out

            
        shim_dma_allocation("B_CT_1_2_SHM", DMAChannelDir.S2MM, 0, 0)
        shim_dma_allocation("in_SHM_CT_0_2_0", DMAChannelDir.MM2S, 0, 0)        
        shim_dma_allocation("out_CT_0_2_SHM", DMAChannelDir.S2MM, 1,0)
        shim_dma_allocation("in_SHM_CT_0_2_1", DMAChannelDir.MM2S, 1, 0 )

        @runtime_sequence(np.ndarray[(matrix_size, ), dtype_in], np.ndarray[(matrix_size, ), dtype_out], np.ndarray[(data_flow_in_size,), dtype_in], np.ndarray[(data_flow_out_size,), dtype_out]  )
        def sequence(A,B, in_buf, out_buf):
            # work balance module
            custom_npu_dma_memcpy_nd(metadata="B_CT_1_2_SHM", bd_id=0, mem=B, offsets=[0,0,0,0],sizes= [1,1,1,matrix_size],strides=[0,0,0,1], issue_token=True)
            
            custom_npu_dma_memcpy_nd(
                metadata="in_SHM_CT_0_2_0",
                bd_id=1,
                mem=A, offsets=[0,0,0,0], sizes= [1,1,1,in_0_size],
                strides=[0,0,0,1],
                packet_id=6,
                packet_type=0
            )
            
            
            if (in_1_size-data_flow_in_size > 0):
                custom_npu_dma_memcpy_nd(metadata="in_SHM_CT_0_2_1", bd_id=2, mem=A, offsets=[0,0,0, in_0_size], 
                                         sizes=[1,1,1, in_1_size-data_flow_in_size], strides=[0,0,0,1],  packet_id=8, packet_type=0)

            custom_npu_dma_memcpy_nd(metadata="out_CT_0_2_SHM", bd_id=4, mem=out_buf, offsets=[0,0,0,0], sizes=[1,1,1, data_flow_out_size], 
                                     strides=[0,0,0,1], issue_token=True)
            custom_npu_dma_memcpy_nd(metadata="in_SHM_CT_0_2_1", bd_id=5, mem=in_buf, offsets=[0,0,0,0], 
                                     sizes=[1,1,1, data_flow_in_size ], strides=[0,0,0,1], packet_id=8, packet_type=0)
            # npu_dma_wait("B_CT_1_2_SHM")
            # npu_dma_wait("out_CT_0_2_SHM")
            if(trace_size > 0):
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace=tiles_to_trace,
                    ddr_id=4,   # last in/out parameter(not just need to pass in host, did not define in sequence)
                    shim =ShimTile_1,
                    trace_size=trace_size, # beacuse have 2 tile to,
                )
            npu_dma_wait("B_CT_1_2_SHM")
            npu_dma_wait("out_CT_0_2_SHM")

with mlir_mod_ctx() as ctx:
    single_mat_vect_mult()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
