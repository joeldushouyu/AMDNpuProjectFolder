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
from aie.extras.dialects.ext.arith import constant, index_cast

from aie.ir import *
from aie.ir import MemRefType, IndexType
from aie.dialects import arith, memref
from aie.dialects.memref import AllocaScopeOp

from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.dialects.memref import alloc, store, alloca
from aie.extras import types as T


from custom_npu_dma_memcpy import NpuDmaMemcpyNd as custom_npu_dma_memcpy_nd
from aie.dialects.aiex import control_packet
from CT_0_2_helper import *
from custom_npu_dma_memcpy import generate_packet_attribute
# def round_to_nearest_multiple(n, multiple):
#   """Rounds an integer to the nearest multiple of a given number"""
#   if multiple == 0:
#       return n  # Avoid division by zero
#   return ((n + multiple - 1) // multiple) * multiple

import json
# npu_dma_memcpy_nd
def balance_matrix_transfer_case(switch_diode_matrix_size, buffer_A_B_C_D_size, state_size, output_size,u_size,  total_sw_size):
    mid_point = (switch_diode_matrix_size+ buffer_A_B_C_D_size)//2
    
    # Note: the matrix granduality is divided to 2 matrix
    # one matrix for switch_diode case, and repeats for total_sw_size
    
    # another matrix for A_B_C_D_ case, and repeats for total_sw_size
    
    
    
    mid_size = (mid_point-switch_diode_matrix_size)
    
    A_B_C_D_matrix_number_cutoff = mid_size//(  (state_size+2*output_size) *( state_size + u_size) )
    
    return   A_B_C_D_matrix_number_cutoff

    # if(mid_point > switch_diode_matrix_size and mid_point < (switch_diode_matrix_size+A_B_matrix_size)):
    #     return 1, mid_point-switch_diode_matrix_size  # midpoint in A_B_matrix
    # elif(mid_point   >  (switch_diode_matrix_size+A_B_matrix_size) and mid_point < (switch_diode_matrix_size+A_B_matrix_size+C_D_imp_matrix_size)):
    #     return 2, mid_point-switch_diode_matrix_size-A_B_matrix_size
    # else:
    #     raise ValueError("Unexpected scenario")
    

# def custom_floor(x, multiplier):
#   return math.floor(x / multiplier) * multiplier

# def custom_ceil(x, multiplier):
#   return math.ceil(x / multiplier) * multiplier





def single_mat_vect_mult():
    dev = AIEDevice.npu2
    
    with open("final_config.json", "r") as f:
        extracted_data = json.load(f)
    trace_size = extracted_data.get("trace_size")
    state_size = extracted_data.get("state_size")
    u_size = extracted_data.get("u_size")
    y_size = extracted_data.get("y_size")
    diode_size = extracted_data.get("diode_size")
    switch_size = extracted_data.get("switch_size")
    C1_DSW_row_size = extracted_data.get("C1_DSW_row_size")
    C1_DSW_col_size = extracted_data.get("C1_DSW_col_size")
    C1_DSW_matrix_size = extracted_data.get("C1_DSW_matrix_size")
    C1_DSW_buffer_size = extracted_data.get("C1_DSW_buffer_size")
    A_B_C_D_row_size = extracted_data.get("A_B_C_D_row_size")
    A_B_C_D_col_size = extracted_data.get("A_B_C_D_col_size")
    A_B_C_D_matrix_size = extracted_data.get("A_B_C_D_matrix_size")
    A_B_C_D_buffer_size = extracted_data.get("A_B_C_D_buffer_size")
    input_switch_size = extracted_data.get("input_switch_size")
    input_size = extracted_data.get("input_size")
    iteration_step_per_ping_pong_buffer = extracted_data.get("iteration_step_per_ping_pong_buffer")
    buffer_size_of_in_ping_pong = extracted_data.get("buffer_size_of_in_ping_poing")
    buffer_size_of_out_ping_pong = extracted_data.get("buffer_size_of_out_ping_pong")
    ping_pong_buffer_iteration = extracted_data.get("ping_pong_buffer_iteration")

    
    dtype_in = np.dtype[np.float32]
    dtype_out = np.dtype[np.float32]
    
    
    @device(AIEDevice.npu2)
    def device_body():

        
        total_switch_size = 2**(switch_size + diode_size)
   
        # Tile declarations
        ShimTile_0 = tile(0,0)
        ShimTile_1 = tile(1, 0)
        # ComputeTile_0_2 = tile(0,2)        
        # ComputeTile_0_2 = tile(0,2, allocation_scheme="bank-aware")        
        ComputeTile_0_2 = tile(0,2, allocation_scheme="basic-sequential")
        ComputeTile_1_2 = tile(1,2)

        in_data_ty = np.ndarray[ (buffer_size_of_in_ping_pong*2, ), dtype_in]
        out_data_ty = np.ndarray[ (buffer_size_of_out_ping_pong*2, ), dtype_out]

        
        #NOTE: mem_bank flag seem not working anymore after Tile() is configure to basic-sequential address mode
        in_buffer = [
          buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(in_data_ty), sym_name=f"in_buffer_{0}", address=1024), # 1024 offset, reserve for stack

        ]
        in_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=8, init=2, sym_name="in_buffer_p_lock")
        in_buffer_con_lock = lock(ComputeTile_0_2, lock_id=9, init=0, sym_name="in_buffer_c_lock")
        
        out_buffer_address = (64*1024) - (buffer_size_of_out_ping_pong*2*4) # 4 byte per float
        out_buffer = [
            buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(out_data_ty), sym_name=f"out_buffer_{0}", address=out_buffer_address ), # 
        ]        
        out_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=10, init=2)
        out_buffer_con_lock = lock(ComputeTile_0_2, lock_id=11, init=0)
                
        
        

     

        switch_diode_matrix_ty = np.ndarray[ (C1_DSW_buffer_size, ), dtype_in]
        switch_diode_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=switch_diode_matrix_ty, name=f"switch_diode_buffer") 
        ]
        switch_diode_prod_lock =  lock(ComputeTile_0_2, lock_id=0, init=1, sym_name="switch_diode_prod_lock")
        switch_diode_con_lock = lock(ComputeTile_0_2, lock_id=1, init=0, sym_name="switch_diode_con_lock")


        pass_through_float_diode_matrix = external_func( "passThroughLine_float_0", inputs=[
          switch_diode_matrix_ty, switch_diode_matrix_ty, np.int32
        ] )


        A_B_C_D_ty = np.ndarray[(A_B_C_D_buffer_size,  ), dtype_in]
        
        A_B_C_D_buffer = [
            buffer(tile=ComputeTile_0_2, datatype=A_B_C_D_ty, name="A_B_C_D_buffer")
        ]
        
        A_B_C_D_prod_lock = lock(ComputeTile_0_2, lock_id=2, init=2, sym_name="A_B_C_D_prod_lock")
        A_B_C_D_con_lock = lock(ComputeTile_0_2, lock_id=3, init=0, sym_name="A_B_C_D_con_lock")
        
        pass_through_float_A_B_C_D_matrix =external_func(  "passThroughLine_float_1", inputs=[
            A_B_C_D_ty, A_B_C_D_ty, np.int32
        ])

        # Debug Buffer
        switch_diode_buffer_debug_out = [
          buffer(tile=ComputeTile_1_2, datatype=switch_diode_matrix_ty, name=f"switch_diode_buffer_debug") 
        ]
        switch_diode_debug_prod_lock = lock(ComputeTile_1_2, lock_id= 0,init=1, sym_name="switch_diode_debug_prod_lock" )
        switch_diode_debug_con_lock = lock(ComputeTile_1_2, lock_id=1, init=0, sym_name="switch_diode_debug_con_lock")

        A_B_C_D_debug_buffer = [
            buffer(tile=ComputeTile_1_2, datatype=A_B_C_D_ty, name="A_B_C_D_debug_buffer" )
        ]
        A_B_C_D_debug_prod_lock = lock(ComputeTile_1_2, lock_id=2, init=1)
        A_B_C_D_debug_con_lock = lock(ComputeTile_1_2, lock_id=3, init=0)        
        

        # strategy to balance out the S2MM workload on two port of CT_0_2

        A_B_C_D_num_for_balance_cutoff = balance_matrix_transfer_case(
            switch_diode_matrix_size=C1_DSW_buffer_size,
            buffer_A_B_C_D_size= A_B_C_D_buffer_size,
            state_size= state_size,
            output_size=y_size,
            u_size=u_size,
            total_sw_size=total_switch_size
        )
        # print(A_B_C_D_num_for_balance_cutoff)
        mid_offset = A_B_C_D_num_for_balance_cutoff *(state_size+2*y_size)*(state_size+u_size )
        
        @mem(ComputeTile_0_2)
        def m(block):

            #block_idx, acqire_locks, buffer, buffer_offset, buffer_len, release_locks, next_idx, [packet_id, packet_type]    
            chain0 =[
                (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, C1_DSW_buffer_size, [switch_diode_con_lock], 2, [] ),
                (2, [A_B_C_D_prod_lock],   A_B_C_D_buffer[0],    0, mid_offset,                   [A_B_C_D_con_lock],  1, [])
            ]
            chain0_s_e = (1, 1+len(chain0))
            
            chain1 = [
                (4, [A_B_C_D_prod_lock], A_B_C_D_buffer[0], mid_offset,     A_B_C_D_buffer_size-mid_offset, [A_B_C_D_con_lock], 5, []),
                (5, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock],6, [] ),
                (6, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock],5, [] ), # becase matrix only transfer once
            ]
            chain1_s_e = (chain0_s_e[1]+1,chain0_s_e[1]+1+len(chain1))

            chain2 = [
                (8,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 9, [9,0]),
                (9, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock],  8, [9,0]),
            ]  
            chain2_s_e = (chain1_s_e[1]+1, chain1_s_e[1]+1+len(chain2))     

            handle_dma_sequences(block, chain0=chain0, chain1=chain1, chain2=chain2, chain0_start_end=chain0_s_e, chain1_start_end=chain1_s_e, chain2_start_end=chain2_s_e) 
                
        @mem(ComputeTile_1_2)
        def m(block):
        #   start_block=1
        #   end_block = total_switch_size+start_block
            s0 = dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[3])  

            with block[1]:
                use_lock(switch_diode_debug_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(switch_diode_buffer_debug_out[0], offset=0, len=C1_DSW_buffer_size, packet=generate_packet_attribute(packet_id=7, packet_type=0))
                use_lock(switch_diode_debug_prod_lock, LockAction.Release, value=1)
                next_bd(block[2])
                
            with block[2]:
                use_lock(A_B_C_D_debug_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(A_B_C_D_debug_buffer[0], offset=0, len=A_B_C_D_buffer_size,  packet=generate_packet_attribute(packet_id=7, packet_type=0))
                use_lock(A_B_C_D_debug_prod_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[3]:
                EndOp()
                
                
        CT_0_2_main_func = external_func("CT_main", inputs=[
            in_data_ty, out_data_ty,
            np.int32, np.int32, np.int32,
            np.int32, np.int32, np.int32, np.int32,
            switch_diode_matrix_ty, np.int32, np.int32
            
        ])
        
        @core(ComputeTile_0_2, "passThrough.o")
        def core_body():
            # for _ in range_(sys.maxsize):
            CT_0_2_main_func(
                in_buffer[0], out_buffer[0],
                constant(buffer_size_of_in_ping_pong), constant(buffer_size_of_out_ping_pong), constant(iteration_step_per_ping_pong_buffer),
                constant(8),constant(9),constant(10),constant(11),
                switch_diode_buffer[0], constant(C1_DSW_row_size), constant(C1_DSW_col_size)
                
            )

        # CT_0_2_main_func = external_func("CT_main", inputs=[
        #     in_data_ty, out_data_ty,
        #     np.int32, np.int32, np.int32,
        #     np.int32, np.int32, np.int32, np.int32,
        #     switch_diode_matrix_ty, np.int32, np.int32
            
        # ])
        
        # @core(ComputeTile_0_2, "mainKernel.o")
        # def core_body():

        #     CT_0_2_main_func(
        #         in_buffer[0], out_buffer[0],
        #         constant(buffer_size_of_in_ping_pong), constant(buffer_size_of_out_ping_pong), constant(iteration_step_per_ping_pong_buffer),
        #         8,9,10,11,
        #         switch_diode_buffer[0], constant(C1_DSW_row_size), constant(C1_DSW_col_size)
                
        #     )


        @core(ComputeTile_1_2, "passThrough.o")
        def core_body():
          for _ in range_(sys.maxsize):

            use_lock(switch_diode_debug_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(switch_diode_con_lock, LockAction.AcquireGreaterEqual, value=1)
            pass_through_float_diode_matrix( switch_diode_buffer[0], switch_diode_buffer_debug_out[0], constant(C1_DSW_buffer_size)    )
            use_lock(switch_diode_debug_con_lock, LockAction.Release, value=1)
            use_lock(switch_diode_prod_lock, LockAction.Release, value=1)
            
            
        
        
            use_lock(A_B_C_D_debug_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(A_B_C_D_con_lock, LockAction.AcquireGreaterEqual, value=2)  # decrement the lock by 2 after acquire
            pass_through_float_A_B_C_D_matrix(  A_B_C_D_buffer[0], A_B_C_D_debug_buffer[0], constant(A_B_C_D_buffer_size)  )
            use_lock(A_B_C_D_prod_lock, LockAction.Release, value=2) # increment the lock by 2 after acquire            
            use_lock(A_B_C_D_debug_con_lock, LockAction.Release, value=1)
            
            
        matrix_size =C1_DSW_buffer_size+A_B_C_D_buffer_size
        data_flow_out_size = buffer_size_of_out_ping_pong *ping_pong_buffer_iteration   # lest do 4 multple o f ping-pong size
        data_flow_in_size =  buffer_size_of_in_ping_pong*ping_pong_buffer_iteration

        in_0_size = C1_DSW_buffer_size +mid_offset
        in_1_size = (A_B_C_D_buffer_size-mid_offset)  + data_flow_in_size
        
        if(trace_size > 0):
            tiles_to_trace = [ComputeTile_0_2, ComputeTile_1_2] #TODO: also shimtile?
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile_1)

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
            # transfer the switch_diode_matrix in column major order
            
            custom_npu_dma_memcpy_nd(
                metadata="in_SHM_CT_0_2_0",
                bd_id=1,
                mem=A, offsets=[0,0,0,0], sizes= [1, total_switch_size  , C1_DSW_col_size, C1_DSW_row_size ],
                strides=[0,  C1_DSW_matrix_size ,1,C1_DSW_col_size],  
                packet_id=6,
                packet_type=0                  
            )
            assert C1_DSW_buffer_size % A_B_C_D_col_size == 0
            custom_npu_dma_memcpy_nd(
                metadata="in_SHM_CT_0_2_0",
                bd_id=2,
                mem=A, offsets=[0,0,0, C1_DSW_buffer_size //A_B_C_D_col_size ],   # The offset will multiple with the sizes
                sizes= [1, A_B_C_D_num_for_balance_cutoff , A_B_C_D_col_size, A_B_C_D_row_size],
                strides=[0,   A_B_C_D_matrix_size   ,1, A_B_C_D_col_size],
                packet_id=6,
                packet_type=0                                               
            )
            assert in_0_size % A_B_C_D_col_size == 0
            if (in_1_size-data_flow_in_size > 0):
                custom_npu_dma_memcpy_nd(metadata="in_SHM_CT_0_2_1", bd_id=3, mem=A, offsets=[0, 0,  0,  in_0_size //A_B_C_D_col_size  ], #TODO: assert is okay for it
                                        sizes= [1, total_switch_size-A_B_C_D_num_for_balance_cutoff , A_B_C_D_col_size, A_B_C_D_row_size],
                                        strides=[0,   A_B_C_D_matrix_size   ,1, A_B_C_D_col_size],
                                        packet_id=8, packet_type=0)

            custom_npu_dma_memcpy_nd(metadata="out_CT_0_2_SHM", bd_id=4, mem=out_buf, offsets=[0,0,0,0], sizes=[1,1,1, data_flow_out_size], 
                                     strides=[0,0,0,1], issue_token=True)
            custom_npu_dma_memcpy_nd(metadata="in_SHM_CT_0_2_1", bd_id=5, mem=in_buf, offsets=[0,0,0,0], 
                                     sizes=[1,1,1, data_flow_in_size ], strides=[0,0,0,1], packet_id=8, packet_type=0)
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
