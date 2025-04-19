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
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from enum import IntEnum


from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

from aie.extras.dialects.ext.arith import constant
from aie.dialects import memref
from aie.dialects._aie_ops_gen import cascade_flow, CascadeFlowOp

def cascade_flow(
    source,
    dest
):
    assert source is not None
    assert dest is not None
    
    
    return CascadeFlowOp(
        source_tile = source, dest_tile = dest
    )
    
    

# This is a simple demo of vector vector elementwise multiplication(dot product of two vector), both vector and the scalar are floatt32
# CT 0 pass the vector through cascade stream
# CT1(right neigbor of CT0) receives the vector from cascade stream and then do a addition to it before giving the result back

def single_mat_vect_mult():
    dev = AIEDevice.npu2
    

    
    dtype_in = np.dtype[np.float32]  
    dtype_out = np.dtype[np.float32]
    
    
    @device(AIEDevice.npu2)
    
    def device_body():
        Vector_total_size = 1024
        vector_size = 512
        trace_size = 3072
        In_Vector_ty = np.ndarray[ (vector_size, ), dtype_in  ]
        Out_Vector_ty = np.ndarray[ (vector_size,), dtype_out]

        
        # passThrough_func = external_func("passThroughLine_float32",
        #                                  inputs=[In_Vector_ty, Out_Vector_ty, np.int32]) 
        
        # for kernel 1
        loadFloatVectorToCascade_func =external_func("loadFloatVectorToCascade",
                                                     inputs = [In_Vector_ty, np.int32])
        vectorElementWiseMultiplication_from_cascade_func = external_func(
            "vectorElementWiseMultiplication_from_cascade",
            inputs=[ In_Vector_ty, Out_Vector_ty, np.int32 ]
        )
        memref.global_("SHM0_CT_0_0", T.memref( vector_size, T.f32()), sym_visibility="public")
        # memref.global_("CT_0_0_SHM0", T.memref( vector_size, T.f32()), sym_visibility="public")
        memref.global_("SHM1_CT_1_0", T.memref(vector_size, T.f32()), sym_visibility="public")
        memref.global_("CT_1_0_SHM1", T.memref(vector_size, T.f32()), sym_visibility="public")
        
        shim_dma_allocation("SHM0_CT_0_0", DMAChannelDir.MM2S, 0,0)
        # shim_dma_allocation("CT_0_0_SHM0", DMAChannelDir.S2MM, 1,0)
        shim_dma_allocation("SHM1_CT_1_0", DMAChannelDir.MM2S, 0, 1)
        shim_dma_allocation("CT_1_0_SHM1", DMAChannelDir.S2MM, 0,1)
        
        #Tile declaration
        ShimTile_0 = tile(0,0)
        ShimTile_1 = tile(1, 0)
        CT_0_0 = tile(0,2)
        CT_1_0 = tile(1, 2)
        
        
        
        #CT_0_0 elements
        inV_CT_0_0_prod_lock = lock(CT_0_0, lock_id=0, init=2, sym_name="inV_CT_0_0_prod_lock")
        inV_CT_0_0_con_lock = lock(CT_0_0, lock_id=1, init=0, sym_name="inV_CT_0_0_con_lock")
        inV_CT_0_0_buffers = [
            buffer(tile=CT_0_0, datatype=In_Vector_ty, name="inV_CT_0_0_buffers_0"),
            buffer(tile=CT_0_0, datatype=In_Vector_ty, name="inV_CT_0_0_buffers_1")
        ]
        
        # outV_CT_0_0_prod_lock = lock(CT_0_0, lock_id=2, init=2, sym_name="outV_CT_0_0_prod_lock")
        # outV_CT_0_0_con_lock = lock(CT_0_0, lock_id=3, init=0, sym_name="outV_CT_0_0_con_lock")
        # outV_CT_0_0_buffers  = [
        #     buffer(tile=CT_0_0, datatype=Out_Vector_ty, name="outV_CT_0_0_buffers_0"),
        #     buffer(tile=CT_0_0, datatype=Out_Vector_ty, name="outV_CT_0_0_buffers_1")
        # ]        
        
        
        #CT_1_0 elements
        inV_CT_1_0_prod_lock = lock(CT_1_0, lock_id=0, init=2, sym_name="inV_CT_1_0_prod_lock")
        inV_CT_1_0_con_lock = lock(CT_1_0, lock_id=1, init=0, sym_name="inV_CT_1_0_con_lock")
        inV_CT_1_0_buffers = [
            buffer(tile=CT_1_0, datatype=In_Vector_ty),buffer(tile=CT_1_0, datatype=In_Vector_ty)
        ]        

        outV_CT_1_0_prod_lock = lock(CT_1_0, lock_id=2, init=2, sym_name="outV_CT_1_0_prod_lock")
        outV_CT_1_0_con_lock = lock(CT_1_0, lock_id=3, init=0, sym_name="outV_CT_1_0_con_lock")
        outV_CT_1_0_buffers = [
            buffer(tile=CT_1_0, datatype=Out_Vector_ty),buffer(tile=CT_1_0, datatype=Out_Vector_ty)
        ]

        if(trace_size > 0):
            tiles_to_trace = [CT_0_0, CT_1_0] #TODO: also shimtile 1?
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile_0)
        
        cascade_flow(CT_0_0, CT_1_0)
        
        # now configure the flow of it
        flow(ShimTile_0, WireBundle.DMA, 0, CT_0_0, WireBundle.DMA, 0)
        # flow(CT_0_0, WireBundle.DMA, 0, ShimTile_0, WireBundle.DMA, 0)
        flow(ShimTile_1, WireBundle.DMA, 0, CT_1_0, WireBundle.DMA, 0)
        flow(CT_1_0, WireBundle.DMA, 0, ShimTile_1, WireBundle.DMA, 0)
        
        # The memtile logic for CT_0_0
        @mem(CT_0_0)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])  # from Input Vector
            with block[1]:
                use_lock(inV_CT_0_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inV_CT_0_0_buffers[0], offset = 0, len=vector_size)
                use_lock(inV_CT_0_0_con_lock, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inV_CT_0_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inV_CT_0_0_buffers[1], offset = 0, len=vector_size)
                use_lock(inV_CT_0_0_con_lock, LockAction.Release, value=1)
                next_bd(block[1])          
            with block[3]:
            #     s1 = dma_start(DMAChannelDir.MM2S, 1, dest=block[4], chain=block[6]) # vector out
            # with block[4]:
            #     use_lock(outV_CT_0_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
            #     dma_bd(outV_CT_0_0_buffers[0], offset=0, len = vector_size)
            #     use_lock(outV_CT_0_0_prod_lock, LockAction.Release, value=1)      
            #     next_bd(block[5])
            # with block[5]:
            #     use_lock(outV_CT_0_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
            #     dma_bd(outV_CT_0_0_buffers[1], offset=0, len = vector_size)
            #     use_lock(outV_CT_0_0_prod_lock, LockAction.Release, value=1)      
            #     next_bd(block[4])         
            # with block[6]:
                EndOp()       
                
        @mem(CT_1_0)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])  # from Input Vector
            with block[1]:
                use_lock(inV_CT_1_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inV_CT_1_0_buffers[0], offset = 0, len=vector_size)
                use_lock(inV_CT_1_0_con_lock, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inV_CT_1_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inV_CT_1_0_buffers[1], offset = 0, len=vector_size)
                use_lock(inV_CT_1_0_con_lock, LockAction.Release, value=1)
                next_bd(block[1])          
            with block[3]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6]) # vector out
            with block[4]:
                use_lock(outV_CT_1_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outV_CT_1_0_buffers[0], offset=0, len = vector_size)
                use_lock(outV_CT_1_0_prod_lock, LockAction.Release, value=1)      
                next_bd(block[5])
            with block[5]:
                use_lock(outV_CT_1_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outV_CT_1_0_buffers[1], offset=0, len = vector_size)
                use_lock(outV_CT_1_0_prod_lock, LockAction.Release, value=1)      
                next_bd(block[4])         
            with block[6]:
                EndOp()       

        #THe logic for CT 0 0
        @core(CT_0_0, "kernel1.o")
        def core_body():
            for _ in range_(sys.maxsize):
                use_lock(inV_CT_0_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(outV_CT_0_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                in_v_a =  inV_CT_0_0_buffers[0]
                # out_v_a = outV_CT_0_0_buffers[0]
                loadFloatVectorToCascade_func(in_v_a, constant(vector_size))

                # use_lock(outV_CT_0_0_con_lock, LockAction.Release, value=1)
                use_lock(inV_CT_0_0_prod_lock, LockAction.Release, value=1)

                
                
                use_lock(inV_CT_0_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(outV_CT_0_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                in_v_a =  inV_CT_0_0_buffers[1]
                # out_v_a = outV_CT_0_0_buffers[1]
                loadFloatVectorToCascade_func(in_v_a,constant(vector_size))
                # use_lock(outV_CT_0_0_con_lock, LockAction.Release, value=1)
                use_lock(inV_CT_0_0_prod_lock, LockAction.Release, value=1)                

        @core(CT_1_0, "kernel2.o")
        def core_body():
            for _ in range_(sys.maxsize):
                use_lock(inV_CT_1_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
                use_lock(outV_CT_1_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                in_v_a = inV_CT_1_0_buffers[0]
                out_v_a_ = outV_CT_1_0_buffers[0]
                vectorElementWiseMultiplication_from_cascade_func(in_v_a, out_v_a_,constant(vector_size))
                use_lock(inV_CT_1_0_prod_lock, LockAction.Release, value=1)
                use_lock(outV_CT_1_0_con_lock, LockAction.Release, value=1)


                use_lock(inV_CT_1_0_con_lock, LockAction.AcquireGreaterEqual, value=1)
                use_lock(outV_CT_1_0_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                in_v_a = inV_CT_1_0_buffers[1]
                out_v_a_ = outV_CT_1_0_buffers[1]
                vectorElementWiseMultiplication_from_cascade_func(in_v_a, out_v_a_, constant(vector_size))
                use_lock(inV_CT_1_0_prod_lock, LockAction.Release, value=1)
                use_lock(outV_CT_1_0_con_lock, LockAction.Release, value=1)


        
                
        @runtime_sequence( np.ndarray[(vector_size,), dtype_in],  
                        #   np.ndarray[(vector_size,), dtype_out],
                          np.ndarray[(vector_size,), dtype_in],  np.ndarray[(vector_size,), dtype_out] )
        def sequence(V1, V2, V2_res):
                # r = M // m 
                # assert r * m  == M
                # rtp_buffer[0] = 1 TODO
                npu_dma_memcpy_nd(
                    metadata="SHM0_CT_0_0",
                    bd_id=0,
                    mem=V1,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, vector_size],
                    strides=[0, 0, 0, 1],
                )
                
                
                if(trace_size > 0):
                    #Note: this automatically write to ddr_id3, app_id 6. Thus host need to be read to receive it
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        ddr_id=3,
                        shim=ShimTile_0,
                        trace_size=trace_size,
                        coretile_events=[
                            CoreEvent.INSTR_EVENT_0,
                            CoreEvent.INSTR_EVENT_1,
                            CoreEvent.INSTR_VECTOR,
                            PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),  # master(1)
                            PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),  # slave(1)
                            CoreEvent.INSTR_CASCADE_GET,
                            CoreEvent.INSTR_CASCADE_PUT,
                            CoreEvent.CASCADE_STALL,
                        ],
                    )
                    
                npu_dma_memcpy_nd(
                    metadata="SHM1_CT_1_0",
                    bd_id=0,
                    mem=V2,
                    offsets=[0,0,0,0],
                    sizes=[1,1,1, vector_size],
                    strides=[0,0,0,1]
                )
                
                npu_dma_memcpy_nd(
                    metadata="CT_1_0_SHM1",
                    bd_id=1,
                    mem=V2_res,
                    offsets= [0,0,0,0],
                    sizes= [ 1,1,1,vector_size],
                    strides=[0,0,0,1]
                    
                )

                # npu_dma_wait("CT_0_0_SHM0")
                npu_dma_wait("CT_1_0_SHM1")
with mlir_mod_ctx() as ctx:
    single_mat_vect_mult()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
