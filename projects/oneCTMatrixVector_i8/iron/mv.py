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


def single_mat_vect_mult():
    dev = AIEDevice.npu2
    

    
    dtype_in = np.dtype[np.int8]
    dtype_out = np.dtype[np.int32]
    
    
    @device(AIEDevice.npu2)
    
    def device_body():
    
        M_size = 512
        K_size = 512
        m_size = 128
        k_size = 128
        K_div_k = K_size//k_size
        m_x_K = m_size * K_size
        m_x_k = m_size*k_size
        A_ty = np.ndarray[( m_size * k_size,), dtype_in]

        B_ty = np.ndarray[(k_size,), dtype_in]
        C_ty = np.ndarray[(m_size, ), dtype_out]
            
        zero = external_func("zero_m_int8", inputs=[C_ty])
        matvec = external_func(
            "mv_int8",
            inputs=[A_ty, B_ty, C_ty],
        )
        
        # define the the shim_dma
        
        memref.global_("C_MT_SHM", T.memref( K_size, T.i32() ), sym_visibility="public")
        memref.global_("A_SHM_MT", T.memref(M_size*K_size, T.i8()), sym_visibility="public")
        memref.global_("B_SHM_MT", T.memref(K_size, T.i8()), sym_visibility="public")
        
        shim_dma_allocation("C_MT_SHM", DMAChannelDir.S2MM, 0,0)
        shim_dma_allocation("A_SHM_MT", DMAChannelDir.MM2S, 0, 0)
        shim_dma_allocation("B_SHM_MT", DMAChannelDir.MM2S, 1, 0)
        # Tile declarations
        ShimTile = tile(0,0)
        MemTile = tile(0,1)
        ComputeTile = tile(0,2)
        
    
        # Memtile elements
        inA_mem_prod_lock = lock(MemTile, lock_id= 0, init = 2)
        inA_mem_con_lock = lock(MemTile, lock_id=1, init= 0)
        inA_mem_buffers = [
            buffer( tile=MemTile, datatype=A_ty, name="inA_shm_to_mt_0"),
            buffer( tile=MemTile, datatype=A_ty, name="inA_shm_to_mt_1")
        ]
        
        inB_mem_prod_lock = lock(MemTile, lock_id=2, init=2)
        inB_mem_con_lock = lock(MemTile, lock_id=3, init= 0)
        inB_mem_buffers = [
            buffer(tile=MemTile, datatype=B_ty, name="inB_shim_to_mt_0" ),
            buffer(tile=MemTile, datatype=B_ty, name="inB_shim_to_mt_1" )
        ]
        
        outC_mem_prod_lock = lock(MemTile, lock_id=4, init=2)
        outC_mem_con_lock = lock(MemTile, lock_id=5, init=0)
        outC_mem_buffers = [
            buffer(tile=MemTile, datatype=C_ty, name="outC_mt_to_shim_0"),
            buffer(tile=MemTile, datatype=C_ty, name="outC_mt_to_shim_1")
        ]
        
        
        # CT elements
        inA_CT_prod_lock = lock(ComputeTile, lock_id=0, init=2, sym_name="inA_CT_prod_lock")
        inA_CT_con_lock = lock(ComputeTile, lock_id=1, init=0, sym_name="inA_CT_con_lock")
        inA_CT_buffers  = [
            buffer(tile=ComputeTile, datatype=A_ty, name="inA_mt_to_ct_0"),
            buffer(tile=ComputeTile, datatype=A_ty, name="inA_mt_to_ct_1")
        ]   
        
        inB_CT_prod_lock = lock(ComputeTile, lock_id=2, init=2, sym_name="inB_CT_prod_lock")
        inB_CT_con_lock = lock(ComputeTile, lock_id=3, init=0, sym_name="inB_CT_con_lock")
        inB_CT_buffers = [
            buffer(tile=ComputeTile, datatype=B_ty, name="inB_mt_to_ct_0"),
            buffer(tile=ComputeTile, datatype=B_ty, name="inB_mt_to_ct_1")
        ]
        
        outC_CT_prod_lock = lock(ComputeTile, lock_id=4, init=2,sym_name="outC_CT_prod_lock")
        outC_CT_con_lock = lock(ComputeTile, lock_id=5, init=0, sym_name="outC_CT_con_lock")
        outC_CT_buffers = [
            buffer(tile=ComputeTile, datatype=C_ty, name="in_C_ct_to_mt_0"),
            buffer(tile=ComputeTile, datatype=C_ty, name="in_C_ct_to_mt_1")
        ]
        
        # define the data flow of it
        # Shimtille with Memtile
        flow(ShimTile, WireBundle.DMA, 0, MemTile, WireBundle.DMA, 0) # A input path
        flow(ShimTile, WireBundle.DMA, 1, MemTile, WireBundle.DMA, 1 ) # B input path
        flow(MemTile,  WireBundle.DMA, 2, ShimTile, WireBundle.DMA, 0)
        
        # Memtile vs Computetile
        flow(MemTile, WireBundle.DMA, 0, ComputeTile, WireBundle.DMA, 0)# A
        flow(MemTile, WireBundle.DMA, 1, ComputeTile, WireBundle.DMA, 1)# B
        flow(ComputeTile, WireBundle.DMA, 0, MemTile, WireBundle.DMA, 2) # C
        
        
        # DMA logic for memorytile
        @memtile_dma(MemTile)
        def m(block):
            # Receive a from Shimtile to Mt
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(inA_mem_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers[0], offset=0, len=m_x_k)
                use_lock(inA_mem_con_lock, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inA_mem_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers[1], offset=0, len=m_x_k)
                use_lock(inA_mem_con_lock, LockAction.Release, value=1)
                next_bd(block[1])                
            with block[3]:
                # Reorder A  to Column major orderwhen send from MT to CT
                s1 =dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])  # A in from shimtile
            with block[4]:
                use_lock(inA_mem_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers[0], offset=0, len=m_x_k, dimensions=[  (k_size // 4, 4), (m_size, k_size), (4, 1)]   )
                use_lock(inA_mem_prod_lock, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(inA_mem_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers[1], offset=0, len=m_x_k, dimensions=[ (k_size // 4, 4), (m_size, k_size), (4, 1)]   )
                use_lock(inA_mem_prod_lock, LockAction.Release, value=1)
                next_bd(block[4])
                
            with block[6]:
                # receive B from shimtile
                s2 = dma_start(DMAChannelDir.S2MM, 1, dest=block[7], chain=block[9])
            with block[7]:
                use_lock(inB_mem_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers[0], offset=0, len=k_size)
                use_lock(inB_mem_con_lock, LockAction.Release, value=1)
                next_bd(block[8])
            with block[8]:
                use_lock(inB_mem_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers[1], offset=0, len=k_size)
                use_lock(inB_mem_con_lock, LockAction.Release, value=1)
                next_bd(block[7])
            with block[9]:
                # send B to CT
                s3 = dma_start(DMAChannelDir.MM2S, 1, dest=block[10], chain=block[12])
            with block[10]:
                use_lock(inB_mem_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers[0], offset=0, len=k_size)
                use_lock(inB_mem_prod_lock, LockAction.Release, value=1)
                next_bd(block[11])
            with block[11]:
                use_lock(inB_mem_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers[1], offset=0, len=k_size)
                use_lock(inB_mem_prod_lock, LockAction.Release, value=1)
                next_bd(block[10])
            with block[12]:
                # receive C from Ct
                s4 = dma_start(DMAChannelDir.S2MM, 2, dest=block[13], chain=block[15])
            with block[13]:
                use_lock(outC_mem_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers[0], offset=0, len=k_size)
                use_lock(outC_mem_con_lock, LockAction.Release, value=1)
                next_bd(block[14])
            with block[14]:
                use_lock(outC_mem_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers[1], offset=0, len=k_size)
                use_lock(outC_mem_con_lock, LockAction.Release, value=1)
                next_bd(block[13])
            with block[15]:
                # send C from mt to shimtile
                s5 = dma_start(DMAChannelDir.MM2S, 2, dest=block[16], chain=block[18])
            with block[16]:
                use_lock(outC_mem_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers[0], offset=0, len=k_size)
                use_lock(outC_mem_prod_lock, LockAction.Release, value=1)
                next_bd(block[17])                                                                                
            with block[17]:
                use_lock(outC_mem_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers[1], offset=0, len=k_size)
                use_lock(outC_mem_prod_lock, LockAction.Release, value=1)
                next_bd(block[16])
            with block[18]:
                EndOp()                                                                     

        # memory logic for compute tile
        @mem(ComputeTile)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])

            with block[1]:
                use_lock(inA_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_CT_buffers[0], offset= 0, len = m_x_k)
                use_lock(inA_CT_con_lock, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inA_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_CT_buffers[1], offset= 0, len = m_x_k)
                use_lock(inA_CT_con_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[3]:
                s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
            with block[4]:
                use_lock(inB_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_CT_buffers[0], offset=0, len= k_size)
                use_lock(inB_CT_con_lock, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(inB_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_CT_buffers[1], offset=0, len=k_size)
                use_lock(inB_CT_con_lock, LockAction.Release,value=1)
                next_bd(block[4])      
            with block[6]:
                s2  = dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
            with block[7]: 
                use_lock(outC_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_CT_buffers[0], offset=0, len=k_size)
                use_lock(outC_CT_prod_lock, LockAction.Release, value=1)          
                next_bd(block[8])
            with block[8]:
                use_lock(outC_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_CT_buffers[1], offset=0, len=k_size)
                use_lock(outC_CT_prod_lock, LockAction.Release, value=1)          
                next_bd(block[7])       
            with block[9]:
                EndOp()
        @core(ComputeTile,  "mvm_i8.o")
        def core_body():
            for _ in range_(sys.maxsize):

                def buffer_0():
                    use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                    ele_a = inA_CT_buffers[0]
                    ele_b = inB_CT_buffers[0]
                    matvec(ele_a, ele_b, ele_out)
                    use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                    use_lock(inB_CT_prod_lock, LockAction.Release,value=1)  
                def buffer_1():
                    use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                    ele_a = inA_CT_buffers[1]
                    ele_b = inB_CT_buffers[1]
                    matvec(ele_a, ele_b, ele_out)
                    use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                    use_lock(inB_CT_prod_lock, LockAction.Release,value=1)  
                    
                use_lock(outC_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                ele_out = outC_CT_buffers[0]
                zero(ele_out)
                for k in range(K_div_k):
                    # if (k %2) == 0:
                    #     buffer_0()
                    # else:   
                    #     buffer_1()
                    if(k == 0):
                        buffer_0()
                    elif(k == 1):
                        buffer_1()
                    elif(k == 2):
                        buffer_0()
                    else:
                        buffer_1()
                
                # buffer_0()
                # buffer_1()
                # buffer_0()
                # buffer_1()
                use_lock(outC_CT_con_lock, LockAction.Release,value=1)
                
                
                
                
                
                
                use_lock(outC_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                ele_out = outC_CT_buffers[1]
                zero(ele_out)

                buffer_0()
                buffer_1()
                buffer_0()
                buffer_1()
                use_lock(outC_CT_con_lock, LockAction.Release,value=1) 
                    
                # use_lock(outC_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_out = outC_CT_buffers[0]
                # zero(ele_out)

                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[0]
                # ele_b = inB_CT_buffers[0]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1)  
                    
                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[1]
                # ele_b = inB_CT_buffers[1]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1) 

                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[0]
                # ele_b = inB_CT_buffers[0]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1)  
                    
                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[1]
                # ele_b = inB_CT_buffers[1]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1)                     
                    
                # use_lock(outC_CT_con_lock, LockAction.Release,value=1)
                
                
                
                
                
                
                # use_lock(outC_CT_prod_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_out = outC_CT_buffers[1]
                # zero(ele_out)

                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[0]
                # ele_b = inB_CT_buffers[0]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1)  
                    
                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[1]
                # ele_b = inB_CT_buffers[1]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1) 

                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[0]
                # ele_b = inB_CT_buffers[0]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1)  
                    
                # use_lock(inA_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # use_lock(inB_CT_con_lock, LockAction.AcquireGreaterEqual, value=1)
                # ele_a = inA_CT_buffers[1]
                # ele_b = inB_CT_buffers[1]
                # matvec(ele_a, ele_b, ele_out)
                # use_lock(inA_CT_prod_lock, LockAction.Release,value=1)
                # use_lock(inB_CT_prod_lock, LockAction.Release,value=1)   
                # use_lock(outC_CT_con_lock, LockAction.Release,value=1)
                
                
        @runtime_sequence( np.ndarray[(M_size*K_size,), dtype_in],  np.ndarray[(K_size,), dtype_in],  np.ndarray[(M_size,), dtype_out])
        def sequence(A, B, C):
                # r = M // m 
                # assert r * m  == M
                # rtp_buffer[0] = 1 TODO
                npu_dma_memcpy_nd(
                    metadata="B_SHM_MT",
                    bd_id=2,
                    mem=B,
                    offsets=[0, 0, 0, 0],
                    sizes=[M_size // m_size , 1, 1, K_size],
                    strides=[0, 0, 0, 1],
                )

                # M offset: each column handles M // mvm_cols row in total
                A_offset = 0
                C_offset = 0
                npu_dma_memcpy_nd(
                    metadata="A_SHM_MT",
                    bd_id=1,
                    mem=A,
                    offsets=[0, 0, 0, A_offset],
                    sizes=[M_size  // (m_size), K_div_k,  m_size, k_size],
                    strides=[m_x_K, k_size, K_size, 1],
                )
                npu_dma_memcpy_nd(
                    metadata="C_MT_SHM",
                    bd_id=0,
                    mem=C,
                    
                    offsets=[0, 0, 0, C_offset],
                    sizes=[1, 1, M_size // m_size, m_size],
                    strides=[0, 0,  m_size, 1],
                )
                npu_dma_wait("C_MT_SHM")
    
with mlir_mod_ctx() as ctx:
    single_mat_vect_mult()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
