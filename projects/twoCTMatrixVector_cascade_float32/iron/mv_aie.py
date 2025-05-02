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
    

    
    dtype_in = np.dtype[np.float32]
    dtype_out = np.dtype[np.float32]
    
    
    @device(AIEDevice.npu2)
    
    def device_body():
        t_size = 16
        M_size = 128
        K_size = 1024
        m_size = 64
        k_size = 64
        K_div_k = K_size//k_size
        m_x_K = m_size * K_size
        m_x_k = m_size*k_size
        In_Matrix_ty = np.ndarray[( m_size * k_size,), dtype_in]

        IN_Vector_ty = np.ndarray[(k_size,), dtype_in]
        Out_Vector_ty = np.ndarray[(m_size, ), dtype_out]
            
        zero = external_func("zero_m_float32", inputs=[Out_Vector_ty])
        matvec = external_func(
            "mv_float32",
            inputs=[In_Matrix_ty, IN_Vector_ty, Out_Vector_ty],
        )
        
        # define the the shim_dma
        
        memref.global_("Vector_MT0_SHM0", T.memref( K_size, T.f32() ), sym_visibility="public")
        memref.global_("Matrix_SHM0_MT0", T.memref(M_size*K_size, T.f32()), sym_visibility="public")
        memref.global_("Vector_SHM0_MT0", T.memref(K_size, T.f32()), sym_visibility="public")
        shim_dma_allocation("Vector_MT0_SHM0", DMAChannelDir.S2MM, 0,0)
        shim_dma_allocation("Matrix_SHM0_MT0", DMAChannelDir.MM2S, 0, 0)
        shim_dma_allocation("Vector_SHM0_MT0", DMAChannelDir.MM2S, 1, 0)

        memref.global_("Vector_MT1_SHM1", T.memref(K_size, T.f32()),  sym_visibility="public")
        memref.global_("Matrix_SHM1_MT1", T.memref(M_size*K_size, T.f32()), sym_visibility="public")
        memref.global_("Vector_SHM1_MT1", T.memref(K_size, T.f32()), sym_visibility="public")
        shim_dma_allocation("Vector_MT1_SHM1", DMAChannelDir.S2MM, 0, 1)
        shim_dma_allocation("Matrix_SHM1_MT1",DMAChannelDir.MM2S, 0, 1)
        shim_dma_allocation("Vector_SHM1_MT1", DMAChannelDir.MM2S, 1, 1)
        
        # Tile declarations
        ShimTile_0 = tile(0,0)
        MemTile_0 = tile(0,1)
        ComputeTile_0 = tile(0,2)
        ShimTile_1 = tile(1,0)
        MemTile_1 = tile(1,1)
        ComputeTile_1 = tile(1,2)
        
    
        # Memtile elements
        inA_mem_prod_lock_MT0 = lock(MemTile_0, lock_id= 0, init = 2)
        inA_mem_con_lock_MT0 = lock(MemTile_0, lock_id=1, init= 0)
        inA_mem_buffers_MT0 = [
            buffer( tile=MemTile_0, datatype=In_Matrix_ty, name="inA_mem_buffers_MT0_0"),
            buffer( tile=MemTile_0, datatype=In_Matrix_ty, name="inA_mem_buffers_MT0_1")
        ]
        
        inB_mem_prod_lock_MT0 = lock(MemTile_0, lock_id=2, init=2)
        inB_mem_con_lock_MT0 = lock(MemTile_0, lock_id=3, init= 0)
        inB_mem_buffers_MT0 = [
            buffer(tile=MemTile_0, datatype=IN_Vector_ty, name="inB_mem_buffers_MT0_0" ),
            buffer(tile=MemTile_0, datatype=IN_Vector_ty, name="inB_mem_buffers_MT0_1" )
        ]
        
        outC_mem_prod_lock_MT0 = lock(MemTile_0, lock_id=4, init=2)
        outC_mem_con_lock_MT0 = lock(MemTile_0, lock_id=5, init=0)
        outC_mem_buffers_MT0 = [
            buffer(tile=MemTile_0, datatype=Out_Vector_ty, name="outC_mem_buffers_MT0_0"),
            buffer(tile=MemTile_0, datatype=Out_Vector_ty, name="outC_mem_buffers_MT0_1")
        ]
        
        
        # CT elements
        inA_CT_prod_lock_CT_0_0 = lock(ComputeTile_0, lock_id=0, init=2, sym_name="inA_CT_prod_lock_CT_0_0")
        inA_CT_con_lock_CT_0_0 = lock(ComputeTile_0, lock_id=1, init=0, sym_name="inA_CT_con_lock_CT_0_0")
        inA_CT_buffers_CT_0_0  = [
            buffer(tile=ComputeTile_0, datatype=In_Matrix_ty, name="inA_CT_buffers_CT_0_0_0"),
            buffer(tile=ComputeTile_0, datatype=In_Matrix_ty, name="inA_CT_buffers_CT_0_0_1")
        ]   
        
        inB_CT_prod_lock_CT_0_0 = lock(ComputeTile_0, lock_id=2, init=2, sym_name="inB_CT_prod_lock_CT_0_0")
        inB_CT_con_lock_CT_0_0 = lock(ComputeTile_0, lock_id=3, init=0, sym_name="inB_CT_con_lock_CT_0_0")
        inB_CT_buffers_CT_0_0 = [
            buffer(tile=ComputeTile_0, datatype=IN_Vector_ty, name="inB_CT_buffers_CT_0_0_0"),
            buffer(tile=ComputeTile_0, datatype=IN_Vector_ty, name="inB_CT_buffers_CT_0_0_1")
        ]
        
        outC_CT_prod_lock_CT_0_0 = lock(ComputeTile_0, lock_id=4, init=2,sym_name="outC_CT_prod_lock_CT_0_0")
        outC_CT_con_lock_CT_0_0 = lock(ComputeTile_0, lock_id=5, init=0, sym_name="outC_CT_con_lock_CT_0_0")
        outC_CT_buffers_CT_0_0 = [
            buffer(tile=ComputeTile_0, datatype=Out_Vector_ty, name="outC_CT_buffers_CT_0_0_0"),
            buffer(tile=ComputeTile_0, datatype=Out_Vector_ty, name="outC_CT_buffers_CT_0_0_1")
        ]
        
        
        # Element for MT2
        inA_mem_prod_lock_MT1 = lock(MemTile_1, lock_id=0, init=2)
        inA_mem_con_lock_MT1 = lock(MemTile_1, lock_id=1, init=0)
        inA_mem_buffers_MT1 = [
            buffer( tile=MemTile_1, datatype=In_Matrix_ty, name="inA_mem_buffers_MT1_0"),
            buffer( tile=MemTile_1, datatype=In_Matrix_ty, name="inA_mem_buffers_MT1_1")
        ]
        
        inB_mem_prod_lock_MT1 = lock(MemTile_1, lock_id=2, init=2)
        inB_mem_con_lock_MT1 = lock(MemTile_1, lock_id=3, init=0)
        inB_mem_buffers_MT1 = [
            buffer(tile=MemTile_1, datatype=IN_Vector_ty, name="inB_mem_buffers_MT1_0" ),
            buffer(tile=MemTile_1, datatype=IN_Vector_ty, name="inB_mem_buffers_MT1_1" )
        ]
        
        outC_mem_prod_lock_MT1 = lock(MemTile_1, lock_id=4, init=2)
        outC_mem_con_lock_MT1 = lock(MemTile_1, lock_id=5, init=0)
        outC_mem_buffers_MT1 = [
            buffer(tile=MemTile_1, datatype=Out_Vector_ty, name="outC_mem_buffers_MT1_0"),
            buffer(tile=MemTile_1, datatype=Out_Vector_ty, name="outC_mem_buffers_MT1_1")
        ]
        
        #CT 2 elements
        inA_CT_prod_lock_CT_0_1 = lock(ComputeTile_1, lock_id=0, init=2, sym_name="inA_CT_prod_lock_CT_0_1")
        inA_CT_con_lock_CT_0_1 = lock(ComputeTile_1, lock_id=1, init=0, sym_name="inA_CT_con_lock_CT_0_1")
        inA_CT_buffers_CT_0_1 = [
            buffer(tile=ComputeTile_1, datatype=In_Matrix_ty, name="inA_CT_buffers_CT_0_1_0"),
            buffer(tile=ComputeTile_1, datatype=In_Matrix_ty, name="inA_CT_buffers_CT_0_1_1")
        ]
        
        inB_CT_prod_lock_CT_0_1 = lock(ComputeTile_1, lock_id=2, init=2, sym_name="inB_CT_prod_lock_CT_0_1")
        inB_CT_con_lock_CT_0_1 = lock(ComputeTile_1, lock_id=3, init=0, sym_name="inB_CT_con_lock_CT_0_1")
        inB_CT_buffers_CT_0_1 = [
            buffer(tile=ComputeTile_1, datatype=IN_Vector_ty, name="inB_CT_buffers_CT_0_1_0"),
            buffer(tile=ComputeTile_1, datatype=IN_Vector_ty, name="inB_CT_buffers_CT_0_1_1")
        ]
        
        outC_CT_prod_lock_CT_0_1 = lock(ComputeTile_1, lock_id=4, init=2, sym_name="outC_CT_prod_lock_CT_0_1")
        outC_CT_con_lock_CT_0_1 = lock(ComputeTile_1, lock_id=5, init=0, sym_name="outC_CT_con_lock_CT_0_1")
        outC_CT_buffers_CT_0_1 = [
            buffer(tile=ComputeTile_1, datatype=Out_Vector_ty, name="outC_CT_buffers_CT_0_1_0"),
            buffer(tile=ComputeTile_1, datatype=Out_Vector_ty, name="outC_CT_buffers_CT_0_1_1")
        ]
        
        # define the data flow of it
        # Shimtille with Memtile
        flow(ShimTile_0, WireBundle.DMA, 0, MemTile_0, WireBundle.DMA, 0) # A input path
        flow(ShimTile_0, WireBundle.DMA, 1, MemTile_0, WireBundle.DMA, 1 ) # B input path
        flow(MemTile_0,  WireBundle.DMA, 2, ShimTile_0, WireBundle.DMA, 0)
        
        # Memtile vs Computetile
        flow(MemTile_0, WireBundle.DMA, 0, ComputeTile_0, WireBundle.DMA, 0)# A
        flow(MemTile_0, WireBundle.DMA, 1, ComputeTile_0, WireBundle.DMA, 1)# B
        flow(ComputeTile_0, WireBundle.DMA, 0, MemTile_0, WireBundle.DMA, 2) # C
        
        # Shimtille with Memtile
        flow(ShimTile_1, WireBundle.DMA, 0, MemTile_1, WireBundle.DMA, 0) # A input path
        flow(ShimTile_1, WireBundle.DMA, 1, MemTile_1, WireBundle.DMA, 1 ) # B input path
        flow(MemTile_1,  WireBundle.DMA, 2, ShimTile_1, WireBundle.DMA, 0)
        
        # Memtile vs Computetile
        flow(MemTile_1, WireBundle.DMA, 0, ComputeTile_1, WireBundle.DMA, 0)# A
        flow(MemTile_1, WireBundle.DMA, 1, ComputeTile_1, WireBundle.DMA, 1)# B
        flow(ComputeTile_1, WireBundle.DMA, 0, MemTile_1, WireBundle.DMA, 2) # C
        
                

        # DMA logic for memorytile
        @memtile_dma(MemTile_0)
        def m(block):
            # Receive a from Shimtile to Mt
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(inA_mem_prod_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT0[0], offset=0, len=m_x_k)
                use_lock(inA_mem_con_lock_MT0, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inA_mem_prod_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT0[1], offset=0, len=m_x_k)
                use_lock(inA_mem_con_lock_MT0, LockAction.Release, value=1)
                next_bd(block[1])                
            with block[3]:
                # Reorder A  to Column major orderwhen send from MT to CT
                s1 =dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])  # A in from shimtile
            with block[4]:
                use_lock(inA_mem_con_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT0[0], offset=0, len=m_x_k, dimensions=[  (k_size // t_size, t_size*k_size   ), (k_size, 1), (t_size, k_size)]   )  
                use_lock(inA_mem_prod_lock_MT0, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(inA_mem_con_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT0[1], offset=0, len=m_x_k, dimensions=[  (k_size // t_size, t_size*k_size   ), (k_size, 1), (t_size, k_size)]   )
                use_lock(inA_mem_prod_lock_MT0, LockAction.Release, value=1)
                next_bd(block[4])
                
            with block[6]:
                # receive B from shimtile
                s2 = dma_start(DMAChannelDir.S2MM, 1, dest=block[7], chain=block[9])
            with block[7]:
                use_lock(inB_mem_prod_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT0[0], offset=0, len=k_size)
                use_lock(inB_mem_con_lock_MT0, LockAction.Release, value=1)
                next_bd(block[8])
            with block[8]:
                use_lock(inB_mem_prod_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT0[1], offset=0, len=k_size)
                use_lock(inB_mem_con_lock_MT0, LockAction.Release, value=1)
                next_bd(block[7])
            with block[9]:
                # send B to CT
                s3 = dma_start(DMAChannelDir.MM2S, 1, dest=block[10], chain=block[12])
            with block[10]:
                use_lock(inB_mem_con_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT0[0], offset=0, len=k_size)
                use_lock(inB_mem_prod_lock_MT0, LockAction.Release, value=1)
                next_bd(block[11])
            with block[11]:
                use_lock(inB_mem_con_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT0[1], offset=0, len=k_size)
                use_lock(inB_mem_prod_lock_MT0, LockAction.Release, value=1)
                next_bd(block[10])
            with block[12]:
                # receive C from Ct
                s4 = dma_start(DMAChannelDir.S2MM, 2, dest=block[13], chain=block[15])
            with block[13]:
                use_lock(outC_mem_prod_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT0[0], offset=0, len=k_size)
                use_lock(outC_mem_con_lock_MT0, LockAction.Release, value=1)
                next_bd(block[14])
            with block[14]:
                use_lock(outC_mem_prod_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT0[1], offset=0, len=k_size)
                use_lock(outC_mem_con_lock_MT0, LockAction.Release, value=1)
                next_bd(block[13])
            with block[15]:
                # send C from mt to shimtile
                s5 = dma_start(DMAChannelDir.MM2S, 2, dest=block[16], chain=block[18])
            with block[16]:
                use_lock(outC_mem_con_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT0[0], offset=0, len=k_size)
                use_lock(outC_mem_prod_lock_MT0, LockAction.Release, value=1)
                next_bd(block[17])                                                                                
            with block[17]:
                use_lock(outC_mem_con_lock_MT0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT0[1], offset=0, len=k_size)
                use_lock(outC_mem_prod_lock_MT0, LockAction.Release, value=1)
                next_bd(block[16])
            with block[18]:
                EndOp()                                                                     

        # memory logic for compute tile
        @mem(ComputeTile_0)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])

            with block[1]:
                use_lock(inA_CT_prod_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_CT_buffers_CT_0_0[0], offset= 0, len = m_x_k)
                use_lock(inA_CT_con_lock_CT_0_0, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inA_CT_prod_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_CT_buffers_CT_0_0[1], offset= 0, len = m_x_k)
                use_lock(inA_CT_con_lock_CT_0_0, LockAction.Release, value=1)
                next_bd(block[1])
            with block[3]:
                s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
            with block[4]:
                use_lock(inB_CT_prod_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_CT_buffers_CT_0_0[0], offset=0, len= k_size)
                use_lock(inB_CT_con_lock_CT_0_0, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(inB_CT_prod_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_CT_buffers_CT_0_0[1], offset=0, len=k_size)
                use_lock(inB_CT_con_lock_CT_0_0, LockAction.Release,value=1)
                next_bd(block[4])      
            with block[6]:
                s2  = dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
            with block[7]: 
                use_lock(outC_CT_con_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_CT_buffers_CT_0_0[0], offset=0, len=k_size)
                use_lock(outC_CT_prod_lock_CT_0_0, LockAction.Release, value=1)          
                next_bd(block[8])
            with block[8]:
                use_lock(outC_CT_con_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_CT_buffers_CT_0_0[1], offset=0, len=k_size)
                use_lock(outC_CT_prod_lock_CT_0_0, LockAction.Release, value=1)          
                next_bd(block[7])       
            with block[9]:
                EndOp()
        @core(ComputeTile_0,  "mv_float.o")
        def core_body():
            for _ in range_(sys.maxsize):

                def buffer_0():
                    use_lock(inA_CT_con_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(inB_CT_con_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                    ele_a = inA_CT_buffers_CT_0_0[0]
                    ele_b = inB_CT_buffers_CT_0_0[0]
                    matvec(ele_a, ele_b, ele_out)
                    use_lock(inA_CT_prod_lock_CT_0_0, LockAction.Release,value=1)
                    use_lock(inB_CT_prod_lock_CT_0_0, LockAction.Release,value=1)  
                def buffer_1():
                    use_lock(inA_CT_con_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(inB_CT_con_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                    ele_a = inA_CT_buffers_CT_0_0[1]
                    ele_b = inB_CT_buffers_CT_0_0[1]
                    matvec(ele_a, ele_b, ele_out)
                    use_lock(inA_CT_prod_lock_CT_0_0, LockAction.Release,value=1)
                    use_lock(inB_CT_prod_lock_CT_0_0, LockAction.Release,value=1)  
                    
                use_lock(outC_CT_prod_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                ele_out = outC_CT_buffers_CT_0_0[0]
                zero(ele_out)
                
                # it repeats for K_SIZE/512 timesd 
                assert (K_size//(k_size)) % 2 == 0 
                repeat_ping_ping =  (K_size // k_size) // 2
                for _ in range_(repeat_ping_ping):
                    buffer_0()
                    buffer_1()
           
                use_lock(outC_CT_con_lock_CT_0_0, LockAction.Release,value=1)
                

                use_lock(outC_CT_prod_lock_CT_0_0, LockAction.AcquireGreaterEqual, value=1)
                ele_out = outC_CT_buffers_CT_0_0[1]
                zero(ele_out)
                for _ in range_(repeat_ping_ping):
                    buffer_0()
                    buffer_1()
              
                use_lock(outC_CT_con_lock_CT_0_0, LockAction.Release,value=1) 

                
                
        @memtile_dma(MemTile_1)
        def m(block):
            # Receive a from Shimtile to Mt
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(inA_mem_prod_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT1[0], offset=0, len=m_x_k)
                use_lock(inA_mem_con_lock_MT1, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inA_mem_prod_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT1[1], offset=0, len=m_x_k)
                use_lock(inA_mem_con_lock_MT1, LockAction.Release, value=1)
                next_bd(block[1])                
            with block[3]:
                # Reorder A  to Column major orderwhen send from MT to CT
                s1 =dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])  # A in from shimtile
            with block[4]:
                use_lock(inA_mem_con_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT1[0], offset=0, len=m_x_k, dimensions=[  (k_size // t_size, t_size*k_size   ), (k_size, 1), (t_size, k_size)]   )  
                use_lock(inA_mem_prod_lock_MT1, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(inA_mem_con_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_mem_buffers_MT1[1], offset=0, len=m_x_k, dimensions=[  (k_size // t_size, t_size*k_size   ), (k_size, 1), (t_size, k_size)]   )
                use_lock(inA_mem_prod_lock_MT1, LockAction.Release, value=1)
                next_bd(block[4])
            with block[6]:
                # receive B from shimtile
                s2 = dma_start(DMAChannelDir.S2MM, 1, dest=block[7], chain=block[9])
            with block[7]:
                use_lock(inB_mem_prod_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT1[0], offset=0, len=k_size)
                use_lock(inB_mem_con_lock_MT1, LockAction.Release, value=1)
                next_bd(block[8])
            with block[8]:
                use_lock(inB_mem_prod_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT1[1], offset=0, len=k_size)
                use_lock(inB_mem_con_lock_MT1, LockAction.Release, value=1)
                next_bd(block[7])
            with block[9]:
                # send B to CT
                s3 = dma_start(DMAChannelDir.MM2S, 1, dest=block[10], chain=block[12])
            with block[10]:
                use_lock(inB_mem_con_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT1[0], offset=0, len=k_size)
                use_lock(inB_mem_prod_lock_MT1, LockAction.Release, value=1)
                next_bd(block[11])
            with block[11]:
                use_lock(inB_mem_con_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_mem_buffers_MT1[1], offset=0, len=k_size)
                use_lock(inB_mem_prod_lock_MT1, LockAction.Release, value=1)
                next_bd(block[10])
            with block[12]:
                # receive C from Ct
                s4 = dma_start(DMAChannelDir.S2MM, 2, dest=block[13], chain=block[15])
            with block[13]:
                use_lock(outC_mem_prod_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT1[0], offset=0, len=k_size)
                use_lock(outC_mem_con_lock_MT1, LockAction.Release, value=1)
                next_bd(block[14])
            with block[14]:
                use_lock(outC_mem_prod_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT1[1], offset=0, len=k_size)
                use_lock(outC_mem_con_lock_MT1, LockAction.Release, value=1)
                next_bd(block[13])
            with block[15]:
                # send C from mt to shimtile
                s5 = dma_start(DMAChannelDir.MM2S, 2, dest=block[16], chain=block[18])
            with block[16]:
                use_lock(outC_mem_con_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT1[0], offset=0, len=k_size)
                use_lock(outC_mem_prod_lock_MT1, LockAction.Release, value=1)
                next_bd(block[17])                                                                                
            with block[17]:
                use_lock(outC_mem_con_lock_MT1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_mem_buffers_MT1[1], offset=0, len=k_size)
                use_lock(outC_mem_prod_lock_MT1, LockAction.Release, value=1)
                next_bd(block[16])
            with block[18]:
                EndOp()                                                                     
        # memory logic for compute tile
        @mem(ComputeTile_1)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])

            with block[1]:
                use_lock(inA_CT_prod_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_CT_buffers_CT_0_1[0], offset= 0, len = m_x_k)
                use_lock(inA_CT_con_lock_CT_0_1, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(inA_CT_prod_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inA_CT_buffers_CT_0_1[1], offset= 0, len = m_x_k)
                use_lock(inA_CT_con_lock_CT_0_1, LockAction.Release, value=1)
                next_bd(block[1])
            with block[3]:
                s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
            with block[4]:
                use_lock(inB_CT_prod_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_CT_buffers_CT_0_1[0], offset=0, len= k_size)
                use_lock(inB_CT_con_lock_CT_0_1, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(inB_CT_prod_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(inB_CT_buffers_CT_0_1[1], offset=0, len=k_size)
                use_lock(inB_CT_con_lock_CT_0_1, LockAction.Release,value=1)
                next_bd(block[4])      
            with block[6]:
                s2  = dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
            with block[7]: 
                use_lock(outC_CT_con_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_CT_buffers_CT_0_1[0], offset=0, len=k_size)
                use_lock(outC_CT_prod_lock_CT_0_1, LockAction.Release, value=1)          
                next_bd(block[8])
            with block[8]:
                use_lock(outC_CT_con_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(outC_CT_buffers_CT_0_1[1], offset=0, len=k_size)
                use_lock(outC_CT_prod_lock_CT_0_1, LockAction.Release, value=1)          
                next_bd(block[7])       
            with block[9]:
                EndOp()
                
                
        @core(ComputeTile_1,  "mv_float.o")
        def core_body():
            for _ in range_(sys.maxsize):

                def buffer_0():
                    use_lock(inA_CT_con_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(inB_CT_con_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                    ele_a = inA_CT_buffers_CT_0_1[0]
                    ele_b = inB_CT_buffers_CT_0_1[0]
                    matvec(ele_a, ele_b, ele_out)
                    use_lock(inA_CT_prod_lock_CT_0_1, LockAction.Release,value=1)
                    use_lock(inB_CT_prod_lock_CT_0_1, LockAction.Release,value=1)  
                def buffer_1():
                    use_lock(inA_CT_con_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                    use_lock(inB_CT_con_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                    ele_a = inA_CT_buffers_CT_0_1[1]
                    ele_b = inB_CT_buffers_CT_0_1[1]
                    matvec(ele_a, ele_b, ele_out)
                    use_lock(inA_CT_prod_lock_CT_0_1, LockAction.Release,value=1)
                    use_lock(inB_CT_prod_lock_CT_0_1, LockAction.Release,value=1)  
                    
                use_lock(outC_CT_prod_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                ele_out = outC_CT_buffers_CT_0_1[0]
                zero(ele_out)
                
                # it repeats for K_SIZE/512 timesd 
                assert (K_size//(k_size)) % 2 == 0 
                repeat_ping_ping =  (K_size // k_size) // 2
                for _ in range_(repeat_ping_ping):
                    buffer_0()
                    buffer_1()
           
                use_lock(outC_CT_con_lock_CT_0_1, LockAction.Release,value=1)
                

                use_lock(outC_CT_prod_lock_CT_0_1, LockAction.AcquireGreaterEqual, value=1)
                ele_out = outC_CT_buffers_CT_0_1[1]
                zero(ele_out)
                for _ in range_(repeat_ping_ping):
                    buffer_0()
                    buffer_1()
              
                use_lock(outC_CT_con_lock_CT_0_1, LockAction.Release,value=1) 

                
        @runtime_sequence( np.ndarray[(M_size*K_size,), dtype_in],  np.ndarray[(K_size,), dtype_in],  np.ndarray[(M_size,), dtype_out],
                          
                          np.ndarray[(M_size*K_size,), dtype_in],  np.ndarray[(K_size,), dtype_in])
                        
        def sequence(A1, B1, C1, A2, B2):
                # r = M // m 
                # assert r * m  == M
                # rtp_buffer[0] = 1 TODO
                npu_dma_memcpy_nd(
                    metadata="Vector_SHM0_MT0",
                    bd_id=2,
                    mem=B1,
                    offsets=[0, 0, 0, 0],
                    sizes=[M_size // m_size , 1, 1, K_size],
                    strides=[0, 0, 0, 1],
                )

                # M offset: each column handles M // mvm_cols row in total
                A_offset = 0
                C_offset = 0
                npu_dma_memcpy_nd(
                    metadata="Matrix_SHM0_MT0",
                    bd_id=1,
                    mem=A1,
                
                    offsets=[0, 0, 0, A_offset],
                    sizes=[M_size  // (m_size), K_div_k,  m_size, k_size],
                    strides=[m_x_K, k_size, K_size, 1],
                )
                npu_dma_memcpy_nd(
                    metadata="Vector_MT0_SHM0",
                    bd_id=0,
                    mem=C1,
                    
                    offsets=[0, 0, 0, C_offset],
                    sizes=[1, 1, M_size // m_size, m_size],  #TODO:
                    strides=[0, 0,  m_size, 1],
                )
                
            
                # M offset: each column handles M // mvm_cols row in total
                A_offset = 0
                C_offset = M_size # send the result of CT2 to the second half of C
                B_offset = 0
                npu_dma_memcpy_nd(
                    metadata="Vector_SHM1_MT1",
                    bd_id=5,
                    mem=B2,
                    offsets=[0, 0, 0, B_offset ],
                    sizes=[M_size // m_size , 1, 1, K_size],
                    strides=[0, 0, 0, 1],
                )

 
                npu_dma_memcpy_nd(
                    metadata="Matrix_SHM1_MT1",
                    bd_id=4,
                    mem=A2,
                
                    offsets=[0, 0, 0, A_offset],
                    sizes=[M_size  // (m_size), K_div_k,  m_size, k_size],
                    strides=[m_x_K, k_size, K_size, 1],
                )
                npu_dma_memcpy_nd(
                    metadata="Vector_MT1_SHM1",
                    bd_id=3,
                    mem=C1,
                    
                    offsets=[0, 0, 0, C_offset],
                    sizes=[1, 1, M_size // m_size, m_size],  #TODO:
                    strides=[0, 0,  m_size, 1],
                )
                npu_dma_wait("Vector_MT0_SHM0")
                npu_dma_wait("Vector_MT1_SHM1")
    
with mlir_mod_ctx() as ctx:
    single_mat_vect_mult()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
