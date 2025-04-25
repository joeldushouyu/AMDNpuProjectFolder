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


        
def with_block_unroll(block, chain):
    for idx, prod_locks, buf, off, len, con_locks, nxt_idx in chain:
        with block[idx]:
            for p_lock in prod_locks:
                use_lock(p_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd( buf, offset=off, len=len) # only allow one dma_buffers in each with block
            for c_lock in con_locks:
                use_lock(c_lock, LockAction.Release, value=1)
            next_bd(block[nxt_idx])

def handle_dma_sequences(block, chain0, chain1, chain2):

    # # block_idx, acqire_locks, buffer, buffer_offset, buffer_len, release_locks, next_idx      
    # chain0 = [
    #     (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, buffer_size_of_switch_diode, [switch_diode_con_lock], 2 ),
    #     (2, [A_B_buffer_prod_lock],   A_B_buffer[0], 0, buffer_size_of_A_B_matrix, [A_B_buffer_con_lock], 3   ),
    #     (3, [C_D_imp_buffer_prod_lock], C_D_imp_buffer[0], 0, buffer_size_of_C_D_imp_non, [C_D_imp_buffer_con_lock], 4  ),
    #     (4, [C_D_non_imp_buffer_prod_lock], C_D_non_imp_buffer[0], 0, buffer_size_of_C_D_imp_non,  [C_D_non_imp_buffer_con_lock], 1 )
    # ]
    s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[5])
    with_block_unroll(block=block, chain=chain0)
    
    # chain1 = [
    #     (6, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock], 7),
    #     (7, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock], 6),
    # ]
    with block[5]:
        s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[6], chain=block[8])
    with_block_unroll(block=block, chain=chain1)


    # chain2 = [
    #     (9,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 10),
    #     (10, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock],  9),
    # ]
    with block[8]:
        s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[9], chain=block[11])
    with_block_unroll(block=block, chain=chain2)
    with block[11]:
        EndOp()            