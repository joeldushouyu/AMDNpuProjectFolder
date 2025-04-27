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
from custom_npu_dma_memcpy import generate_packet_attribute

        
# def with_block_unroll(block, chain):
#     for idx, prod_locks, buf, off, len, con_locks, nxt_idx in chain:
#         with block[idx]:
#             for p_lock in prod_locks:
#                 use_lock(p_lock, LockAction.AcquireGreaterEqual, value=1)
#             dma_bd( buf, offset=off, len=len) # only allow one dma_buffers in each with block
#             for c_lock in con_locks:
#                 use_lock(c_lock, LockAction.Release, value=1)
#             next_bd(block[nxt_idx])


# def with_block_unroll_with_packet_out(block, chain):
#     for idx, prod_locks, buf, off, len, con_locks, nxt_idx in chain:
#         with block[idx]:
#             for p_lock in prod_locks:
#                 use_lock(p_lock, LockAction.AcquireGreaterEqual, value=1)
#             # dma_bd_packet(packet_id=3, packet_type=0)
#             dma_bd( buf, offset=off, len=len, packet=generate_packet_attribute(packet_id=3, packet_type=0)) # only allow one dma_buffers in each with block
#             for c_lock in con_locks:
#                 use_lock(c_lock, LockAction.Release, value=1)
#             next_bd(block[nxt_idx])



def with_block_unroll_with_optional_packet_header(block, chain):
    for idx, prod_locks, buf, off, len, con_locks, nxt_idx, packet_id_type in chain:
        with block[idx]:
            for p_lock in prod_locks:
                use_lock(p_lock, LockAction.AcquireGreaterEqual, value=1)
            packet = None if packet_id_type == []  else generate_packet_attribute(packet_id=packet_id_type[0], packet_type=packet_id_type[1] )
            dma_bd( buf, offset=off, len=len, packet=packet) # only allow one dma_buffers in each with block
            for c_lock in con_locks:
                use_lock(c_lock, LockAction.Release, value=1)
            next_bd(block[nxt_idx])


def handle_dma_sequences(block, chain0, chain1, chain2, chain0_start_end: tuple, chain1_start_end:tuple, chain2_start_end:tuple):

    # # block_idx, acqire_locks, buffer, buffer_offset, buffer_len, release_locks, next_idx      
    # chain0 = [
    #     (1, [switch_diode_prod_lock], switch_diode_buffer[0], 0, buffer_size_of_switch_diode, [switch_diode_con_lock], 2 ),
    #     (2, [A_B_buffer_prod_lock],   A_B_buffer[0], 0, buffer_size_of_A_B_matrix, [A_B_buffer_con_lock], 3   ),
    #     (3, [C_D_imp_buffer_prod_lock], C_D_imp_buffer[0], 0, buffer_size_of_C_D_imp_non, [C_D_imp_buffer_con_lock], 4  ),
    #     (4, [C_D_non_imp_buffer_prod_lock], C_D_non_imp_buffer[0], 0, buffer_size_of_C_D_imp_non,  [C_D_non_imp_buffer_con_lock], 1 )
    # ]
    s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[chain0_start_end[0]], chain=block[chain0_start_end[1]])
    with_block_unroll_with_optional_packet_header(block=block, chain=chain0)
    
    # chain1 = [
    #     (6, [in_buffer_prod_lock],  in_buffer[0],   0,                          buffer_size_of_in_ping_pong, [in_buffer_con_lock], 7),
    #     (7, [in_buffer_prod_lock],  in_buffer[0],   buffer_size_of_in_ping_pong, buffer_size_of_in_ping_pong, [in_buffer_con_lock], 6),
    # ]
    with block[chain0_start_end[1]]:
        s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[chain1_start_end[0]], chain=block[chain1_start_end[1]])
    with_block_unroll_with_optional_packet_header(block=block, chain=chain1)


    # chain2 = [
    #     (9,  [out_buffer_con_lock], out_buffer[0],       0,                          buffer_size_of_out_ping_pong, [out_buffer_prod_lock], 10),
    #     (10, [out_buffer_con_lock], out_buffer[0],       buffer_size_of_out_ping_pong, buffer_size_of_out_ping_pong, [out_buffer_prod_lock],  9),
    # ]
    with block[chain1_start_end[1]]:
        s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[chain2_start_end[0]], chain=block[chain2_start_end[1]])
    with_block_unroll_with_optional_packet_header(block=block, chain=chain2)
    with block[chain2_start_end[1]]:
        EndOp()            