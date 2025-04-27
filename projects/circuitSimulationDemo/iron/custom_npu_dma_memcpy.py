
from aie.dialects.transform.structured import MixedValues, _dispatch_mixed_values
from aie.dialects._aiex_ops_gen import NpuDmaMemcpyNdOp, npu_dma_memcpy_nd
from aie._mlir_libs._mlir.ir import DenseF64ArrayAttr

from aie.dialects._ods_common import get_default_loc_context as _ods_get_default_loc_context
from aie.dialects._ods_common  import _cext as _ods_cext
_ods_ir = _ods_cext.ir


from aie._mlir_libs._mlir.ir import Attribute

# from aie.dialects._aiex_ops_gen import *
# from aie.dialects._aiex_ops_gen import ObjectFifoCreateOp, dma_bd, EndOp

# @_ods_cext.register_dialect
# class _Dialect(_ods_ir.Dialect):
#   DIALECT_NAMESPACE = "aiex"
from aie.dialects._aiex_ops_gen import _Dialect
# @_ods_cext.register_operation(_Dialect)

def generate_packet_attribute(packet_id:int, packet_type:int):
    return Attribute.parse(f"#aie.packet_info<pkt_type = {packet_type}, pkt_id = {packet_id}>")
class NpuDmaMemcpyNd(NpuDmaMemcpyNdOp):


    def __init__(
        self,
        metadata: str ,
        bd_id,
        mem,
        tap:   None = None,
        offsets: MixedValues | None = None,
        sizes: MixedValues | None = None,
        strides: MixedValues | None = None,
        issue_token: bool | None = None,
        burst_length: int = 0,
        packet_id = None,
        packet_type = None 
    ):
        if tap and not (offsets is None and sizes is None and strides is None):
            raise ValueError(
                "NpuDmaMemcpyNd can take either a TileAccessPattern OR (sizes and/or strides and/or offsets), but not both."
            )
        if tap:
            sizes = tap.sizes.copy()
            strides = tap.strides.copy()
            # For some reason, the type checking of offsets does not mesh well with offset being a property
            # so here we make sure it is evaluated and properly is seen as an integer.
            offsets = [0] * 3 + [int(tap.offset)]
        else:
            if offsets is None:
                offsets = [0] * 4
            if sizes is None:
                sizes = [0] * 4
            if strides is None:
                strides = [0] * 3 + [1]
        dynamic_offsets, _packed_offsets, static_offsets = _dispatch_mixed_values(
            offsets
        )
        dynamic_sizes, _packed_sizes, static_sizes = _dispatch_mixed_values(sizes)
        dynamic_strides, _packed_strides, static_strides = _dispatch_mixed_values(
            strides
        )
        # if isinstance(metadata, ObjectFifoCreateOp):
        #     metadata = metadata.sym_name.value

        # # packet_test = _ods_ir.AttrBuilder.get('PacketInfoAttr')(packet, context=_ods_get_default_loc_context())
        if packet_id is not None and packet_type is not None:
            packet =generate_packet_attribute(packet_id=packet_id, packet_type=packet_type)
        else:
            packet = None
        super().__init__(
            mem,
            dynamic_offsets,
            dynamic_sizes,
            dynamic_strides,
            static_offsets,
            static_sizes,
            static_strides,
            metadata,
            bd_id,
            issue_token=issue_token,
            burst_length=burst_length,
            packet = packet
            # packet= Attribute.parse("{pkt_id = 3 : i16, pkt_type = 0 : i16}")
        )
        # # self.packet = Attribute.parse("{pkt_id = 3 : i16, pkt_type = 0 : i16}")
        # y = self.static_strides
        # y2 = self.packet
        # k =200
        # obj = npu_dma_memcpy_nd(memref=mem, offsets=dynamic_offsets, sizes=dynamic_sizes, strides=dynamic_strides, static_offsets=static_offsets,
        #                     static_sizes=static_sizes, static_strides=static_strides, 
        #                     metadata=metadata, id=bd_id, issue_token=issue_token,
        #                     burst_length=burst_length, packet=Attribute.parse("#aie.packet_info<pkt_type = 0, pkt_id = 2>")
        #                   )
        # y2 = obj.packet
        # print(y2)
