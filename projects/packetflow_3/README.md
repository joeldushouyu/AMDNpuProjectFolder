## Point 1
This example is a example of using packet flow to configure single CT to do either add or multiple demo.
## Point 2
Demonstrate of 1KB useage of stack frame out of 64KB memory
In addition, only can allocate 16KB max of memory.

### Point 1 Detail
IF one look at the documentation for the 4 byte [packet header](https://docs.amd.com/r/en-US/am020-versal-aie-ml/AXI4-Stream-Interconnect), one can see that 11-5th bit is being unused.

Also, if apply [aie-opt -aie-create-pathfinder-flows](https://xilinx.github.io/mlir-aie/AIEPasses.html) on packetflow_2, one can see that the aie.switch() only judge incoming data base on stream id, not even stream type.
However, from the example in 'packetflow_2' and this example, one can see that the packetheader being send back matches to the documentation.

For example, when running the program with following. 
0x220004 is the row, column of source sender.
NOTE: the mlir is written as" aie.device(npu1_1col)", so the colum is 1 in this case.
```bash
(ironenv) shouyud@nock8-NucBox-K8:~/AMDNpuProjectFolder/projects/packetflow_3$ make run
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
./host.exe 
APP id is 0
Name: MLIR_AIE
a: 0x80
b: 0x21
c: 0x40
d: 0x02
Header 0(MT->ShimT): 0x80214002
a: 0x00
b: 0x22
c: 0x00
d: 0x04
Header 0(CT->MT): 0x220004

PASS!

```

# Point 2 Detail
In the commented out buffer allocation code in CT_0_2, it demonstrate that although
there is 64kB on each CT, by 1KB is reserved for stack. If not, it will overflow.


