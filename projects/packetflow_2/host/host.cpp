//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#define IN_DATATYPE int8_t
#define OUT_DATATYPE int8_t

int main(int argc, const char *argv[]) {
    std::vector<uint32_t> instr_v;
    int app_id = 1;
    if (argc > 1){
        app_id = atoi(argv[1]);
    }

//    if (app_id == 1){
//      instr_v = test_utils::load_instr_sequence("insts_mul.txt");
//    }
//    else{
//      instr_v = test_utils::load_instr_sequence("insts_add.txt");
//
//    }

    instr_v = test_utils::load_instr_sequence("insts.txt");
    int IN_SIZE;
    int OUT_SIZE;

    if(app_id == 1){
        IN_SIZE = 260;
        OUT_SIZE = 256;
    }
    else{
        IN_SIZE = 260;
        OUT_SIZE = 256;
    }
    OUT_SIZE = 264;
    std::cout << "APP id is " <<  app_id << std::endl;
  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  auto xclbin = xrt::xclbin("final.xclbin");

  std::string Node = "MLIR_AIE";

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(IN_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_SIZE * sizeof(OUT_DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  IN_DATATYPE *bufInA = bo_inA.map<IN_DATATYPE *>();
  std::vector<IN_DATATYPE> srcVecA(IN_SIZE);
  for (int i = 0; i < IN_SIZE; i++){
      if (i < 4){
          if(app_id == 1){
            srcVecA[0] = 7;
            srcVecA[1] = 0xf;
            srcVecA[2] = 0xf0;
            srcVecA[3] = 0xf0;
              // srcVecA[i] = 1;
          }
          else{

              srcVecA[i] = 0;
          }
      }
      else{
          srcVecA[i] = i % 10;
      }
  }

  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(IN_DATATYPE)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  OUT_DATATYPE *bufOut = bo_out.map<OUT_DATATYPE *>();

  int errors = 0;

  // packet is in LSB order
  uint8_t d = static_cast<uint8_t>(*(bufOut));
  uint8_t c = static_cast<uint8_t>(*(bufOut+1));
  uint8_t b = static_cast<uint8_t>(*(bufOut+2));
  uint8_t a = static_cast<uint8_t>(*(bufOut+3));
  unsigned int packet_header_receive_1 = (a << 24) | (b << 16) | (c << 8) | d;
  // Print each variable as a 2-digit hex value
  std::cout << "a: 0x" << std::hex << std::setw(2) << std::setfill('0') << +a << std::endl;
  std::cout << "b: 0x" << std::hex << std::setw(2) << std::setfill('0') << +b << std::endl;
  std::cout << "c: 0x" << std::hex << std::setw(2) << std::setfill('0') << +c << std::endl;
  std::cout << "d: 0x" << std::hex << std::setw(2) << std::setfill('0') << +d << std::endl;

  std::cout << "Header 0(CT->MT): 0x" << std::hex << std::uppercase << packet_header_receive_1 << std::endl;

  // packet is in LSB order
   d = static_cast<uint8_t>(*(bufOut+4));
   c = static_cast<uint8_t>(*(bufOut+5));
   b = static_cast<uint8_t>(*(bufOut+6));
   a = static_cast<uint8_t>(*(bufOut+7));
  unsigned int packet_header_receive_2 = (a << 24) | (b << 16) | (c << 8) | d;
  // Print each variable as a 2-digit hex value
  std::cout << "a: 0x" << std::hex << std::setw(2) << std::setfill('0') << +a << std::endl;
  std::cout << "b: 0x" << std::hex << std::setw(2) << std::setfill('0') << +b << std::endl;
  std::cout << "c: 0x" << std::hex << std::setw(2) << std::setfill('0') << +c << std::endl;
  std::cout << "d: 0x" << std::hex << std::setw(2) << std::setfill('0') << +d << std::endl;

  std::cout << "Header 0(CT->MT): 0x" << std::hex << std::uppercase << packet_header_receive_2 << std::endl;

  int result_offset = 8; // 4 byte of header come out of it
  int input_offset = 4; // 4 byte pass into it
  for (uint32_t i = result_offset; i < OUT_SIZE; i++) {
    uint32_t ref;
    if(app_id == 1){
      ref = srcVecA[i+( input_offset - result_offset)] * 3; // ref for the first input packet
    }
    else{
        ref = srcVecA[i+ ( input_offset - result_offset) ] + 2;
    }
    if (*(bufOut + i ) != ref) { // plus 4 to get ride of the 4 byte packet header that
        std::cout << "Error in output " << std::to_string(bufOut[i])
                  << " != " <<  std::to_string(ref) << " at i " << i << std::endl;
        errors++;
    } else
      ;
        // std::cout << "Correct output " << std::to_string(bufOut[i])
        //           << " == " << ref << std::endl;
    }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }

  std::cout << "\nfailed.\n\n";
  return 1;
}
