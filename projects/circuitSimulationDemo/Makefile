#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred
MlirDIR ?= $(or $(Mlir_DIR),/home/shouyud/PROJECT/mlir-aie)

TemplateRepoDir ?= $(or $(TEMPLATE_REPO_DIR),/home/shouyud/PROJECT/AMDNpuProjectFolder/Template)
TemplaeHostDir := $(TemplateRepoDir)/host

# create a dynamic link of /common folder, 
# ENABLE_CHESSCC ?= 1

DEVICE ?= npu2
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


# modify child makefile variable
# Append to HOST_O_DIR or modify it
EXTRA_HEADERS = $(filter-out ${TemplaeHostDir}/typedef.hpp, $(wildcard ${TemplaeHostDir}/*.hpp))
EXTRA_SRCS =  $(filter-out ${TemplaeHostDir}/host.cpp, $(wildcard ${TemplaeHostDir}/*.cpp))






include $(TemplateRepoDir)/makefiles/common.mk
# Kernel makefile
include $(TemplateRepoDir)/makefiles/kernel.mk

# Bitstream makefile
include $(TemplateRepoDir)/makefiles/bitstream.mk
include $(TemplateRepoDir)/makefiles/mlir_bitstream.mk

# # Host makefile
# include $(TemplateRepoDir)/makefiles/host.mk
# # overide child 
# HOST_SRCDIR := ${HOME_DIR}/host
# HOST_HEADERS = $(wildcard ${HOST_SRCDIR}/*.hpp)
# HOST_SRCS = $(wildcard ${HOST_SRCDIR}/*.cpp)
# HOST_HEADERS := $(HOST_HEADERS) $(EXTRA_HEADERS)
# HOST_SRCS := $(HOST_SRCS) $(EXTRA_SRCS)



# Override child
HOST_SRCDIR := ${HOME_DIR}/host

# Print HOST_SRCDIR for debugging
$(info HOST_SRCDIR = $(HOST_SRCDIR))

HOST_HEADERS = $(wildcard ${HOST_SRCDIR}/*.hpp)
HOST_SRCS = $(wildcard ${HOST_SRCDIR}/*.cpp)

# Print HOST_HEADERS and HOST_SRCS before adding EXTRA_HEADERS and EXTRA_SRCS
$(info HOST_HEADERS (before EXTRA) = $(HOST_HEADERS))
$(info HOST_SRCS (before EXTRA) = $(HOST_SRCS))

override HOST_HEADERS := $(HOST_HEADERS) $(EXTRA_HEADERS)
override HOST_SRCS := $(HOST_SRCS) $(EXTRA_SRCS)

# Print HOST_HEADERS and HOST_SRCS after adding EXTRA_HEADERS and EXTRA_SRCS
$(info HOST_HEADERS (after EXTRA) = $(HOST_HEADERS))
$(info HOST_SRCS (after EXTRA) = $(HOST_SRCS))
CXXFLAGS += -I$(TemplaeHostDir)
CXXFLAGS += -I$(TemplateRepoDir)/common
include $(TemplateRepoDir)/makefiles/host.mk


CXX_HOST_TEST = g++
CXXFLAGS_HOST_TEST = -std=c++17 -Wall -O2
TARGET_HOST_TEST = test_host
SRC_TEST = test.cpp kernel/circuitSimulation.cc

.PHONY: test_host_run


.PHONY: run all kernel link bitstream host clean instructions route
all: config_stage ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}



${TARGET_HOST_TEST}: ${SRC_TEST}
	$(CXX_HOST_TEST) $(CXXFLAGS_HOST_TEST) -o $@ $^
test_host_run: ${TARGET_HOST_TEST}
	./${TARGET_HOST_TEST}

clean_test:
	-@rm -f ${TARGET_HOST_TEST}


clean:clean_test 
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf route.json
	-@rm -rf generated.txt
	-@rm -rf route*.txt
	-@rm -f $(TARGET_HOST_TEST)
	-@rm -f ./iron/final_config.json
	-@rm -f ./host/circuitConfig.h
	-@rm -f ./kernel/circuitConfig.h
	-@rm -f final_config.json
	-@rm -f circuitConfig.h

config_stage:
	${clean}
	-@rm -f ./iron/final_config.json
	-@rm -f ./host/circuitConfig.h
	-@rm -f ./kernel/circuitConfig.h
	-@rm -f final_config.json
	-@rm -f circuitConfig.h

	python generate_config.py ./LinearCircuitDynamicLinks/cppReadCircuitDataLibrary/config.json  --final_json final_config.json  --header circuitConfig.h
	cp final_config.json  ./iron/
	cp circuitConfig.h ./host/
	cp circuitConfig.h ./kernel/




kernel: ${KERNEL_OBJS} 


instructions: ${INSTS_TARGETS}


link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGETS} 


host: ${HOST_C_TARGET} 


clean_host:
	-@rm -rf build/host

run: ${HOST_C_TARGET} ${XCLBIN_TARGETS} ${INSTS_TARGETS} 
	./${HOST_C_TARGET} | tee out.log
# compare the generated.txt with the insts, if not the same, say error
# @if diff -q generated.txt ${INSTS_TARGETS}; then \
# 	echo "PASS"; \
# else \
# 	echo "FAIL"; \
# 	exit 1; \
# fi


route: ${IRON_BOTH_MLIR_TARGET}
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > route.json
	python ${TemplateRepoDir}/visualize.py -j route.json



trace: trace.txt
	${MlirDIR}/programming_examples/utils/parse_trace.py --filename trace.txt --mlir build/mlir/mv.mlir --colshift 0 > circutSimulation.json
	${MlirDIR}/programming_examples/utils/get_trace_summary.py --filename circutSimulation.json
