#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred


TemplateRepoDir ?= $(or $(TEMPLATE_REPO_DIR),/home/sdu/AMDNpuProjectFolder/Template)
TemplaeHostDir := $(TemplateRepoDir)/host

# create a dynamic link of /common folder, 


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
include $(TemplateRepoDir)/makefiles/host.mk

# This is a copy of the instruction just for debugging purposes
INSTR_REDUDENT_TARGETS := $(patsubst ${BITSTREAM_O_DIR}/from_iron/%.txt, ${HOME_DIR}/build/insts/%.txt.redundant, ${IRON_BOTH_INSTS_TARGET})


.PHONY: prerequsite run all kernel link bitstream host clean instructions
all:  ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}

prerequsite:
	ln -s ${TemplateRepoDir}/common ${HOME_DIR}/common

clean:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf host.exe
	-@rm -rf trace*

test:
	echo "test"
	echo ${AIEOPT_DIR}


kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS} ${INSTR_REDUDENT_TARGETS}


link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean_host:
	-@rm -rf build/host

${HOME_DIR}/build/insts/%.txt.redundant: ${BITSTREAM_O_DIR}/from_iron/%.txt
	cp $< $@

run: ${HOST_C_TARGET} ${XCLBIN_TARGET} ${INSTS_TARGET} ${INSTR_REDUDENT_TARGETS}
	./${HOST_C_TARGET} | tee out.log
