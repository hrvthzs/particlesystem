################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= particleSystem
# CUDA source files (compiled with cudacc)
CUFILES		:= \
	src/kernel.cu \
	src/main.cu \
	src/marching_renderer.cu \
	src/particle_system.cu \
	src/grid_uniform.cu \
	src/grid_utils.cu \
	src/sph_simulator.cu \
	src/utils.cu \

# CUDA dependency files
CU_DEPS		:= \
	src/colors.cu \
	src/boundary.cu \
	src/boundary_walls.cu \
	src/buffer_memory.cu \
	src/grid_kernel.cu \
	src/marching_kernel.cu \
	src/sph_density.cu \
	src/sph_force.cu \
	src/sph_kernel.cu \
	src/sph_kernels_poly6.cu \
	src/sph_kernels.cu \
	src/sph_neighbours.cu \
	src/marching.h \


# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	src/buffer.cpp \
	src/buffer_allocator.cpp \
	src/particles_renderer.cpp \
	src/particles_simulator.cpp \
	src/particle_system.cpp \
	src/settings_database.cpp \
	src/shader_program.cpp \



# Additional compiler flags and LIBs to include
USEGLLIB             := 1
USEGLUT	             := 0
USERENDERCHECKGL     := 0
USESDL               := 1
dbg                  := 0

OMIT_CUTIL_LIB       := 1

################################################################################
# Rules and targets

SRCDIR := src/
ROOTDIR := ..

include ../../common/common.mk

run:
	$(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/$(EXECUTABLE)

valgrind:
	valgrind $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/$(EXECUTABLE)
