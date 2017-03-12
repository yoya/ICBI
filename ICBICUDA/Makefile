
BINDIR := ./
# Add source files here
EXECUTABLE	:= icbi

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= interpolation.cpp bmploader.cpp

# CUDA source files (compiled with cudacc)
CUFILES		:= interpolation.cu 

CU_DEPS		:= \
interpolation_fcbi_kernel.cu interpolation_icbi_kernel.cu interpolation_copy_kernel.cu interpolation_ddt_kernel.cu  


ROOTDIR :=  /home/giach/NVIDIA_GPU_Computing_SDK/C/common/
include $(ROOTDIR)/../common/common.mk

INCLUDES  += -I/usr/include/qt4
LIB += -lQtGui 

USEGLLIB          := 1
USEGLUT           := 1
USERENDERCHECKGL  := 1


