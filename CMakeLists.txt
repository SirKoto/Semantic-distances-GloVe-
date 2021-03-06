cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(TGA_Project LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 11)


# Uncomment the following line to avoid using pinned memory
# add_compile_definitions(NOT_PINNED_MEMORY)

set(SRC_DIR "GPU_CUDA_Distances")

set(CU_SRC_FILES ${SRC_DIR}/kernel.cu ${SRC_DIR}/CudaHelp.cu)

set(CPP_SRC_FILES 
        ${SRC_DIR}/GlobalHeader.h
        ${SRC_DIR}/loader.h     ${SRC_DIR}/loader.cpp)

set(CPP_MAIN ${SRC_DIR}/main.cpp)


FIND_PACKAGE(CUDA REQUIRED)
ADD_DEFINITIONS(-DUSE_CUDA)

SET(SRC_FILES ${CPP_SRC_FILES} ${CU_SRC_FILES})

#to correctly include the above cuda libraries, a splitting must take place
foreach(libname ${CUDA_LIBS})
LIST(APPEND CUDA_LIBS_SEP ${${libname}})
endforeach(${CUDA_LIBS_SEP})

#compiler flags for cuda
SET(CUDA_NVCC_FLAGS
    -Xcompiler #Explicitly specify the language for the input files, rather than letting the compiler choose a default based on the file name suffix. Allowed values for this option: c, c++, cu. 
    -fPIC
    -use_fast_math
    -gencode=arch=compute_75,code="sm_75,compute_75"
    --ptxas-options=-v
    #--device-debug #possibility to debug device code; uncomment for speedup
    #--debug #possibility to debug host code; uncomment for speedup
    -O3 #specify optimization level for host code
    -Wno-deprecated-gpu-targets #to suppress warning for deprecated architectures
    #-D_FORCE_INLINES #uncommet if cuda<8.0 and some linker error of memcpy in string.h is thrown during build
)
    
CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${CPP_MAIN} ${SRC_FILES} )
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                        ${CUDA_LIBRARIES} #Cudart library.
                        ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                        )
