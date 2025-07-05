# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Try to find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # If pybind11 is not found, try to use pip install
    message(STATUS "pybind11 not found, trying to download from GitHub")
    include(FetchContent)
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Create the Python module
pybind11_add_module(freshdiskann python_wrapper.cpp)

# Set properties for the Python module
set_target_properties(freshdiskann PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Link against diskann library (assuming you have a diskann target from src)
target_link_libraries(freshdiskann PRIVATE diskann)

# Include directories
target_include_directories(freshdiskann PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/tsl/include
        ${PROJECT_SOURCE_DIR}/include/dll
        ${PROJECT_SOURCE_DIR}/include/ols
)

# Compiler-specific settings
if(MSVC)
    target_compile_definitions(freshdiskann PRIVATE
            USE_AVX2
            USE_ACCELERATED_PQ
            _WINDOWS
            NOMINMAX
            UNICODE
            MKL_ILP64
    )
    target_compile_options(freshdiskann PRIVATE
            /arch:AVX2
            /openmp:experimental
            /std:c++14
            /W3
            /EHsc
            /permissive-
    )
else()
    target_compile_definitions(freshdiskann PRIVATE
            USE_AVX2
            USE_ACCELERATED_PQ
            MKL_ILP64
    )
    target_compile_options(freshdiskann PRIVATE
            -march=native
            -fopenmp
            -fopenmp-simd
            -Wall
            -Wextra
            -fno-builtin-malloc
            -fno-builtin-calloc
            -fno-builtin-realloc
            -fno-builtin-free
    )
endif()

# Link MKL and other libraries
if(MSVC)
    target_link_libraries(dfreshdiskann PRIVATE
            "${INTEL_ROOT}/mkl/lib/intel64/mkl_core_dll.lib"
            "${INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib"
            "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_thread_dll.lib"
            "${INTEL_ROOT}/compiler/lib/intel64/libiomp5md.lib"
            "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_ilp64_dll.lib"
            "${INTEL_ROOT}/mkl/lib/intel64/mkl_sequential_dll.lib"
    )
else()
    target_link_libraries(freshdiskann PRIVATE
            mkl_intel_ilp64
            mkl_intel_thread
            mkl_core
            iomp5
            pthread
            m
            dl
            aio
    )
    target_link_directories(freshdiskann PRIVATE
            ${INTEL_ROOT}/lib/intel64
            ${MKL_ROOT}/lib/intel64
    )
endif()

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(freshdiskann PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set the output name (remove the 'lib' prefix and use .pyd on Windows)
set_target_properties(freshdiskann PROPERTIES
        OUTPUT_NAME "freshdiskann"
        PREFIX ""
)

if(WIN32)
    set_target_properties(freshdiskann PROPERTIES SUFFIX ".pyd")
endif()


install(TARGETS freshdiskann
        LIBRARY DESTINATION ${Python3_SITEARCH})