# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_execute_calc_ryy)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
set(ASCEND_DEVLIB_PATH "${INC_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib")
set(ASCEND_DRIVER_LIB_PATH "/usr/local/Ascend/driver/lib64/driver")

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ../inc
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${ASCEND_DEVLIB_PATH}
    ${ASCEND_DRIVER_LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_calc_ryy_op
    operator_desc.cpp
    op_runner.cpp
    main.cpp
    common.cpp
)

target_link_libraries(execute_calc_ryy_op
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    ascend_hal
    stdc++
)

install(TARGETS execute_calc_ryy_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
