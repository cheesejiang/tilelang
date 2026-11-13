# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU ARCH, e.g. dav-3510")
set(CMAKE_ASC_ENABLE_SIMT ON)

find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)

set(SCENARIO_NUM "0" CACHE STRING "Case type to run (0-7)")
if(NOT SCENARIO_NUM MATCHES "^[0-7]$")
    message(FATAL_ERROR "SCENARIO_NUM must be an integer from 0 to 7, but got ${SCENARIO_NUM}")
endif()

set(REPEAT_TIMES_CONFIG "100" CACHE STRING "Timed load/add iterations")
if(NOT REPEAT_TIMES_CONFIG MATCHES "^[1-9][0-9]*$" OR REPEAT_TIMES_CONFIG GREATER 4096)
    message(FATAL_ERROR "REPEAT_TIMES_CONFIG must be 1..4096")
endif()

add_executable(demo
    ub_merge_cycles.asc
)

target_compile_definitions(demo PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
    REPEAT_TIMES_CONFIG=${REPEAT_TIMES_CONFIG}
)

target_link_libraries(demo PRIVATE
    m
)
