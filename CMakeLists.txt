#-------------------------------------------------------------------------------
# gbperftest/CMakeLists.txt:  cmake script for gbperftest
#-------------------------------------------------------------------------------

# gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.23 )

project ( gbperftest LANGUAGES C CXX )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules )

include ( SuiteSparsePolicy )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

#-------------------------------------------------------------------------------
# Find ../GraphBLAS, ../LAGraph, and OpenMP
#-------------------------------------------------------------------------------

include_directories ( ${PROJECT_SOURCE_DIR}/include
                      ${PROJECT_SOURCE_DIR}/../LAGraph/src/test/include
                      ${PROJECT_SOURCE_DIR}/../LAGraph/src/benchmark
                      ${PROJECT_SOURCE_DIR}/../LAGraph/src/utility )

find_package ( OpenMP COMPONENTS C )
find_package ( Threads )

add_library ( LAGraph SHARED IMPORTED )
set_target_properties ( LAGraph PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../LAGraph/build/src/liblagraph.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../LAGraph/include" )

add_library ( LAGraphX SHARED IMPORTED )
set_target_properties ( LAGraphX PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../LAGraph/build/experimental/liblagraphx.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../LAGraph/include" )

add_library ( GraphBLAS SHARED IMPORTED )
set_target_properties ( GraphBLAS PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../GraphBLAS/build/libgraphblas.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../GraphBLAS/Include" )

#-------------------------------------------------------------------------------
# set compiler flags
#-------------------------------------------------------------------------------

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
else ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}" )
endif ( )

message ( STATUS "CMAKE C flags: " ${CMAKE_C_FLAGS} )

#-------------------------------------------------------------------------------
# add source files for the gbperftest utility library
#-------------------------------------------------------------------------------

file ( GLOB PERFTEST_LIB_SOURCES "src/*.c" )
add_library ( gbperftest SHARED ${PERFTEST_LIB_SOURCES} )

target_link_libraries ( gbperftest PUBLIC OpenMP::OpenMP_C )
target_link_libraries ( gbperftest PUBLIC GraphBLAS )
target_link_libraries ( gbperftest PUBLIC LAGraph )
target_link_libraries ( gbperftest PUBLIC LAGraphX )
target_link_libraries ( gbperftest PUBLIC m )

#-------------------------------------------------------------------------------
# optional: CUPTI for per-kernel CPU<->GPU unified-memory traffic stats
#-------------------------------------------------------------------------------

# If the CUDA toolkit (with CUPTI) is found, gbperf_um.c is compiled with real
# CUPTI unified-memory counters; otherwise it falls back to no-op stubs so the
# benchmarks still build on hosts without CUDA (e.g. macOS).

find_package ( CUDAToolkit QUIET )
if ( CUDAToolkit_FOUND AND TARGET CUDA::cupti )
    message ( STATUS "CUPTI found: unified-memory stats ENABLED" )
    target_compile_definitions ( gbperftest PUBLIC GBPERF_HAS_CUPTI )
    target_link_libraries ( gbperftest PUBLIC CUDA::cupti CUDA::cudart
        Threads::Threads )
else ( )
    message ( STATUS "CUPTI not found: unified-memory stats disabled" )
endif ( )

#-------------------------------------------------------------------------------
# add testing executables
#-------------------------------------------------------------------------------

file ( GLOB TEST_SOURCES LIST_DIRECTORIES false *.c )

foreach ( test_source_file ${TEST_SOURCES} )
    get_filename_component ( test_filename ${test_source_file} NAME )
    string ( REPLACE ".c" "" test ${test_filename} )
    message ( STATUS "Test program: ${test}" )
    add_executable ( ${test} ${test_source_file} )
    target_link_libraries ( ${test} PUBLIC gbperftest )
endforeach ( test_source_file ${TEST_SOURCES} )

