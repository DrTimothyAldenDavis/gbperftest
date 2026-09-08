//------------------------------------------------------------------------------
// gbperftest/include/gbperftest.h: include file for gbperftest utility library
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include <stdio.h>
#include "GraphBLAS.h"
#include "LAGraph.h"
#include "LAGraphX.h"
#include "LAGraph_demo.h"
#include "../GraphBLAS/Source/global/GB_Global.h"

void gbperftest_nothing (void) ;

#define OK(method)                                                          \
{                                                                           \
    GrB_Info this_info = (method) ;                                         \
    if (this_info < GrB_SUCCESS)                                            \
    {                                                                       \
        fprintf (stderr, "fail: line %d file %s\n", __LINE__, __FILE__) ;   \
        abort ( ) ;                                                         \
    }                                                                       \
}

GrB_Info gbperf_random  // random uint64 matrix
(
    // output
    GrB_Matrix *A,      // A is constructed on output
    // input
    GrB_Index nrows,    // # of rows of A
    GrB_Index ncols,    // # of columns of A
    GrB_Index nvals,    // # of entries of A
    uint64_t seed,      // random number seed
    char *msg
) ;

