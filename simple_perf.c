//------------------------------------------------------------------------------
// gbperftest/simple_perf: basic performance tests
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage: ./build/simple_perf matrixfile sourcenodes

#include "gbperftest.h"

int main (int argc, char **argv)
{

    LAGraph_Graph G = NULL ;
    GrB_Matrix T = NULL, D = NULL ;
    GrB_Vector V = NULL ;

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS and load the test problem
    //--------------------------------------------------------------------------

    #ifdef GRAPHBLAS_HAS_CUDA
    int ngpus = 1 ;
    #else
    int ngpus = 0 ;
    #endif

    // turn off the GPUs
    ngpus = 0 ;

    char msg [LAGRAPH_MSG_LEN] ;
    printf ("%s: argc %d\n", __FILE__, argc) ;
//  gbperftest_nothing ( ) ;

    // enable memory tracking
    GB_Global_malloc_tracking_set (true) ;

    // turn off the GPU
    GB_Global_hack_set (2, 2) ;
    OK (demo_init (0)) ;
    GB_Global_hack_set (2, 2) ;

    OK (readproblem (&G,
        /* srcs: */ NULL,
        /* make_symmetric: */ false,
        /* remove_self_edges: */ false,
        /* structure: */ false,
        /* preferred type: */ NULL,
        /* ensure_positive: */ false,
        argc, argv)) ;

    OK (LAGraph_Graph_Print (G, 2, stdout, msg)) ;

    int64_t n ;
    OK (GrB_Matrix_nrows (&n, G->A)) ;

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, G->A)) ;

    OK (GrB_free (&(G->AT))) ;

    //--------------------------------------------------------------------------
    // test row/col scale
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&V, type, n)) ;
    OK (GrB_assign (V, NULL, NULL, 1, GrB_ALL, n, NULL)) ;
    OK (GrB_Matrix_diag (&D, V, 0)) ;
    OK (GrB_free (&V)) ;

    OK (GrB_set (GrB_GLOBAL, (int32_t) 1, GxB_BURBLE)) ;

    for (int gpu = 0 ; gpu <= ngpus ; gpu++)
    {
        GB_Global_hack_set (2, gpu ? 1:2) ;

        // rowscale
        for (int k = 0 ; k < 3 ; k++)
        {
            double t = LAGraph_WallClockTime ( ) ;
            OK (GrB_Matrix_new (&T, type, n, n)) ;
            OK (GrB_mxm (T, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                G->A, D, NULL)) ;
            t = LAGraph_WallClockTime ( ) - t ;
            printf ("GPU: %d, T=A*D, trial %d: %g sec\n", gpu, k, t) ;
            OK (GrB_free (&T)) ;
        }

        // colscale
        for (int k = 0 ; k < 3 ; k++)
        {
            double t = LAGraph_WallClockTime ( ) ;
            OK (GrB_Matrix_new (&T, type, n, n)) ;
            OK (GrB_mxm (T, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                D, G->A, NULL)) ;
            t = LAGraph_WallClockTime ( ) - t ;
            printf ("GPU: %d, T=D*A, trial %d: %g sec\n", gpu, k, t) ;
            OK (GrB_free (&T)) ;
        }
    }

    OK (GrB_set (GrB_GLOBAL, (int32_t) 0, GxB_BURBLE)) ;

    OK (GrB_free (&D)) ;

    //--------------------------------------------------------------------------
    // test the transpose
    //--------------------------------------------------------------------------

    for (int gpu = 0 ; gpu <= ngpus ; gpu++)
    {
        GB_Global_hack_set (2, gpu ? 1:2) ;
        for (int k = 0 ; k < 3 ; k++)
        {
            double t = LAGraph_WallClockTime ( ) ;
            OK (GrB_Matrix_new (&T, type, n, n)) ;
            OK (GrB_transpose (T, NULL, NULL, G->A, NULL)) ;
            t = LAGraph_WallClockTime ( ) - t ;
            printf ("GPU: %d, Transpose, trial %d: %g sec\n", gpu, k, t) ;
            OK (GrB_free (&T)) ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the test
    //--------------------------------------------------------------------------

    OK (LAGraph_Delete (&G, msg)) ;
    OK (LAGraph_Finalize (msg)) ;
    return (0) ;
}

