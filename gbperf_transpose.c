//------------------------------------------------------------------------------
// gbperftest/gbperf_transpose: basic performance tests (scale and transpose)
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:
//     ./build/simple_perf matrixfile sourcenodes
//     ./build/simple_perf m n nvals

#include "gbperftest.h"

int main (int argc, char **argv)
{

    LAGraph_Graph G = NULL ;
    GrB_Matrix T = NULL, R = NULL, C = NULL ;
    GrB_Vector V = NULL ;

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS and load the test problem
    //--------------------------------------------------------------------------

    #ifdef GRAPHBLAS_HAS_CUDA
    int ngpus = 1 ;
    #else
    int ngpus = 0 ;
    #endif

    char msg [LAGRAPH_MSG_LEN] ;
    printf ("%s: argc %d\n", __FILE__, argc) ;

    // enable memory tracking
//  GB_Global_malloc_tracking_set (true) ;

    OK (demo_init (0)) ;

    if (argc == 4)
    {
        int64_t m = 0, n = 0, nvals = 0 ;
        sscanf (argv [1], "%" PRId64, &m) ;
        sscanf (argv [2], "%" PRId64, &n) ;
        sscanf (argv [3], "%" PRId64, &nvals) ;
        printf ("Random problem: m %ld, n %ld, nvals %ld\n", m, n, nvals) ;
        double density = ((double) nvals) / (((double) m) * ((double) n)) ;
        OK (LAGraph_Random_Matrix (&T, GrB_FP64, m, n, density,
            (uint64_t) 1, msg)) ;
        OK (LAGraph_New (&G, &T, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    }
    else
    {
        OK (readproblem (&G,
            /* srcs: */ NULL,
            /* make_symmetric: */ false,
            /* remove_self_edges: */ false,
            /* structure: */ false,
            /* preferred type: */ NULL,
            /* ensure_positive: */ false,
            argc, argv)) ;
    }

    OK (LAGraph_Graph_Print (G, 2, stdout, msg)) ;

    int64_t m, n ;
    OK (GrB_Matrix_nrows (&m, G->A)) ;
    OK (GrB_Matrix_ncols (&n, G->A)) ;

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, G->A)) ;

    OK (GrB_free (&(G->AT))) ;

    //--------------------------------------------------------------------------
    // test row/col scale
    //--------------------------------------------------------------------------

    uint64_t I [1] = {0} ;
    OK (GrB_Vector_new (&V, type, n)) ;
    OK (GrB_assign (V, NULL, NULL, 1, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (V, NULL, NULL, 2, I, 1, NULL)) ;    // make V non-iso
    OK (GrB_Matrix_diag (&C, V, 0)) ;
    OK (GrB_free (&V)) ;

    OK (GrB_Vector_new (&V, type, m)) ;
    OK (GrB_assign (V, NULL, NULL, 1, GrB_ALL, m, NULL)) ;
    OK (GrB_assign (V, NULL, NULL, 2, I, 1, NULL)) ;    // make V non-iso
    OK (GrB_Matrix_diag (&R, V, 0)) ;
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
                G->A, C, NULL)) ;
            t = LAGraph_WallClockTime ( ) - t ;
            printf ("GPU: %d, T=A*C, trial %d: %g sec\n", gpu, k, t) ;
            OK (GrB_free (&T)) ;
        }

        // colscale
        for (int k = 0 ; k < 3 ; k++)
        {
            double t = LAGraph_WallClockTime ( ) ;
            OK (GrB_Matrix_new (&T, type, n, n)) ;
            OK (GrB_mxm (T, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                R, G->A, NULL)) ;
            t = LAGraph_WallClockTime ( ) - t ;
            printf ("GPU: %d, T=R*A, trial %d: %g sec\n", gpu, k, t) ;
            OK (GrB_free (&T)) ;
        }
    }

    OK (GrB_set (GrB_GLOBAL, (int32_t) 0, GxB_BURBLE)) ;

    OK (GrB_free (&R)) ;
    OK (GrB_free (&C)) ;

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

