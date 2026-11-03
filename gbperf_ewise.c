//------------------------------------------------------------------------------
// gbperftest/gbperf_ewise: basic performance tests (eWiseAdd, eWiseMult)
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:
//     ./build/gbperf_ewise m n nvals

#include "gbperftest.h"

//------------------------------------------------------------------------------
// TEST_METHOD: test a given method
//------------------------------------------------------------------------------

#define TEST_METHOD(description)                                            \
{                                                                           \
    for (int gpu = 0 ; gpu <= ngpus ; gpu++)                                \
    {                                                                       \
        GB_Global_hack_set (2, gpu ? 1:2) ;                                 \
        printf ("\n------ testing with gpu: %s\n", gpu ? "yes" : "no ") ;   \
        double best_ac = -1 ;                                               \
        for (int k = 0 ; k < 3 ; k++)                                       \
        {                                                                   \
            int collect = (k > 0) ;     /* skip the first (warm-up) run */  \
            if (collect) gbperf_um_reset (um) ;                             \
            double t = LAGraph_WallClockTime ( ) ;                          \
            METHOD ;                                                        \
            t = LAGraph_WallClockTime ( ) - t ;                             \
            uint64_t cnvals = 0 ;                                           \
            OK (GrB_Matrix_nvals (&cnvals, C)) ;                            \
            printf ("GPU: %s, %s, cnvals: %lu, trial %d: %g sec, %s\n",     \
                gpu ? "yes" : "no ", description, cnvals, k, t,             \
                collect ? "" : "  (warm-up)") ;                             \
            if (collect)                                                    \
            {                                                               \
                gbperf_um_sample (um) ;                                     \
                if (best_ac < 0 || t < best_ac)                             \
                {                                                           \
                    best_ac = t ;                                           \
                    gbperf_um_keep (um) ;                                   \
                }                                                           \
            }                                                               \
            if (check && k == 0)                                            \
            {                                                               \
                GrB_free (&R [gpu]) ;                                       \
                OK (GxB_Matrix_dup_arena (&R [gpu], C, arena, arena)) ;     \
            }                                                               \
            /* OK (GxB_print (C, 5)) ; */                                   \
            OK (GrB_free (&C)) ;                                            \
        }                                                                   \
        printf ("GPU: %s, %s, fastest: %g sec\n", gpu ? "yes" : "no ",      \
            description, best_ac) ;                                         \
        tt [gpu] = best_ac ;                                                \
        if (gpu > 0) gbperf_um_report (um, "description") ;                 \
    }                                                                       \
    if (check)                                                              \
    {                                                                       \
        printf ("Checking result:\n") ;                                     \
        bool ok = false ;                                                   \
        OK (LAGraph_Matrix_IsEqual (&ok, R [0], R [1], NULL)) ;             \
        CHK (ok) ;                                                          \
        printf ("%s, tests pass\n", description) ;                          \
    }                                                                       \
    if (ngpus > 0)                                                          \
    {                                                                       \
        printf ("%s, CPU time %g, GPU time: %g, speedup: %g\n",             \
            description,tt [0], tt [1], tt [0] / tt [1]) ;                  \
    }                                                                       \
    OK (GrB_free (&C)) ;                                                    \
    OK (GrB_free (&R [0])) ;                                                \
    OK (GrB_free (&R [1])) ;                                                \
}

//------------------------------------------------------------------------------
// main test program
//------------------------------------------------------------------------------

int main (int argc, char **argv)
{

    GrB_Matrix A = NULL, B = NULL, M = NULL, C = NULL ;
    GrB_Matrix R [2] = { NULL, NULL } ;

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS and create the test problem
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
    int device = 0 ;
    int arena = GxB_NARENAS + device ; 
    OK (GrB_set (GrB_GLOBAL, arena, GxB_ARENA_DATA)) ;
    OK (GrB_set (GrB_GLOBAL, arena, GxB_ARENA_HEADER)) ;
    OK (GrB_set (GrB_GLOBAL, (int32_t) 0, GxB_BURBLE)) ;

    // enable per-kernel CPU<->GPU unified-memory traffic stats (no-op if the
    // build has no CUPTI, or if UM profiling is unavailable on this device)
    gbperf_um_t um = gbperf_um_create (device) ;

    int64_t m = 50, n = 50, nvals = 100 ;
    if (argc >= 2) sscanf (argv [1], "%" PRId64, &m) ;
    if (argc >= 3) sscanf (argv [2], "%" PRId64, &n) ;
    if (argc >= 4) sscanf (argv [3], "%" PRId64, &nvals) ;
    printf ("Random problem: m %ld, n %ld, nvals %ld\n", m, n, nvals) ;
    uint64_t seed = 1 ;
    OK (gbperf_random (&A, m, n, nvals, seed, msg)) ;
    seed += nvals ;
    OK (gbperf_random (&B, m, n, nvals, seed, msg)) ;
    seed += nvals ;
    OK (gbperf_random (&M, m, n, nvals, seed, msg)) ;
    int sparsity = GxB_SPARSE + GxB_HYPERSPARSE ;
    OK (GrB_Matrix_set_INT32 (A, sparsity, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (B, sparsity, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (M, sparsity, GxB_SPARSITY_CONTROL)) ;
    #if 0
    {
        OK (GxB_print (A, 5)) ;
        OK (GxB_print (B, 5)) ;
        OK (GxB_print (M, 5)) ;
    }
    #endif
    uint64_t anvals, bnvals, mnvals ;
    OK (GrB_Matrix_nvals (&anvals, A)) ;
    OK (GrB_Matrix_nvals (&bnvals, B)) ;
    OK (GrB_Matrix_nvals (&mnvals, M)) ;
    printf ("nvals (A): %lu\n", anvals) ;
    printf ("nvals (B): %lu\n", bnvals) ;
    printf ("nvals (M): %lu\n", mnvals) ;

    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;

    bool check = (m <= 100000 && n <= 100000 && nvals <= 100000000) ;
    printf ("check results: %d\n", check) ;
    double tt [8] ;

    //--------------------------------------------------------------------------
    // test eWiseAdd, no mask
    //--------------------------------------------------------------------------

    printf ("\n========================================= eWiseAdd, no mask\n") ;
    #undef  METHOD
    #define METHOD                                                          \
        OK (GrB_Matrix_new (&C, type, m, n)) ;                              \
        OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;     \
        OK (GrB_eWiseAdd (C, NULL, NULL, GrB_PLUS_UINT64, A, B, NULL)) ;
    TEST_METHOD ("C=A+B") ;

    //--------------------------------------------------------------------------
    // test eWiseAdd, with mask
    //--------------------------------------------------------------------------

    printf ("\n======================================= eWiseAdd, with mask\n") ;
    #undef  METHOD
    #define METHOD                                                          \
        OK (GrB_Matrix_new (&C, type, m, n)) ;                              \
        OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;     \
        OK (GrB_eWiseAdd (C, M, NULL, GrB_PLUS_UINT64, A, B, GrB_DESC_S)) ;
    TEST_METHOD ("C<M>=A+B") ;

    //--------------------------------------------------------------------------
    // test eWiseAdd, with complemented mask
    //--------------------------------------------------------------------------

    printf ("\n========================== eWiseAdd, with complemented mask\n") ;
    #undef  METHOD
    #define METHOD                                                          \
        OK (GrB_Matrix_new (&C, type, m, n)) ;                              \
        OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;     \
        OK (GrB_eWiseAdd (C, M, NULL, GrB_PLUS_UINT64, A, B, GrB_DESC_SC)) ;
    TEST_METHOD ("C<!M>=A+B") ;

    //--------------------------------------------------------------------------
    // test eWiseMult, no mask
    //--------------------------------------------------------------------------

    printf ("\n======================================== eWiseMult, no mask\n") ;
    #undef  METHOD
    #define METHOD                                                          \
        OK (GrB_Matrix_new (&C, type, m, n)) ;                              \
        OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;     \
        OK (GrB_eWiseMult (C, NULL, NULL, GrB_PLUS_UINT64, A, B, NULL)) ;
    TEST_METHOD ("C=A.*B") ;

    //--------------------------------------------------------------------------
    // test eWiseMult, mask
    //--------------------------------------------------------------------------

//  OK (GrB_set (GrB_GLOBAL, (int32_t) 1, GxB_BURBLE)) ;
    printf ("\n====================================== eWiseMult, with mask\n") ;
    #undef  METHOD
    #define METHOD                                                          \
        OK (GrB_Matrix_new (&C, type, m, n)) ;                              \
        OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;     \
        OK (GrB_eWiseMult (C, M, NULL, GrB_PLUS_UINT64, A, B, GrB_DESC_S)) ;
    TEST_METHOD ("C=<M>A.*B") ;

    //--------------------------------------------------------------------------
    // test eWiseMult, mask
    //--------------------------------------------------------------------------

    printf ("\n========================= eWiseMult, with complemented mask\n") ;
    #undef  METHOD
    #define METHOD                                                          \
        OK (GrB_Matrix_new (&C, type, m, n)) ;                              \
        OK (GrB_Matrix_set_INT32 (C, sparsity, GxB_SPARSITY_CONTROL)) ;     \
        OK (GrB_eWiseMult (C, M, NULL, GrB_PLUS_UINT64, A, B, GrB_DESC_SC)) ;
    TEST_METHOD ("C=<!M>A.*B") ;

    //--------------------------------------------------------------------------
    // finalize the test
    //--------------------------------------------------------------------------

    OK (GrB_free (&A)) ;
    OK (GrB_free (&B)) ;
    OK (GrB_free (&M)) ;
    gbperf_um_destroy (um) ;
    OK (LAGraph_Finalize (msg)) ;
    return (0) ;
}

