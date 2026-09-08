//------------------------------------------------------------------------------
// gbperf_build.c: benchmark build
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage:  ./gbperf_build m n nvals seed
// to build a random m-by-n matrix with nvals entries
// If not present, m=4, n=6, nvals=10, seed=1 is the default.

// This main program makes use of supporting utilities in
// src/benchmark/LAGraph_demo.h and src/utility/LG_internal.h.

#include "gbperftest.h"

#undef  LG_FREE_ALL
#define LG_FREE_ALL                             \
{                                               \
    GrB_free (&Mod) ;                           \
    GrB_free (&A) ;                             \
    GrB_free (&B) ;                             \
    GrB_free (&C) ;                             \
    GrB_free (&I) ;                             \
    GrB_free (&J) ;                             \
    GrB_free (&X) ;                             \
    GrB_free (&(Results [0])) ;                 \
    GrB_free (&(Results [1])) ;                 \
    GrB_free (&State) ;                         \
    GrB_free (&Zero) ;                          \
}

//------------------------------------------------------------------------------
// mod function for uint64: z = x % y
//------------------------------------------------------------------------------

void build2_demo_mod (void *z, const void *x, const void *y)
{
    uint64_t a = (*((uint64_t *) x)) ;
    uint64_t b = (*((uint64_t *) y)) ;
    (*((uint64_t *) z)) = a % b ;
}

#define MOD_FUNCTION_DEFN                                           \
"void build2_demo_mod (void *z, const void *x, const void *y)   \n" \
"{                                                              \n" \
"    uint64_t a = (*((uint64_t *) x)) ;                         \n" \
"    uint64_t b = (*((uint64_t *) y)) ;                         \n" \
"    (*((uint64_t *) z)) = a % b ;                              \n" \
"}"

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // startup LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;        // for error messages from LAGraph
    GrB_BinaryOp Mod = NULL ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Matrix Results [2] = { NULL, NULL } ;
    GrB_Vector I = NULL, J = NULL, X = NULL, State = NULL ;
    GrB_Scalar Zero = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;       // BURBLE
    demo_init (burble) ;
    int device = 0 ;
    OK (GrB_set (GrB_GLOBAL, GxB_NARENAS + device, GxB_ARENA_DATA)) ;
    OK (GrB_set (GrB_GLOBAL, GxB_NARENAS + device, GxB_ARENA_HEADER)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Index nrows = 4, ncols = 6, nvals = 10, seed = 1 ;
    if (argc > 1) { sscanf (argv [1], "%lu", &nrows) ; }
    if (argc > 2) { sscanf (argv [2], "%lu", &ncols) ; }
    if (argc > 3) { sscanf (argv [3], "%lu", &nvals) ; }
    if (argc > 4) { sscanf (argv [4], "%lu", &seed) ; }
    printf ("Generating a random %lu-by-%lu matrix with %lu entries"
        " (seed: %lu)\n", nrows, ncols, nvals, seed) ;

    int32_t nthreads_max, ngpus_max ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads_max, GxB_NTHREADS)) ;
    printf ("# OpenMP threads: %d\n", nthreads_max) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &ngpus_max, GxB_NGPUS)) ;
    printf ("# GPUs:           %d\n", ngpus_max) ;
    if (ngpus_max > 1) ngpus_max = 1 ;

    OK (GxB_BinaryOp_new (&Mod, build2_demo_mod,
        GrB_UINT64, GrB_UINT64, GrB_UINT64,
        "build2_demo_mod", MOD_FUNCTION_DEFN)) ;

    //--------------------------------------------------------------------------
    // construct the random tuples
    //--------------------------------------------------------------------------

    double t = LAGraph_WallClockTime ( ) ;
    OK (GrB_Vector_new (&State, GrB_UINT64, nvals)) ;
    OK (GrB_assign (State, NULL, NULL, 0, GrB_ALL, nvals, NULL)) ;
    OK (LAGraph_Random_Seed (State, seed, msg)) ;

    // I = mod (State, nrows) ;
    OK (GrB_Vector_new (&I, GrB_UINT64, nvals)) ;
    OK (GrB_apply (I, NULL, NULL, Mod, State, nrows, NULL)) ;

    // State = next (State)
    OK (LAGraph_Random_Next (State, msg)) ;

    // J = mod (State, ncols) ;
    OK (GrB_Vector_new (&J, GrB_UINT64, nvals)) ;
    OK (GrB_apply (J, NULL, NULL, Mod, State, ncols, NULL)) ;

    // State = next (State)
    OK (LAGraph_Random_Next (State, msg)) ;

    // X = (double) State
    OK (GrB_Vector_new (&X, GrB_FP64, nvals)) ;
    OK (GrB_assign (X, NULL, NULL, State, GrB_ALL, nvals, NULL)) ;
    GrB_free (&State) ;

    // X = X / (double) UINT64_MAX
    OK (GrB_apply (X, NULL, NULL, GrB_DIV_FP64, X, (double) UINT64_MAX,
        NULL)) ;

    t = LAGraph_WallClockTime ( ) - t ;
    printf ("Time to construct random tuples: %g sec\n", t) ;

    //--------------------------------------------------------------------------
    // build with duplicates/unsorted, and then with no duplicates/sorted
    //--------------------------------------------------------------------------

    for (int pass = 0 ; pass <= 1 ; pass++)
    {

        printf ("#### PASS: %d : %s\n", pass,
            (pass == 0) ? "with duplicates/need sorting" :
            "no duplicates and no sorting needed\n") ;

        if (pass == 1)
        {
            // sort the tuples by constructing the matrix on the CPU
            OK (GrB_Global_set_INT32 (GrB_GLOBAL, 0, GxB_NGPUS)) ;
            OK (GrB_Matrix_new (&A, GrB_FP64, nrows, ncols)) ;
            OK (GxB_Matrix_build_Vector (A, I, J, X, GrB_PLUS_FP64,
                NULL)) ;
            OK (GxB_Matrix_extractTuples_Vector (I, J, X, A, NULL)) ;
        }

        //----------------------------------------------------------------------
        // build the matrix
        //----------------------------------------------------------------------

        double tbest [8], ttran [8], tadd ;
        tadd = INFINITY ;
        for (int32_t ngpus = 0 ; ngpus <= ngpus_max ; ngpus++)
        {
            printf ("\n======================== Benchmark with %d GPUs:\n",
                ngpus) ;
            OK (GrB_Global_set_INT32 (GrB_GLOBAL, ngpus, GxB_NGPUS)) ;
            int32_t ngpus_used = 0 ;
            OK (GrB_Global_get_INT32 (GrB_GLOBAL, &ngpus_used, GxB_NGPUS));
            tbest [ngpus] = INFINITY ;
            ttran [ngpus] = INFINITY ;

            for (int32_t k = 0 ; k < 3 ; k++)
            {

                //--------------------------------------------------------------
                // build
                //--------------------------------------------------------------

                printf ("\n\nBUILD (%d)================================:\n",k) ;
                for (int kk = 0 ; kk <= 1 ; kk++)
                {
                    t = LAGraph_WallClockTime ( ) ;
                    GrB_Matrix_free (&A) ;
                    OK (GrB_Matrix_new (&A, GrB_FP64, nrows, ncols)) ;
                    OK (GrB_Matrix_set_INT32 (A, GxB_HYPERSPARSE,
                        GxB_SPARSITY_CONTROL)) ;
                    OK (GxB_Matrix_build_Vector (A, I, J, X, GrB_PLUS_FP64,
                        NULL)) ;
                    t = LAGraph_WallClockTime ( ) - t ;
                    printf ("#gpus: %d, Time for build (%d):         %g sec (kk: %d)\n",
                        ngpus_used, k, t, kk) ;
                }

                tbest [ngpus] = fmin (tbest [ngpus], t) ;

                //--------------------------------------------------------------
                // print the matrix and reduce to scalar
                //--------------------------------------------------------------

                // OK (GxB_print (A, 1)) ;

                if (nvals <= (10 * 1000 * 1000) && k == 0)
                {
                    printf ("DUP to check\n") ;
                    OK (GrB_Matrix_dup (&(Results [ngpus]), A)) ;
                }

                t = LAGraph_WallClockTime ( ) ;
                double sum = 0 ;
                OK (GrB_Matrix_reduce_FP64 (&sum, NULL,
                    GrB_PLUS_MONOID_FP64, A, NULL)) ;
                t = LAGraph_WallClockTime ( ) - t ;
                // printf ("sum %g\n", sum) ;
                printf ("#gpus: %d, Time for reduce (%d)         %g sec\n",
                    ngpus_used, k, t) ;

                //--------------------------------------------------------------
                // test the transpose
                //--------------------------------------------------------------

                OK (GrB_Matrix_new (&B, GrB_FP64, ncols, nrows)) ;
                OK (GrB_Matrix_new (&C, GrB_FP64, nrows, ncols)) ;

                printf ("\n\nTRANSPOSES (%d) ==========================:\n",k) ;
                t = LAGraph_WallClockTime ( ) ;
                OK (GrB_transpose (B, NULL, NULL, A, NULL)) ;
                t = LAGraph_WallClockTime ( ) - t ;
                // printf ("first transpose time: %g\n", t) ;
                OK (GrB_Matrix_clear (B)) ;

                double t1 = LAGraph_WallClockTime ( ) ;
                OK (GrB_transpose (B, NULL, NULL, A, NULL)) ;
                t1 = LAGraph_WallClockTime ( ) - t1 ;
                // printf ("first transpose time: %g\n", t) ;

                if (nvals <= (10 * 1000 * 1000))
                {
                    double t2 = LAGraph_WallClockTime ( ) ;
                    OK (GrB_transpose (C, NULL, NULL, B, NULL)) ;
                    t2 = LAGraph_WallClockTime ( ) - t2 ;
                    bool ok = false ;
                    OK (LAGraph_Matrix_IsEqual (&ok, A, C, msg)) ;
                    printf ("transpose OK: %d\n", ok) ;
                    fflush (stdout) ;
                    if (!ok) abort ( ) ;
                }

                printf ("---- transpose times: %g and %g\n", t, t1) ;
                ttran [ngpus] = fmin (ttran [ngpus], fmin (t, t1)) ;

                //--------------------------------------------------------------
                // test C=A+B
                //--------------------------------------------------------------

                if (ngpus == 0 && nrows == ncols)
                {
                    printf ("\n\nADD (%d) ==========================:\n",k) ;
                    // GxB_print (A, 2) ;
                    // GxB_print (B, 2) ;

                    t1 = LAGraph_WallClockTime ( ) ;
                    OK (GrB_eWiseAdd (C, NULL, NULL, GrB_PLUS_FP64,
                        A, B, NULL)) ;
                    t1 = LAGraph_WallClockTime ( ) - t1 ;
                    tadd = fmin (tadd, t1) ;

                    // GxB_print (C, 2) ;
                    printf ("add time: %g\n", t) ;
                }

                GrB_Matrix_free (&A) ;
                GrB_Matrix_free (&B) ;
                GrB_Matrix_free (&C) ;
            }
        }

        printf ("\n-------------------------------------------------------\n") ;
        printf ("PASS %d, Best build times: CPU %g, GPU %g, speedup %g\n",
            pass, tbest [0], tbest [1], tbest [0] / tbest [1]) ;
        printf ("PASS %d, Best trans times: CPU %g, GPU %g, speedup %g\n",
            pass, ttran [0], ttran [1], ttran [0] / ttran [1]) ;
        if (nrows == ncols) printf ("add time on CPU: %g\n", tadd) ;
        printf ("---------------------------------------------------------\n") ;

        //----------------------------------------------------------------------
        // check results (unless the matrices are too big)
        //----------------------------------------------------------------------

        if (Results [0] != NULL && Results [1] != NULL)
        {
            bool ok = false ;
            OK (LAGraph_Matrix_IsEqual (&ok, Results [0], Results [1],
                msg)) ;
            printf ("CPU == GPU: %d\n", ok) ;
            if (!ok)
            {
                // A = Results [0] - Results [1]
                OK (GrB_Matrix_new (&A, GrB_FP64, nrows, ncols)) ;
                OK (GrB_Scalar_new (&Zero, GrB_FP64)) ;
                OK (GrB_Scalar_setElement_FP64 (Zero, (double) 0)) ;
                OK (GxB_Matrix_eWiseUnion (A, NULL, NULL, GrB_MINUS_FP64,
                    Results [0], Zero, Results [1], Zero, NULL)) ;
                // drop explicit zeros from A
                OK (GrB_Matrix_select_Scalar (A, NULL, NULL,
                    GrB_VALUENE_FP64, A, Zero, NULL)) ;
                printf ("mismatch: CPU results - GPU results:\n") ;
                OK (GxB_print (A, 3)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free everyting and finish
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    OK (LAGraph_Finalize (msg)) ;
}

