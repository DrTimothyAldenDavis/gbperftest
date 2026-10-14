
//------------------------------------------------------------------------------
// GB_sort_seed_test: ordered input must not cause quadratic sorting
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

#include "GraphBLAS.h"
#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#define OK(method)                                                           \
{                                                                           \
    GrB_Info info = (method) ;                                               \
    if (info != GrB_SUCCESS)                                                 \
    {                                                                       \
        fprintf (stderr, "%s failed: %d\n", #method, (int) info) ;             \
        exit (EXIT_FAILURE) ;                                               \
    }                                                                       \
}

static atomic_uint_fast64_t comparisons ;

static void counted_lt (void *z, const void *x, const void *y)
{
    atomic_fetch_add_explicit (&comparisons, 1, memory_order_relaxed) ;
    *((bool *) z) = *((const int64_t *) x) < *((const int64_t *) y) ;
}

static bool test_sort (GrB_BinaryOp op, int nthreads, int bits)
{
    // The small case uses the first vector's seed.  The large case exceeds
    // GB_SORT_BASECASE and exercises the first parallel quicksort leaf's seed.
    GrB_Index n = (nthreads == 1) ? 8192 : 131072 ;
    GrB_Index size = 2*n + 1 ;
    int64_t *values = malloc (n * sizeof (int64_t)) ;
    GrB_Index *indices = malloc (n * sizeof (GrB_Index)) ;
    if (values == NULL || indices == NULL) exit (EXIT_FAILURE) ;
    for (GrB_Index k = 0 ; k < n ; k++)
    {
        values [k] = (int64_t) k - (int64_t) (n/2) ;
        indices [k] = 2*k + 1 ;
    }

    GxB_Context context = NULL ;
    GrB_Vector A = NULL, C = NULL, P = NULL ;
    // Sorting duplicates A using global integer hints for its working copy.
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, bits, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GxB_Context_new (&context)) ;
    OK (GxB_Context_set_INT32 (context, GxB_CONTEXT_NTHREADS, nthreads)) ;
    OK (GxB_Context_set_FP64 (context, GxB_CONTEXT_CHUNK, 1)) ;
    OK (GxB_Context_engage (context)) ;
    OK (GrB_Vector_new (&A, GrB_INT64, size)) ;
    OK (GrB_Vector_build_INT64 (A, indices, values, n, GrB_PLUS_INT64)) ;
    OK (GrB_Vector_new (&C, GrB_INT64, size)) ;
    OK (GrB_Vector_new (&P, GrB_UINT64, size)) ;
    atomic_store (&comparisons, 0) ;
    OK (GxB_Vector_sort (C, P, op, A, NULL)) ;
    uint64_t count = atomic_load (&comparisons) ;
    bool passed = count >= n && count < 128*n ;
    printf ("threads=%d bits=%d n=%" PRIu64 " comparisons=%" PRIu64
        " bound=%" PRIu64 " %s\n", nthreads, bits, (uint64_t) n,
        count, (uint64_t) (128*n), passed ? "passed" : "FAILED") ;

    // Check sorted values and the original sparse indices in the permutation.
    GrB_Index nvals = n ;
    OK (GrB_Vector_extractTuples_INT64 (indices, values, &nvals, C)) ;
    if (nvals != n) passed = false ;
    for (GrB_Index k = 0 ; k < nvals ; k++)
    {
        if (indices [k] != k || values [k] != (int64_t) k - (int64_t) (n/2))
            passed = false ;
    }
    nvals = n ;
    OK (GrB_Vector_extractTuples_UINT64 (NULL, indices, &nvals, P)) ;
    if (nvals != n) passed = false ;
    for (GrB_Index k = 0 ; k < nvals ; k++)
    {
        if (indices [k] != 2*k + 1) passed = false ;
    }

    OK (GrB_Vector_free (&P)) ;
    OK (GrB_Vector_free (&C)) ;
    OK (GrB_Vector_free (&A)) ;
    OK (GxB_Context_disengage (context)) ;
    OK (GxB_Context_free (&context)) ;
    free (values) ;
    free (indices) ;
    return passed ;
}

int main (void)
{
    OK (GrB_init (GrB_NONBLOCKING)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_OFF, GxB_JIT_C_CONTROL)) ;
    GrB_BinaryOp op = NULL ;
    OK (GrB_BinaryOp_new (&op, counted_lt, GrB_BOOL, GrB_INT64, GrB_INT64)) ;
    bool passed = true ;
    for (int bits = 32 ; bits <= 64 ; bits += 32)
    {
        passed = test_sort (op, 1, bits) && passed ;
        passed = test_sort (op, 4, bits) && passed ;
    }
    OK (GrB_BinaryOp_free (&op)) ;
    OK (GrB_finalize ( )) ;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE ;
}
