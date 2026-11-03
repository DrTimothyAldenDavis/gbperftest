//------------------------------------------------------------------------------
// gbperf_random: generate a random matrix
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define FREE_WORK                               \
{                                               \
    GrB_free (&Mod) ;                           \
    GrB_free (&Rows) ;                          \
    GrB_free (&Cols) ;                          \
    GrB_free (&Values) ;                        \
    GrB_free (&State) ;                         \
}

#define FREE_ALL                                \
{                                               \
    FREE_WORK ;                                 \
    GrB_free (A) ;                              \
}

#include "gbperftest.h"

//------------------------------------------------------------------------------
// mod function for uint64: z = x % y
//------------------------------------------------------------------------------

void gbperf_mod (void *z, const void *x, const void *y)
{
    uint64_t a = (*((uint64_t *) x)) ;
    uint64_t b = (*((uint64_t *) y)) ;
    (*((uint64_t *) z)) = a % b ;
}

#define MOD_FUNCTION_DEFN                                           \
"void gbperf_mod (void *z, const void *x, const void *y)        \n" \
"{                                                              \n" \
"    uint64_t a = (*((uint64_t *) x)) ;                         \n" \
"    uint64_t b = (*((uint64_t *) y)) ;                         \n" \
"    (*((uint64_t *) z)) = a % b ;                              \n" \
"}"

//------------------------------------------------------------------------------
// gbperf_random
//------------------------------------------------------------------------------

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
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_BinaryOp Mod = NULL ;
    GrB_Vector Rows = NULL, Cols = NULL, Values = NULL, State = NULL ;

    OK (GrB_Matrix_new (A, GrB_UINT64, nrows, ncols)) ;
    if (nrows == 0 || ncols == 0)
    {
        // nothing to do: return A as the requested empty matrix
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // create the Mod operator
    //--------------------------------------------------------------------------

    OK (GxB_BinaryOp_new (&Mod, gbperf_mod,
        GrB_UINT64, GrB_UINT64, GrB_UINT64,
        "gbperf_mod", MOD_FUNCTION_DEFN)) ;

    //--------------------------------------------------------------------------
    // construct the random dense State vector of size nvals
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&State, GrB_UINT64, nvals)) ;
    OK (GrB_assign (State, NULL, NULL, 0, GrB_ALL, nvals, NULL)) ;
    OK (LAGraph_Random_Seed (State, seed, msg)) ;

    //--------------------------------------------------------------------------
    // construct random indices and values
    //--------------------------------------------------------------------------

    // Rows = mod (State, nrows) ;
    OK (GrB_Vector_new (&Rows, GrB_UINT64, nvals)) ;
    OK (GrB_apply (Rows, NULL, NULL, Mod, State, nrows, NULL)) ;

    // State = next (State)
    OK (LAGraph_Random_Next (State, msg)) ;

    // Cols = mod (State, ncols) ;
    OK (GrB_Vector_new (&Cols, GrB_UINT64, nvals)) ;
    OK (GrB_apply (Cols, NULL, NULL, Mod, State, ncols, NULL)) ;

    // State = next (State)
    OK (LAGraph_Random_Next (State, msg)) ;

    //--------------------------------------------------------------------------
    // build the matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_build_Vector (*A, Rows, Cols, State, GxB_IGNORE_DUP, NULL)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

