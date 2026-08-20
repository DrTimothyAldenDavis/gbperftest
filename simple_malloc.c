//------------------------------------------------------------------------------
// gbperftest/simple_malloc: basic tests of malloc/free
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// usage: ./build/simple_perf matrixfile sourcenodes

#include "gbperftest.h"

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS and load the test problem
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;
    printf ("%s: argc %d\n", __FILE__, argc) ;

    #define MBYTE (1024L) * (1024L)
    int64_t nbig = 100 * MBYTE ;
    int nmallocs = 16 ;
    int ntrials = 1000000 ;
    if (argc > 1)
    { 
        nbig = atoi (argv [1]) * MBYTE ;
    }
    if (argc > 2)
    {
        nmallocs = atoi (argv [2]) ;
    }
    if (argc > 3)
    {
        ntrials = atoi (argv [3]) ;
    }
    if (nmallocs > 1024)
    {
        nmallocs = 1024 ;
    }

    void *p [1024] ;

    // enable memory tracking
    // GB_Global_malloc_tracking_set (true) ;

    // turn off the GPU
    GB_Global_hack_set (2, 2) ;
    OK (demo_init (0)) ;
    GB_Global_hack_set (2, 2) ;

    srand (1) ;

    //--------------------------------------------------------------------------
    // test LAGraph_Malloc / LAGraph_Free
    //--------------------------------------------------------------------------

    printf ("nbig: %g MB, nmallocs: %d ntrials: %d\n",
        ((double) nbig) / ((double) MBYTE), nmallocs, ntrials) ;
    double tmalloc = 0, tfree = 0 ;

    for (int trial = 0 ; trial < ntrials ; trial++)
    {
        if (trial % 1000 == 0)
        {
            printf ("\n------------------- trial: %d (malloc %g, free %g)\n",
                trial, tmalloc, tfree) ;
        }
        double t = LAGraph_WallClockTime ( ) ;
        for (int k = 0 ; k < nmallocs ; k++)
        {
            double x = ((double) rand ( )) / ((double) RAND_MAX) ;
            size_t n = (size_t) (((double) nbig) * x) ;
            OK (LAGraph_Malloc (& (p [k]), n, sizeof (uint8_t), msg)) ;
        }
        t = LAGraph_WallClockTime ( ) - t ;
        tmalloc += t ;
        t = LAGraph_WallClockTime ( ) ;
        for (int k = 0 ; k < nmallocs ; k++)
        {
            OK (LAGraph_Free (& (p [k]), msg)) ;
        }
        t = LAGraph_WallClockTime ( ) - t ;
        tfree += t ;
    }
    
    printf ("total tmalloc %g tfree %g\n", tmalloc, tfree) ;

    //--------------------------------------------------------------------------
    // finalize the test
    //--------------------------------------------------------------------------

    printf ("\n------------------- finalize: %d\n", ntrials) ;
    OK (LAGraph_Finalize (msg)) ;
    return (0) ;
}

