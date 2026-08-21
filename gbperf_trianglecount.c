//------------------------------------------------------------------------------
// gbperf/gbperf_trianglecount.c: benchmark for LAGr_TriangleCount
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

// Usage:  gbperf_trianglecount < matrixmarketfile.mtx
//         gbperf_trianglecount matrixmarketfile.mtx
//         gbperf_trianglecount matrixmarketfile.grb

//  Known triangle counts:
//      kron:       106873365648
//      urand:      5378
//      twitter:    34824916864
//      web:        84907041475
//      road:       438804

#include "gbperftest.h"

#if defined ( GRAPHBLAS_HAS_CUDA )

#include "LAGraph_demo.h"
void GB_Global_hack_set (int k, int64_t hack) ;

// to run just once, with p = omp_get_max_threads() threads
#define NTHREAD_LIST 1
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LG_FREE_ALL                 \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&A) ;                 \
}

char t [256] ;

char *method_name (int method, int sorting)
{
    char *s ;
    switch (method)
    {
        case LAGr_TriangleCount_AutoMethod: s = "default (Sandia_LUT)           " ; break ;
        case LAGr_TriangleCount_Burkhardt:  s = "Burkhardt: sum ((A^2) .* A) / 6" ; break ;
        case LAGr_TriangleCount_Cohen:      s = "Cohen:     sum ((L*U) .* A) / 2" ; break ;
        case LAGr_TriangleCount_Sandia_LL:  s = "Sandia_LL: sum ((L*L) .* L)    " ; break ;
        case LAGr_TriangleCount_Sandia_UU:  s = "Sandia_UU: sum ((U*U) .* U)    " ; break ;
        case LAGr_TriangleCount_Sandia_LUT: s = "Sandia_LUT: sum ((L*U') .* L)  " ; break ;
        case LAGr_TriangleCount_Sandia_ULT: s = "Sandia_ULT: sum ((U*L') .* U)  " ; break ;
	default:    printf("Unrecognized method: %d\n", method);
		    abort ( ) ;
    }

    if (sorting == LAGr_TriangleCount_Descending) sprintf (t, "%s sort: descending degree", s) ;
    else if (sorting == LAGr_TriangleCount_Ascending) sprintf (t, "%s ascending degree", s) ;
    else if (sorting == LAGr_TriangleCount_AutoSort) sprintf (t, "%s auto-sort", s) ;
    else sprintf (t, "%s sort: none", s) ;
    return (t) ;
}


void print_method (FILE *f, int method, int sorting)
{
    fprintf (f, "%s\n", method_name (method, sorting)) ;
}
#endif

int main (int argc, char **argv)
{
#if defined ( GRAPHBLAS_HAS_CUDA )

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;

    GrB_Matrix A = NULL ;
    LAGraph_Graph G = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = true ;
    demo_init (burble) ;

    int ntrials = 5 ;
    // ntrials = 1 ;        // HACK
    printf ("# of trials: %d\n", ntrials) ;
    printf ("sizeof (clock_t): %d\n", (int) sizeof (clock_t)) ;
    printf ("sizeof (long long int): %d\n", (int) sizeof (long long int)) ;
    printf ("sizeof (long): %d\n", (int) sizeof (long)) ;

    int nt = NTHREAD_LIST ;
    int Nthreads [20] = { 0, THREAD_LIST } ;
    int nthreads_max, nthreads_outer, nthreads_inner ;
    OK (LAGraph_GetNumThreads (&nthreads_outer, &nthreads_inner, msg)) ;
    nthreads_max = nthreads_outer * nthreads_inner ;
    if (Nthreads [1] == 0)
    {
        // create thread list automatically
        Nthreads [1] = nthreads_max ;
        for (int t = 2 ; t <= nt ; t++)
        {
            Nthreads [t] = Nthreads [t-1] / 2 ;
            if (Nthreads [t] == 0) nt = t-1 ;
        }
    }
    printf ("threads to test: ") ;
    for (int t = 1 ; t <= nt ; t++)
    {
        int nthreads = Nthreads [t] ;
        if (nthreads > nthreads_max) continue ;
        printf (" %d", nthreads) ;
    }
    printf ("\n") ;

    //--------------------------------------------------------------------------
    // read in the graph
    //--------------------------------------------------------------------------

    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    OK (readproblem (&G, NULL,
        true, true, true, NULL, false, argc, argv)) ;

#if 0
    OK (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;
#endif

    // determine the cached out degree property
    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    GrB_Index n, nvals ;
    OK (GrB_Matrix_nrows (&n, G->A)) ;
    OK (GrB_Matrix_nvals (&nvals, G->A)) ;

    //--------------------------------------------------------------------------
    // triangle counting
    //--------------------------------------------------------------------------

    LG_SET_BURBLE (true) ;
    GrB_Index ntriangles, ntsimple = 0 ;

#if 0
    // check # of triangles
    double tsimple = LAGraph_WallClockTime ( ) ;
    OK (LG_check_tri (&ntsimple, G, NULL)) ;
    tsimple = LAGraph_WallClockTime ( ) - tsimple ;
    printf ("# of triangles: %" PRId64 " slow time: %g sec\n",
        ntsimple, tsimple) ;
#endif

    // warmup for more accurate timing, and also print # of triangles
    printf ("\nwarmup method: ") ;
//  int presort = LAGr_TriangleCount_AutoSort ; // = 0 (auto selection)
    int presort = LAGr_TriangleCount_NoSort ; // HACK
    print_method (stdout, 6, presort) ;

    // warmup method:
    // LAGr_TriangleCount_Sandia_ULT: sum (sum ((U * L') .* U))
    GB_Global_hack_set (2, 2) ; // never use the GPU
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_ULT ;
#if 1
    for (int trial = 1 ; trial <= 3 ; trial++)
    {
        double ttot = LAGraph_WallClockTime ( ) ;
        OK (LAGr_TriangleCount (&ntriangles, G, &method, &presort, msg)) ;
        printf ("ON CPU (trial %d): # of triangles: %" PRIu64 "\n",
            trial, ntriangles) ;
        print_method (stdout, 6, presort) ;
        ttot = LAGraph_WallClockTime ( ) - ttot ;
        printf ("nthreads: %3d time: %12.6f rate: %6.2f "
            "(Sandia_ULT, one trial)\n",
            nthreads_max, ttot, 1e-6 * nvals / ttot) ;
    }
#endif

    // warmup method WITH GPU:
    // LAGr_TriangleCount_Sandia_ULT: sum (sum ((U * L') .* U))

    GrB_Index ntriangles_gpu ;
    GB_Global_hack_set (2, 1) ; // always use the GPU

    for (int trial = 1 ; trial <= 3 ; trial++)
    {
        //LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_ULT ;
        double ttot = LAGraph_WallClockTime ( ) ;
        OK (LAGr_TriangleCount (&ntriangles_gpu, G, &method, &presort, msg)) ;
        ttot = LAGraph_WallClockTime ( ) - ttot ;
        printf ("ON GPU (trial %d): # of triangles: %" PRIu64 " (GPU)\n",
            trial, ntriangles_gpu) ;
        print_method (stdout, 6, presort) ;
        printf ("nthreads: %3d time: %12.6f rate: %6.2f"
            " (Sandia_ULT, one trial)\n",
            nthreads_max, ttot, 1e-6 * nvals / ttot) ;
    }

    if (ntriangles_gpu != ntriangles)
    {
        printf ("wrong # triangles: %g %g\n", (double) ntriangles,
            (double) ntriangles_gpu) ;
        fflush (stdout) ;
        fflush (stderr) ;
        abort ( ) ;
    }

    LG_FREE_ALL ;
    OK (LAGraph_Finalize (msg)) ;
#endif
    return (GrB_SUCCESS) ;
}
