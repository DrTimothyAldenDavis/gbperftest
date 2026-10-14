
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <GraphBLAS.h>

#define OK(method)                                                     \
    do {                                                               \
        GrB_Info info = (method);                                      \
        if (info != GrB_SUCCESS) {                                     \
            printf("Error at line %d: code %d\n", __LINE__, info);     \
            return info;                                               \
        }                                                              \
    } while (0)

int main(void)
{
    OK(GrB_init(GrB_BLOCKING));

    // V = 2^16 + 1 = 65537
    GrB_Index V = 65537;
    GrB_Matrix A = NULL;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        printf ("\n======================================== TRIAL : %d\n", trial) ;

        OK(GrB_Matrix_new(&A, GrB_BOOL, V, V));

        OK(GrB_Matrix_setElement_BOOL(A, true, 65536, 65536));
        OK (GxB_print (A, 5)) ;

        // Reshape V x V into 1 x (V * V)
        // V * V = 4,295,098,369 > 2^32
        GrB_Info info ;
        if (trial == 0)
        {
            printf ("\n=== reshape to 1-by-(V*V):\n") ;
            info = GxB_Matrix_reshape(A, true, 1, V * V, NULL);
        }
        else
        {
            printf ("\n=== reshape to (V*V)-by-1:\n") ;
            info = GxB_Matrix_reshape(A, true, V * V, 1, NULL);
        }

        printf("GxB_Matrix_reshape return code: %d (%s)\n",
               info, (info == GrB_SUCCESS) ? "GrB_SUCCESS" : "ERROR");

        printf ("after reshape\n") ;
        OK (GxB_print (A, 5)) ;

        size_t nvals = 0;
        OK(GrB_Matrix_nvals(&nvals, A));
        if (nvals > 0)
        {
            GrB_Index *rows = malloc(nvals * sizeof(GrB_Index));
            GrB_Index *cols = malloc(nvals * sizeof(GrB_Index));
            bool *vals = malloc(nvals * sizeof(bool));

            OK(GrB_Matrix_extractTuples_BOOL(rows, cols, vals, &nvals, A));

            // Expected new index: row 0, col = 65536 * 65537 + 65536 = 4,295,032,832
            printf("Extracted edge: (%" PRIu64 ", %" PRIu64 ")\n", rows[0], cols[0]);

            free(rows);
            free(cols);
            free(vals);
        }

        OK(GrB_Matrix_free(&A));
    }

    OK(GrB_finalize());
    return 0;
}

