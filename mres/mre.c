// GrB_select with a built-in VALUE* op whose type is not A's type gives wrong results.
//
// By the C API spec (and the SuiteSparse:GraphBLAS User Guide), select casts each A(i) to the
// op's x type and the thunk to its y type, compares them, and keeps A(i) unchanged, with its
// own value and type, wherever the op is true.  So select(A, op, thunk) must keep exactly the
// entries where apply(A, op, thunk) is true, and each kept C(i) must equal A(i).  apply with
// the same op and thunk is the control.  Each case runs with A sparse and with A bitmap.
//
// There are two bugs:
//
//   1. Wrong entries.  The factory kernels compare A(i) with the thunk in A's type, reading
//      the thunk (already cast to the op's y type) as if it had A's type: FP64 1.0 is read as
//      the INT8 0x00.  When A's type is wider than the op's (FC64 A, VALUEEQ_FP64) this is
//      also a heap over-read, which AddressSanitizer reports if GraphBLAS itself is built
//      with -fsanitize=address.  Compact builds have no factory kernels and do not have it.
//   2. Wrong values.  VALUEEQ always returns C as iso, with the thunk (cast to A's type) as
//      its value, but after the cast to the op's type, different values of A can equal the
//      thunk: as INT64, both 1.0 and 1.5 are 1.  Every build has it, but where the factory
//      kernels run, bug 1 hides it in these cases.  In a compact build, or with only bug 1
//      fixed, the VALUEEQ_BOOL case gives {1, 1, 1} and the FP64 case gives {1, 1}.
//
//   ./mre         # bug present: prints each wrong case, exit 1
//   ./mre nojit   # the same with the JIT off
//
// Exit status: 0 every select is right; 1 a select is wrong; 2 a GraphBLAS call failed.
#include <complex.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "GraphBLAS.h"

#define OK(call)                                                       \
    do {                                                               \
        GrB_Info info = (call);                                        \
        if (info != GrB_SUCCESS) {                                     \
            fprintf(stderr, "%s failed: %d\n", #call, (int) info);    \
            return 2;                                                  \
        }                                                              \
    } while (0)

#define N 3
enum { A_INT8, A_FP64, A_FC64, NA };
static const char *anames[NA] = {"INT8 {1, 100, -5}", "FP64 {1, 1.5, 2}", "FC64 {1+2i, 5, 1}"};

// Write the entries of v to out as "{x, y, ...}", each cast to FC64 (so any type works).
static void show(GrB_Vector v, char *out, size_t size) {
    size_t n = snprintf(out, size, "{");
    for (GrB_Index i = 0; i < N && n < size; i++) {
        GxB_FC64_t x;
        if (GxB_Vector_extractElement_FC64(&x, v, i) != GrB_SUCCESS) continue;
        n += snprintf(out + n, size - n, "%s%g", n > 1 ? ", " : "", creal(x));
        if (cimag(x) != 0 && n < size) n += snprintf(out + n, size - n, "%+gi", cimag(x));
    }
    if (n < size) snprintf(out + n, size - n, "}");
}

int main(int argc, char **argv) {
    OK(GrB_init(GrB_NONBLOCKING));
    if (argc > 1 && strcmp(argv[1], "nojit") == 0)
        OK(GxB_Global_Option_set(GxB_JIT_C_CONTROL, GxB_JIT_OFF));
    printf("SuiteSparse:GraphBLAS %d.%d.%d (header)%s\n", GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR, GxB_IMPLEMENTATION_SUB, argc > 1 ? ", JIT off" : "");

    struct {
        int a;                  // which A
        GrB_IndexUnaryOp op;
        const char *opname;
        GrB_Type ttype;         // type of the thunk scalar (the op's type)
        double thunk;
    } cases[] = {
        // bug 1: the thunk's bytes are read as A's type
        {A_INT8, GrB_VALUEEQ_FP64, "GrB_VALUEEQ_FP64,  thunk 1.0 ", GrB_FP64, 1.0},
        {A_INT8, GrB_VALUEGE_FP64, "GrB_VALUEGE_FP64,  thunk 2.5 ", GrB_FP64, 2.5},
        {A_INT8, GrB_VALUEEQ_INT64, "GrB_VALUEEQ_INT64, thunk 356 ", GrB_INT64, 356},
        {A_INT8, GrB_VALUELT_INT64, "GrB_VALUELT_INT64, thunk 300 ", GrB_INT64, 300},
        // bug 2: C is iso, with the thunk as its value
        {A_INT8, GrB_VALUEEQ_BOOL, "GrB_VALUEEQ_BOOL,  thunk true", GrB_BOOL, 1},
        {A_FP64, GrB_VALUEEQ_INT64, "GrB_VALUEEQ_INT64, thunk 1   ", GrB_INT64, 1},
        // both, and the FC64 kernel reads 16 bytes of the 8-byte FP64 thunk
        {A_FC64, GrB_VALUEEQ_FP64, "GrB_VALUEEQ_FP64,  thunk 1.0 ", GrB_FP64, 1.0},
        // control: the op has A's type
        {A_INT8, GrB_VALUEEQ_INT8, "GrB_VALUEEQ_INT8,  thunk 1   ", GrB_INT8, 1},
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);
    GrB_Type atypes[NA] = {GrB_INT8, GrB_FP64, GxB_FC64};
    int nbad = 0;
    for (int format = 0; format < 2; format++) {
        GrB_Vector A[NA];
        // 3 entries of 1000: sparse unless forced
        for (int a = 0; a < NA; a++) OK(GrB_Vector_new(&A[a], atypes[a], 1000));
        const double i8[N] = {1, 100, -5}, f64[N] = {1, 1.5, 2};
        const GxB_FC64_t fc64[N] = {GxB_CMPLX(1, 2), GxB_CMPLX(5, 0), GxB_CMPLX(1, 0)};
        for (GrB_Index i = 0; i < N; i++) {
            OK(GrB_Vector_setElement_FP64(A[A_INT8], i8[i], i));
            OK(GrB_Vector_setElement_FP64(A[A_FP64], f64[i], i));
            OK(GxB_Vector_setElement_FC64(A[A_FC64], fc64[i], i));
        }
        for (int a = 0; a < NA; a++) {
            int sparsity = format ? GxB_BITMAP : GxB_SPARSE;
            OK(GxB_Vector_Option_set(A[a], GxB_SPARSITY_CONTROL, sparsity));
            OK(GrB_Vector_wait(A[a], GrB_MATERIALIZE));
        }
        for (int k = 0; k < ncases; k++) {
            GrB_Vector Ak = A[cases[k].a], C, B, W;
            GrB_Type atype = atypes[cases[k].a];
            GrB_Scalar thunk;
            OK(GrB_Scalar_new(&thunk, cases[k].ttype));
            OK(GrB_Scalar_setElement_FP64(thunk, cases[k].thunk));
            // C = select (A), B = apply (A), W = the entries of A where B is true
            OK(GrB_Vector_new(&C, atype, 1000));
            OK(GrB_Vector_new(&B, GrB_BOOL, 1000));
            OK(GrB_Vector_new(&W, atype, 1000));

            printf ("\n=================================================\n") ;
            GxB_print (atype, 3) ;
            GxB_print (C, 3) ;
            GxB_print (cases [k].op, 3) ;
            GxB_print (Ak, 3) ;
            GxB_print (thunk, 3) ;
            OK (GrB_set (GrB_GLOBAL, (int32_t) 1, GxB_BURBLE)) ;
            OK(GrB_Vector_select_Scalar(C, NULL, NULL, cases[k].op, Ak, thunk, NULL));
            OK (GrB_set (GrB_GLOBAL, (int32_t) 0, GxB_BURBLE)) ;
            GxB_print (C, 3) ;

            OK(GrB_Vector_apply_IndexOp_Scalar(B, NULL, NULL, cases[k].op, Ak, thunk, NULL));
            OK(GrB_Vector_assign(W, B, NULL, Ak, GrB_ALL, 1000, NULL));
            char got[128], want[128];
            show(C, got, sizeof(got));
            show(W, want, sizeof(want));
            bool ok = strcmp(got, want) == 0;
            printf("%s %-17s %s: select gives %-16s want %-16s %s\n",
                format ? "bitmap" : "sparse", anames[cases[k].a], cases[k].opname, got, want,
                ok ? "ok" : "WRONG");
            nbad += !ok;
            OK(GrB_Vector_free(&C));
            OK(GrB_Vector_free(&B));
            OK(GrB_Vector_free(&W));
            OK(GrB_Scalar_free(&thunk));
        }
        for (int a = 0; a < NA; a++) OK(GrB_Vector_free(&A[a]));
    }
    OK(GrB_finalize());
    printf(nbad ? "BUG: %d select(s) wrong\n" : "OK: every select is right\n", nbad);
    return nbad != 0;
}
