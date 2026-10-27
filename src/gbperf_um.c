//------------------------------------------------------------------------------
// gbperftest/src/gbperf_um.c: per-kernel CPU<->GPU unified-memory stats (CUPTI)
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gbperf_um.h"
#include <stdio.h>
#include <string.h>

#ifdef GBPERF_HAS_CUPTI

#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cupti.h>

//------------------------------------------------------------------------------
// tunables
//------------------------------------------------------------------------------

// CUPTI record buffer size (holds the UM + kernel records of one op).
#define GBPERF_UM_BUF_SIZE (256 * 1024)

// UM records are delivered to the completion callback a little after the kernel
// finishes, even after sync + forced flush. um_drain() flushes repeatedly until
// the record stream goes quiet: wait >= SETTLE_MIN ms, then stop after
// SETTLE_IDLE ms with no new records (capped at SETTLE_MAX).
#define GBPERF_UM_SETTLE_MIN_MS   5
#define GBPERF_UM_SETTLE_IDLE_MS 12
#define GBPERF_UM_SETTLE_MAX_MS 300

// A UM record in no kernel window is attributed to the most recent kernel that
// ended within this tolerance before it (its faults deferred the migration).
#define GBPERF_UM_ATTRIB_TOL_NS  2000000ULL     // 2 ms

// Maximum number of comma-separated substrings in the GBPERF_UM_KERNELS filter.
#define GBPERF_UM_MAXFILT 32

// The CUDA 13.x concurrent-kernel activity record. Bump if building against a
// CUPTI that defines a newer CUpti_ActivityKernel<N>.
typedef CUpti_ActivityKernel11 gbperf_akernel_t ;

// CUPTI_OK: run a CUPTI call and print a diagnostic if it does not succeed.
//   call - the CUPTI expression to evaluate (returns a CUptiResult)
#define CUPTI_OK(call)                                                        \
{                                                                             \
    CUptiResult cupti_status = (call) ;                                       \
    if (cupti_status != CUPTI_SUCCESS)                                        \
    {                                                                         \
        const char *errstr = NULL ;                                           \
        cuptiGetResultString (cupti_status, &errstr) ;                        \
        fprintf (stderr, "gbperf_um: CUPTI error %s:%d : %s -> %s\n",         \
            __FILE__, __LINE__, #call, errstr ? errstr : "(unknown)") ;       \
    }                                                                         \
}

//------------------------------------------------------------------------------
// data held per session
//------------------------------------------------------------------------------

// aggregate UM counters accumulated over one measurement window
typedef struct
{
    uint64_t bytes_htod ;       // bytes migrated host -> device
    uint64_t bytes_dtoh ;       // bytes migrated device -> host
    uint64_t cpu_page_faults ;  // # CPU page faults on managed memory
    uint64_t gpu_page_faults ;  // # GPU page-fault groups
    uint64_t thrash_count ;     // # thrashing events
    uint64_t thrash_bytes ;     // bytes implicated in thrashing
    uint64_t throttle_count ;   // # throttling events
    uint64_t throttle_bytes ;   // bytes implicated in throttling
}
um_stats_t ;

// one UM counter record, retained for per-kernel correlation
typedef struct
{
    int      kind ;             // CUpti_ActivityUnifiedMemoryCounterKind
    uint64_t value ;            // bytes / fault count / PC, depending on kind
    uint64_t start ;            // CUPTI ns timestamp (correlated to kernels)
}
um_rec_t ;

// one GPU kernel execution interval + its launch config, retained per record
typedef struct
{
    char    *name ;             // kernel name, owned copy (demangled on print)
    uint64_t start ;            // kernel start time, CUPTI ns timestamp
    uint64_t end ;              // kernel end time, CUPTI ns timestamp
    int      block ;            // threads per block (blockX*blockY*blockZ)
    int      grid ;             // blocks launched (gridX*gridY*gridZ)
    int      regs ;             // registers per thread
    int      shmem ;            // static + dynamic shared memory/block (bytes)
}
um_kern_t ;

// device limits needed for theoretical occupancy (filled by capture_devprops)
typedef struct
{
    int valid ;             // nonzero once the fields below are set
    int warpSize ;          // threads per warp
    int maxThreadsPerSM ;   // max resident threads per SM
    int regsPerSM ;         // registers per SM
    int shmemPerSM ;        // shared memory per SM, in bytes
    int maxBlocksPerSM ;    // hardware cap on blocks per SM
}
um_devprops_t ;

// the whole session state: no loose globals, one struct per handle
struct gbperf_um_ctx
{
    int     device ;                    // CUDA device being profiled
    pthread_mutex_t   lock ;            // guards acc / recs / kern / saved slot
    volatile int      collecting ;      // append records only when set
    volatile uint64_t nrec ;            // monotonic UM count (drives drain)

    um_stats_t acc ;                    // current-window aggregate totals
    um_rec_t  *recs ;                   // current-window UM records
    size_t nrecs ;                      // # UM records in recs
    size_t recs_cap ;                   // allocated capacity of recs
    um_kern_t *kern ;                   // current-window kernel records
    size_t nkern ;                      // # kernel records in kern
    size_t kern_cap ;                   // allocated capacity of kern

    int        s_valid ;                // nonzero if the saved slot is set
    um_stats_t s_acc ;                  // saved (fastest) window totals
    um_rec_t  *s_recs ;                 // saved (fastest) window UM records
    size_t s_nrecs ;                    // # UM records in s_recs
    um_kern_t *s_kern ;                 // saved (fastest) window kernel records
    size_t s_nkern ;                    // # kernel records in s_kern

    um_devprops_t dev ;                 // device limits, for occupancy

    char *filt_store ;                  // owned copy of GBPERF_UM_KERNELS
    char *filt [GBPERF_UM_MAXFILT] ;    // filter substrings (into filt_store)
    int nfilt ;                         // # filter substrings in filt
} ;

// CUPTI's buffer-completed callback carries no user pointer and its UM counters
// are per-process, so exactly one session can be active at a time. This is that
// session; only buffer_completed() reads it.
static struct gbperf_um_ctx *g_active = NULL ;

//------------------------------------------------------------------------------
// small helpers
//------------------------------------------------------------------------------

// dupstr: return a malloc'd copy of a string ("?" if s is NULL; NULL on OOM)
static char *dupstr
(
    const char *s  // string to copy
)
{
    if (s == NULL)
    {
        s = "?" ;
    }
    size_t n = strlen (s) + 1 ;
    char *p = (char *) malloc (n) ;
    if (p != NULL)
    {
        memcpy (p, s, n) ;
    }
    return (p) ;
}

// demangle_short: extract the readable identifier from an Itanium C++ name,
// handling _Z<len><name>... and _ZN<len><name>...E (first component); falls
// back to copying the raw string.
static void demangle_short
(
    const char *s,  // mangled (or plain) symbol name
    char *out,      // output buffer for the extracted name
    size_t outsz    // size of out, in bytes
)
{
    const char *p = s ;
    if (p != NULL && p [0] == '_' && p [1] == 'Z')
    {
        p += 2 ;
        if (*p == 'N')
        {
            p++ ;
        }
        if (*p >= '0' && *p <= '9')
        {
            long len = strtol (p, (char **) &p, 10) ;
            if (len > 0)
            {
                size_t n = (size_t) len ;
                if (n >= outsz)
                {
                    n = outsz - 1 ;
                }
                memcpy (out, p, n) ;
                out [n] = '\0' ;
                return ;
            }
        }
    }
    snprintf (out, outsz, "%s", s ? s : "?") ;
}

// capture_devprops: query the device's SM limits (used by occupancy_pct) into
// *dev; dev->valid stays 0 if the query fails.
static void capture_devprops
(
    int device,         // CUDA device id to query
    um_devprops_t *dev  // output: device limits to fill
)
{
    struct cudaDeviceProp p ;
    memset (dev, 0, sizeof (*dev)) ;

    if (cudaGetDeviceProperties (&p, device) == cudaSuccess)
    {
        dev->warpSize        = p.warpSize ;
        dev->maxThreadsPerSM = p.maxThreadsPerMultiProcessor ;
        dev->regsPerSM       = p.regsPerMultiprocessor ;
        dev->shmemPerSM      = (int) p.sharedMemPerMultiprocessor ;
        dev->maxBlocksPerSM  = p.maxBlocksPerMultiProcessor ;
        dev->valid           = 1 ;
    }
}

// occupancy_pct: theoretical (launch) occupancy as a percent of peak warps/SM,
// limited by threads, registers, shared memory and the hardware block cap.
// Approximate: 256-reg/warp allocation granularity; ignores shared-mem
// granularity. Returns 0 if device limits are unknown.
static int occupancy_pct
(
    const um_devprops_t *dev,  // device limits (from capture_devprops)
    int block,                 // threads per block
    int regs,                  // registers per thread
    int shmem,                 // shared memory per block, in bytes
    const char **limit         // output: capping resource (short name)
)
{
    *limit = "-" ;
    if (!dev->valid || block <= 0)
    {
        return (0) ;
    }

    int ws = dev->warpSize > 0 ? dev->warpSize : 32 ;
    int warps_per_block = (block + ws - 1) / ws ;
    int max_warps_sm = dev->maxThreadsPerSM / ws ;
    if (warps_per_block <= 0 || max_warps_sm <= 0)
    {
        return (0) ;
    }

    int b_threads = dev->maxThreadsPerSM / block ;
    int b_hw   = dev->maxBlocksPerSM > 0 ? dev->maxBlocksPerSM : b_threads ;
    int b_regs = 1 << 30 ;
    if (regs > 0 && dev->regsPerSM > 0)
    {
        int regs_per_warp = regs * ws ;
        regs_per_warp = ((regs_per_warp + 255) / 256) * 256 ;
        int warps_by_regs = dev->regsPerSM / regs_per_warp ;
        b_regs = warps_by_regs / warps_per_block ;
    }
    int b_smem = 1 << 30 ;
    if (shmem > 0 && dev->shmemPerSM > 0)
    {
        b_smem = dev->shmemPerSM / shmem ;
    }

    int blocks = b_threads ;
    const char *lim = "thr" ;
    if (b_hw < blocks)
    {
        blocks = b_hw ;
        lim = "blk" ;
    }
    if (b_regs < blocks)
    {
        blocks = b_regs ;
        lim = "reg" ;
    }
    if (b_smem < blocks)
    {
        blocks = b_smem ;
        lim = "smem" ;
    }
    if (blocks < 0)
    {
        blocks = 0 ;
    }

    int active_warps = blocks * warps_per_block ;
    if (active_warps > max_warps_sm)
    {
        active_warps = max_warps_sm ;
    }
    *limit = lim ;
    return (int) (100.0 * active_warps / max_warps_sm + 0.5) ;
}

//------------------------------------------------------------------------------
// record intake (CUPTI callbacks)
//------------------------------------------------------------------------------

// accumulate: fold one UM counter record into the session's running totals
static void accumulate
(
    struct gbperf_um_ctx *c,
    const CUpti_ActivityUnifiedMemoryCounter3 *um  // the UM counter record
)
{
    switch (um->counterKind)
    {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
        {
            c->acc.bytes_htod += um->value ;
            break ;
        }
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
        {
            c->acc.bytes_dtoh += um->value ;
            break ;
        }
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
        {
            c->acc.cpu_page_faults += 1 ;
            break ;  // value is a PC, not a count
        }
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
        {
            c->acc.gpu_page_faults += um->value ;
            break ;
        }
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
        {
            c->acc.thrash_count += 1 ; c->acc.thrash_bytes += um->value ;
            break ;
        }
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
        {
            c->acc.throttle_count += 1 ;
            c->acc.throttle_bytes += um->value ;
            break ;
        }
        default:
        {
            break ;
        }
    }
}

// buffer_requested: CUPTI callback - hand CUPTI a buffer to fill with records
static void CUPTIAPI buffer_requested
(
    uint8_t **buffer,      // output: the allocated buffer
    size_t *size,          // output: its size, in bytes
    size_t *maxNumRecords  // output: max records to store (0 = as many as fit)
)
{
    *buffer = (uint8_t *) malloc (GBPERF_UM_BUF_SIZE) ;
    *size = (*buffer != NULL) ? GBPERF_UM_BUF_SIZE : 0 ;
    *maxNumRecords = 0 ;
}

// buffer_completed: CUPTI callback - fold a completed record buffer into the
// active session (g_active), then free the buffer. UM records always bump
// c->nrec (to drive the drain) but are only stored while c->collecting is set.
static void CUPTIAPI buffer_completed
(
    CUcontext ctx,      // CUDA context (unused)
    uint32_t streamId,  // stream id (unused)
    uint8_t *buffer,    // the record buffer to drain
    size_t size,        // buffer size, in bytes (unused)
    size_t validSize    // bytes of valid records in buffer
)
{
    struct gbperf_um_ctx *c = g_active ;
    (void) ctx ; (void) streamId ; (void) size ;

    if (c == NULL)
    {
        free (buffer) ;
        return ;
    }

    CUptiResult status ;
    CUpti_Activity *record = NULL ;

    pthread_mutex_lock (&c->lock) ;
    if (validSize > 0)
    {
        for ( ; ; )
        {
            status = cuptiActivityGetNextRecord (buffer, validSize, &record) ;
            if (status != CUPTI_SUCCESS)
            {
                if (status != CUPTI_ERROR_MAX_LIMIT_REACHED)
                {
                    CUPTI_OK (status) ;
                }
                break ;
            }

            if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)
            {
                const CUpti_ActivityUnifiedMemoryCounter3 *um =
                    (const CUpti_ActivityUnifiedMemoryCounter3 *) record ;
                c->nrec++ ;
                if (c->collecting)
                {
                    accumulate (c, um) ;
                    if (c->nrecs == c->recs_cap)
                    {
                        size_t nc = c->recs_cap ? 2 * c->recs_cap : 256 ;
                        um_rec_t *np = (um_rec_t *)
                            realloc (c->recs, nc * sizeof (um_rec_t)) ;
                        if (np != NULL)
                        {
                            c->recs = np ;
                            c->recs_cap = nc ;
                        }
                    }
                    if (c->nrecs < c->recs_cap)
                    {
                        c->recs [c->nrecs].kind  = (int) um->counterKind ;
                        c->recs [c->nrecs].value = um->value ;
                        c->recs [c->nrecs].start = um->start ;
                        c->nrecs++ ;
                    }
                }
            }
            else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
            {
                if (c->collecting)
                {
                    const gbperf_akernel_t *k =
                        (const gbperf_akernel_t *) record ;
                    if (c->nkern == c->kern_cap)
                    {
                        size_t nc = c->kern_cap ? 2 * c->kern_cap : 64 ;
                        um_kern_t *np = (um_kern_t *)
                            realloc (c->kern, nc * sizeof (um_kern_t)) ;
                        if (np != NULL)
                        {
                            c->kern = np ;
                            c->kern_cap = nc ;
                        }
                    }
                    if (c->nkern < c->kern_cap)
                    {
                        c->kern [c->nkern].name  = dupstr (k->name) ;
                        c->kern [c->nkern].start = k->start ;
                        c->kern [c->nkern].end   = k->end ;
                        c->kern [c->nkern].block =
                            k->blockX * k->blockY * k->blockZ ;
                        c->kern [c->nkern].grid =
                            k->gridX * k->gridY * k->gridZ ;
                        c->kern [c->nkern].regs  = k->registersPerThread ;
                        c->kern [c->nkern].shmem = k->staticSharedMemory +
                                                   k->dynamicSharedMemory ;
                        c->nkern++ ;
                    }
                }
            }
        }
    }
    pthread_mutex_unlock (&c->lock) ;
    free (buffer) ;
}

//------------------------------------------------------------------------------
// process-global CUPTI configuration (no per-session state)
//------------------------------------------------------------------------------

// configure_counters: enable or disable the six UM counter kinds, process-wide
// returns 0 on success, 1 on failure
static int configure_counters
(
    int device,      // CUDA device id
    uint32_t enable, // nonzero to enable the counters, zero to disable them
    int quiet        // suppress the "unavailable" warning on failure
)
{
    const CUpti_ActivityUnifiedMemoryCounterKind kinds [6] =
    {
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT,
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING,
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING
    } ;

    CUpti_ActivityUnifiedMemoryCounterConfig cfg [6] ;
    for (int i = 0 ; i < 6 ; i++)
    {
        cfg [i].scope =
            CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE ;
        cfg [i].kind     = kinds [i] ;
        cfg [i].deviceId = (uint32_t) device ;
        cfg [i].enable   = enable ;
    }

    CUptiResult s = cuptiActivityConfigureUnifiedMemoryCounter (cfg, 6) ;
    if (s != CUPTI_SUCCESS)
    {
        if (!quiet)
        {
            const char *errstr = NULL ;
            cuptiGetResultString (s, &errstr) ;
            fprintf (stderr, "gbperf_um: unified-memory counters unavailable "
                "on device %d: %s\n", device, errstr ? errstr : "(unknown)") ;
        }
        return (1) ;
    }
    return (0) ;
}

// um_drain: sync the device, then force-flush CUPTI until the UM record stream
// goes quiet, so a kernel's records are fully delivered before they are read.
static void um_drain
(
    struct gbperf_um_ctx *c
)
{
    cudaDeviceSynchronize ( ) ;
    int idle_ms = 0 ;
    for (int ms = 0 ; ms < GBPERF_UM_SETTLE_MAX_MS ; ms++)
    {
        uint64_t before = c->nrec ;
        CUPTI_OK (cuptiActivityFlushAll (CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)) ;
        if (c->nrec != before)
        {
            idle_ms = 0 ;
        }
        else
        {
            idle_ms++ ;
        }
        if (ms >= GBPERF_UM_SETTLE_MIN_MS &&
            idle_ms >= GBPERF_UM_SETTLE_IDLE_MS)
        {
            break ;
        }
        usleep (1000) ;
    }
}

//------------------------------------------------------------------------------
// kernel-name filter
//------------------------------------------------------------------------------

// parse_filter: split the GBPERF_UM_KERNELS env var into c->filt[] substrings
// (empty/unset leaves the filter off, so all kernels print).
static void parse_filter
(
    struct gbperf_um_ctx *c
)
{
    c->nfilt = 0 ;
    c->filt_store = NULL ;
    const char *env = getenv ("GBPERF_UM_KERNELS") ;
    if (env == NULL || *env == '\0')
    {
        return ;
    }
    c->filt_store = dupstr (env) ;
    if (c->filt_store == NULL)
    {
        return ;
    }
    char *tok = strtok (c->filt_store, ",") ;
    while (tok != NULL && c->nfilt < GBPERF_UM_MAXFILT)
    {
        if (*tok != '\0')
        {
            c->filt [c->nfilt++] = tok ;
        }
        tok = strtok (NULL, ",") ;
    }
}

// filter_pass: true if name passes the filter (true if the filter is off)
static int filter_pass
(
    struct gbperf_um_ctx *c,  // session holding the filter substrings
    const char *name          // kernel name to test
)
{
    if (c->nfilt == 0)
    {
        return (1) ;
    }
    for (int i = 0 ; i < c->nfilt ; i++)
    {
        if (strstr (name, c->filt [i]) != NULL)
        {
            return (1) ;
        }
    }
    return (0) ;
}

//------------------------------------------------------------------------------
// per-kernel breakdown
//------------------------------------------------------------------------------

// one row of the per-kernel table: traffic + launch config, by kernel name
typedef struct
{
    const char *name ;          // kernel name, or the host-bucket label
    uint64_t h2d ;              // bytes migrated host -> device
    uint64_t d2h ;              // bytes migrated device -> host
    uint64_t cpuPF ;            // # CPU page faults
    uint64_t gpuPF ;            // # GPU page-fault groups
    uint64_t thrash ;           // # thrashing events
    uint64_t throttle ;         // # throttling events
    uint64_t active_ns ;        // summed kernel execution time, in ns
    uint64_t nlaunch ;          // # launches of this kernel
    int block ;                 // threads per block (from first launch)
    int grid ;                  // blocks launched (from first launch)
    int regs ;                  // registers per thread (from first launch)
    int shmem ;                 // shared memory per block, bytes (first launch)
}
um_bucket_t ;

// find_kernel_in: index of the kernel owning timestamp t: the innermost
// kernel containing t, else the most recent kernel ending within the
// tolerance before t, else -1 (host / between kernels).
static int find_kernel_in
(
    const um_kern_t *kern,  // array of kernel records
    size_t nkern,           // number of kernel records
    uint64_t t              // UM record timestamp, in CUPTI ns
)
{
    int best = -1 ;
    uint64_t best_start = 0 ;
    for (size_t i = 0 ; i < nkern ; i++)
    {
        if (kern [i].start <= t && t < kern [i].end)
        {
            if (best < 0 || kern [i].start >= best_start)
            {
                best = (int) i ;
                best_start = kern [i].start ;
            }
        }
    }
    if (best >= 0)
    {
        return (best) ;
    }

    int recent = -1 ;
    uint64_t best_end = 0 ;
    for (size_t i = 0 ; i < nkern ; i++)
    {
        if (kern [i].end <= t && (t - kern [i].end) <= GBPERF_UM_ATTRIB_TOL_NS)
        {
            if (recent < 0 || kern [i].end > best_end)
            {
                recent = (int) i ;
                best_end = kern [i].end ;
            }
        }
    }
    return (recent) ;
}

// bucket_for: find the bucket named name in b[0..*nb), creating it if absent
static um_bucket_t *bucket_for
(
    um_bucket_t *b,  // bucket array (assumed to have room for a new entry)
    int *nb,         // in/out: number of buckets in use
    const char *name // bucket (kernel) name to find or add
)
{
    for (int i = 0 ; i < *nb ; i++)
    {
        if (strcmp (b [i].name, name) == 0)
        {
            return (&b [i]) ;
        }
    }
    memset (&b [*nb], 0, sizeof (um_bucket_t)) ;
    b [*nb].name = name ;
    return (&b [(*nb)++]) ;
}

// print_summary: print the one-line aggregate UM summary for a window
static void print_summary
(
    const char *tag,     // label for the operation (e.g. "A+B")
    const um_stats_t *s  // the aggregate counters to print
)
{
    const double MB = 1024.0 * 1024.0 ;
    printf ("    [UM %-8s] H2D %9.3f MB  D2H %9.3f MB  "
        "cpuPF %6llu  gpuPF %6llu  thrash %4llu (%7.3f MB)  throttle %4llu\n",
        tag ? tag : "",
        s->bytes_htod / MB, s->bytes_dtoh / MB,
        (unsigned long long) s->cpu_page_faults,
        (unsigned long long) s->gpu_page_faults,
        (unsigned long long) s->thrash_count, s->thrash_bytes / MB,
        (unsigned long long) s->throttle_count) ;
}

// bucket_add: fold one UM record's value into a kernel bucket, by counter kind
static void bucket_add
(
    um_bucket_t *bk,  // destination bucket
    int kind,         // CUpti_ActivityUnifiedMemoryCounterKind of the record
    uint64_t v        // the record's value
)
{
    switch (kind)
    {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
        {
            bk->h2d += v ;
            break ;
        }

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
        {
            bk->d2h += v ;
            break ;
        }

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
        {
            bk->cpuPF += 1 ;
            break ;
        }

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
        {
            bk->gpuPF += v ;
            break ;
        }

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
        {
            bk->thrash += 1 ;
            break ;
        }

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
        {
            bk->throttle += 1 ;
            break ;
        }

        default:
        {
            break ;
        }
    }
}

// print_breakdown: print the per-kernel table: attribute each UM record to
// the kernel running at its timestamp, aggregate by name, and print each
// with its launch config and occupancy (filtered by GBPERF_UM_KERNELS).
static void print_breakdown
(
    struct gbperf_um_ctx *c,
    const char *tag,        // label for the operation
    const um_rec_t *recs,   // UM records of the window
    size_t nrecs,           // number of UM records
    const um_kern_t *kern,  // kernel records of the window
    size_t nkern            // number of kernel records
)
{
    static const char *HOST = "(host/between-kernels)" ;
    int cap = (int) nkern + 2 ;
    um_bucket_t *b = (um_bucket_t *) calloc (cap, sizeof (um_bucket_t)) ;
    int nb = 0 ;
    if (b != NULL)
    {
        for (size_t i = 0 ; i < nkern ; i++)
        {
            um_bucket_t *bk = bucket_for (b, &nb, kern [i].name) ;
            if (bk->nlaunch == 0)
            {
                bk->block = kern [i].block ;
                bk->grid  = kern [i].grid ;
                bk->regs  = kern [i].regs ;
                bk->shmem = kern [i].shmem ;
            }
            bk->active_ns += (kern [i].end - kern [i].start) ;
            bk->nlaunch++ ;
        }
        for (size_t i = 0 ; i < nrecs ; i++)
        {
            int ki = find_kernel_in (kern, nkern, recs [i].start) ;
            const char *name = (ki >= 0) ? kern [ki].name : HOST ;
            um_bucket_t *bk = bucket_for (b, &nb, name) ;
            bucket_add (bk, recs [i].kind, recs [i].value) ;
        }
    }

    const double MB = 1024.0 * 1024.0 ;
    int nkernels = 0 ;
    for (int i = 0 ; i < nb ; i++)
    {
        if (b [i].name != HOST)
        {
            nkernels++ ;
        }
    }
    printf ("    [UM %s by kernel]  "
        "(%d kernel%s, %zu launch%s, %zu UM records)\n",
        tag ? tag : "", nkernels, nkernels == 1 ? "" : "s",
        nkern, nkern == 1 ? "" : "es", nrecs) ;

    printf ("      %-38s %8s %8s %6s %8s %6s %6s %7s %4s %6s %4s %5s\n",
        "kernel", "H2D(MB)", "D2H(MB)", "gpuPF", "active", "GB/s",
        "block", "grid", "reg", "smem", "occ%", "cap") ;

    for (int i = 0 ; i < nb ; i++)
    {
        char nm [128] ;
        if (b [i].name == HOST)
        {
            snprintf (nm, sizeof (nm), "%s", HOST) ;
        }
        else
        {
            demangle_short (b [i].name, nm, sizeof (nm)) ;
        }
        if (b [i].name != HOST && !filter_pass (c, nm))
        {
            continue ;
        }

        double act_s = b [i].active_ns / 1e9 ;
        double moved = (double) (b [i].h2d + b [i].d2h) ;
        char rate [12], act [12] ;
        if (b [i].active_ns > 0)
        {
            snprintf (rate, sizeof (rate), "%6.2f", (moved / act_s) / 1e9) ;
            snprintf (act, sizeof (act), "%6.3fms", b [i].active_ns / 1e6) ;
        }
        else
        {
            snprintf (rate, sizeof (rate), "%6s", "-") ;
            snprintf (act, sizeof (act), "%8s", "-") ;
        }

        char blk [16], grd [8], reg [6], shm [8], occ [6], cap [6] ;
        if (b [i].name != HOST && b [i].block > 0)
        {
            const char *lim ;
            int o = occupancy_pct (&c->dev, b [i].block, b [i].regs,
                b [i].shmem, &lim) ;
            snprintf (blk, sizeof (blk), "%d", b [i].block) ;
            snprintf (grd, sizeof (grd), "%d", b [i].grid) ;
            snprintf (reg, sizeof (reg), "%d", b [i].regs) ;
            snprintf (shm, sizeof (shm), "%d", b [i].shmem) ;
            snprintf (occ, sizeof (occ), "%d", o) ;
            snprintf (cap, sizeof (cap), "%s",
                (o > 0 && o < 100) ? lim : "-") ;
        }
        else
        {
            snprintf (blk, sizeof (blk), "-") ;
            snprintf (grd, sizeof (grd), "-") ;
            snprintf (reg, sizeof (reg), "-") ;
            snprintf (shm, sizeof (shm), "-") ;
            snprintf (occ, sizeof (occ), "-") ;
            snprintf (cap, sizeof (cap), "-") ;
        }

        printf ("      %-38.38s %8.3f %8.3f %6llu %8s %6s "
            "%6s %7s %4s %6s %4s %5s\n",
            nm, b [i].h2d / MB, b [i].d2h / MB,
            (unsigned long long) b [i].gpuPF, act, rate,
            blk, grd, reg, shm, occ, cap) ;
    }
    free (b) ;
}

//------------------------------------------------------------------------------
// lifecycle helpers
//------------------------------------------------------------------------------

// clear_records: drop the current window's records (caller holds c->lock)
static void clear_records
(
    struct gbperf_um_ctx *c
)
{
    for (size_t i = 0 ; i < c->nkern ; i++)
    {
        free (c->kern [i].name) ;
    }
    c->nrecs = 0 ;
    c->nkern = 0 ;
}

// free_saved: free the saved (fastest) window and mark it empty (holds c->lock)
static void free_saved
(
    struct gbperf_um_ctx *c
)
{
    for (size_t i = 0 ; i < c->s_nkern ; i++)
    {
        free (c->s_kern [i].name) ;
    }

    free (c->s_kern) ;
    c->s_kern = NULL ;
    c->s_nkern = 0 ;

    free (c->s_recs) ;
    c->s_recs = NULL ;
    c->s_nrecs = 0 ;

    c->s_valid = 0 ;
}

//------------------------------------------------------------------------------
// public API  (argument docs are in include/gbperf_um.h)
//------------------------------------------------------------------------------

// gbperf_um_create: open the one CUPTI session and return its handle
gbperf_um_t gbperf_um_create
(
    int device  // CUDA device id to profile
)
{
    if (g_active != NULL)
    {
        return (NULL) ;  // only one session per process (CUPTI constraint)
    }

    struct gbperf_um_ctx *c =
        (struct gbperf_um_ctx *) calloc (1, sizeof (struct gbperf_um_ctx)) ;
    if (c == NULL)
    {
        return (NULL) ;
    }

    c->device = device ;
    pthread_mutex_init (&c->lock, NULL) ;
    capture_devprops (c->device, &c->dev) ;
    parse_filter (c) ;

    g_active = c ;  // visible to buffer_completed
    CUPTI_OK (cuptiActivityRegisterCallbacks (buffer_requested,
        buffer_completed)) ;

    if (configure_counters (device, 1, /* quiet: */ 0) != 0)
    {
        g_active = NULL ;
        free (c->filt_store) ;
        pthread_mutex_destroy (&c->lock) ;
        free (c) ;
        return (NULL) ;
    }

    CUPTI_OK (cuptiActivityEnable (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) ;
    CUPTI_OK (cuptiActivityEnable (
        CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)) ;
    return (c) ;
}

// gbperf_um_reset: begin a measurement window (c may be NULL)
void gbperf_um_reset
(
    gbperf_um_t c
)
{
    if (c == NULL)
    {
        return ;
    }

    c->collecting = 0 ;
    um_drain (c) ;
    pthread_mutex_lock (&c->lock) ;
    memset (&c->acc, 0, sizeof (c->acc)) ;
    clear_records (c) ;
    pthread_mutex_unlock (&c->lock) ;
    c->collecting = 1 ;
}

// gbperf_um_sample: close the window, delivering its records (c may be NULL)
void gbperf_um_sample
(
    gbperf_um_t c
)
{
    if (c == NULL)
    {
        return ;
    }

    um_drain (c) ;
    c->collecting = 0 ;
}

// gbperf_um_keep: snapshot the current window into the saved slot
void gbperf_um_keep
(
    gbperf_um_t c
)
{
    if (c == NULL)
    {
        return ;
    }

    pthread_mutex_lock (&c->lock) ;
    free_saved (c) ;
    c->s_acc = c->acc ;
    if (c->nrecs > 0)
    {
        c->s_recs = (um_rec_t *) malloc (c->nrecs * sizeof (um_rec_t)) ;
        if (c->s_recs != NULL)
        {
            memcpy (c->s_recs, c->recs, c->nrecs * sizeof (um_rec_t)) ;
            c->s_nrecs = c->nrecs ;
        }
    }

    if (c->nkern > 0)
    {
        c->s_kern = (um_kern_t *) malloc (c->nkern * sizeof (um_kern_t)) ;
        if (c->s_kern != NULL)
        {
            for (size_t i = 0 ; i < c->nkern ; i++)
            {
                c->s_kern [i] = c->kern [i] ;
                c->s_kern [i].name = dupstr (c->kern [i].name) ;
            }
            c->s_nkern = c->nkern ;
        }
    }

    c->s_valid = 1 ;
    pthread_mutex_unlock (&c->lock) ;
}

// gbperf_um_report: print the saved (fastest) window (c may be NULL)
void gbperf_um_report
(
    gbperf_um_t c,
    const char *tag // label for the operation
)
{
    if (c == NULL)
    {
        return ;
    }

    pthread_mutex_lock (&c->lock) ;
    if (!c->s_valid)
    {
        printf ("    [UM %s] (no collected run)\n", tag ? tag : "") ;
    }
    else
    {
        print_summary (tag, &c->s_acc) ;
        print_breakdown (c, tag, c->s_recs, c->s_nrecs, c->s_kern, c->s_nkern) ;
    }
    pthread_mutex_unlock (&c->lock) ;
}

// gbperf_um_destroy: disable CUPTI, free the session (c may be NULL)
void gbperf_um_destroy
(
    gbperf_um_t c
)
{
    if (c == NULL)
    {
        return ;
    }

    CUPTI_OK (cuptiActivityFlushAll (CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)) ;
    CUPTI_OK (cuptiActivityDisable (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) ;
    CUPTI_OK (cuptiActivityDisable (
        CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)) ;
    configure_counters (c->device, 0, /* quiet: */ 1) ;
    g_active = NULL ;                           // stop callbacks

    pthread_mutex_lock (&c->lock) ;
    clear_records (c) ;
    free_saved (c) ;
    pthread_mutex_unlock (&c->lock) ;
    free (c->recs) ;
    free (c->kern) ;
    free (c->filt_store) ;
    pthread_mutex_destroy (&c->lock) ;
    free (c) ;
}

#else

//------------------------------------------------------------------------------
// stubs (no CUPTI): keep the benchmarks building/running on non-CUDA hosts
//------------------------------------------------------------------------------

gbperf_um_t gbperf_um_create
(
    int device
)
{
    (void) device ;
    return (NULL) ;
}

void gbperf_um_reset
(
    gbperf_um_t c
)
{
    (void) c ;
}

void gbperf_um_sample
(
    gbperf_um_t c
)
{
    (void) c ;
}

void gbperf_um_keep
(
    gbperf_um_t c
)
{
    (void) c ;
}

void gbperf_um_report
(
    gbperf_um_t c,
    const char *tag
)
{
    (void) c ;
    (void) tag ;
}

void gbperf_um_destroy
(
    gbperf_um_t c
)
{
    (void) c ;
}

#endif
