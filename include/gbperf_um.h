//------------------------------------------------------------------------------
// gbperftest/include/gbperf_um.h: per-kernel CPU<->GPU unified-memory stats
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Measures *actual* host<->device traffic for CUDA Unified (managed) memory
// using the CUPTI Activity API, and attributes it per GPU kernel by correlating
// UM records with concurrent-kernel records by timestamp. See gbperf_um.c.
//
// API is a managed opaque handle (gbperf_um_t):
//
//     gbperf_um_t um = gbperf_um_create (device) ;    // NULL if unavailable
//     for (trials) {
//         gbperf_um_reset (um) ;                      // before the op
//         ... run the GraphBLAS op ...
//         gbperf_um_sample (um) ;                     // after the op
//         if (fastest_so_far) gbperf_um_keep (um) ;   // snapshot the fastest
//     }
//     gbperf_um_report (um, "tag") ;                  // print the fastest run
//     gbperf_um_destroy (um) ;
//
// Every call is NULL-safe: if gbperf_um_create() returns NULL (no CUPTI, UM
// profiling unsupported, or a session is already active), all other calls are
// no-ops, so callers need no conditionals.
//
// NOTE: CUPTI's activity callbacks carry no user pointer and its UM
// counters are per-process, so only ONE session can be active per process.
// The handle wraps that single session; creating a second returns NULL.

#ifndef GBPERF_UM_H
#define GBPERF_UM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gbperf_um_ctx *gbperf_um_t ;

// Create a profiling session. Returns NULL if CUPTI / UM profiling is
// unavailable or a session is already active. Call after GraphBLAS has been
// initialized (a CUDA context must exist).
gbperf_um_t gbperf_um_create
(
    int device  // CUDA device id to profile
) ;

// Begin a measurement window: discard any pending traffic and start collecting.
// Call immediately before the op to be measured.
void gbperf_um_reset
(
    gbperf_um_t um  // session handle (all calls are NULL-safe)
) ;

// Close the window: sync the device and force-flush CUPTI so the op's records
// are fully delivered. Call immediately after the op.
void gbperf_um_sample
(
    gbperf_um_t um  // session handle
) ;

// Snapshot the just-sampled window into the "saved" slot, replacing any prior
// snapshot. The harness runs each op several times and reports the fastest;
// call this only when the current trial is the fastest so far.
void gbperf_um_keep
(
    gbperf_um_t um  // session handle
) ;

// Print the saved (fastest-run) stats: a one-line summary plus a per-kernel
// breakdown (H2D/D2H, faults, active time, launch config, theoretical
// occupancy). If env GBPERF_UM_KERNELS is set (comma-separated substrings),
// only matching kernel names are shown.
void gbperf_um_report
(
    gbperf_um_t um,  // session handle
    const char *tag  // label for the operation (e.g. "A+B")
) ;

// Destroy the session and free all resources.
void gbperf_um_destroy
(
    gbperf_um_t um  // session handle
) ;

#ifdef __cplusplus
}
#endif

#endif
