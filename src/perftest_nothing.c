//------------------------------------------------------------------------------
// perftest/perftest_nothing: basic utility
//------------------------------------------------------------------------------

// perftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "perftest.h"

void perftest_nothing (void)
{
    printf ("do nothing\n") ;

    char msg [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_Finalize (msg)) ;
    printf ("ok\n") ;
}

