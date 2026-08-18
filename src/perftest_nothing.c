//------------------------------------------------------------------------------
// gbperftest/gbperftest_nothing: basic utility
//------------------------------------------------------------------------------

// gbperftest, Timothy A. Davis, (c) 2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gbperftest.h"

void gbperftest_nothing (void)
{
    printf ("do nothing\n") ;

    char msg [LAGRAPH_MSG_LEN] ;
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_Finalize (msg)) ;
    printf ("ok\n") ;
}

