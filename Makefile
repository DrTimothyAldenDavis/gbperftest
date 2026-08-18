#-------------------------------------------------------------------------------
# gbperftest/Makefile
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

JOBS ?= 8

F = -DSUITESPARSE_USE_FORTRAN=OFF

default: library

library:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# compile with -g for debugging
debug:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Release -j${JOBS} )

all: library

# remove all files not in the distribution
clean: distclean

purge: distclean

distclean:
	- $(RM) -rf build/* Config/*.tmp

