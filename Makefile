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

# quick tests with small matrices
test: library
	# gbperf_build: m n nvals seed
	./build/gbperf_build 1000 2000 10000 2
	./build/gbperf_build 1000 1000 10000 2
	# gbperf_malloc: nbig nmalloc ntrials
	./build/gbperf_malloc 1000 10 1000
	# gbperf_select: m n nvals
	./build/gbperf_select 1000 1000 10000
	# gbperf_transpose: m n nvals
	./build/gbperf_transpose 1000 1000 10000
	# gbperf_trianglecount: matrixmarketfile.mtx
	./build/gbperf_trianglecount matrices/karate.mtx

# performance tests
perf: library
	# gbperf_transpose: m n nvals
	./build/gbperf_transpose 1000000 1000000 1000000000
	# gbperf_build: m n nvals seed
	./build/gbperf_build 1000000 1000000 1000000000 1

tri:
	# gbperf_trianglecount: matrixmarketfile.mtx
	./build/gbperf_trianglecount /raid/GAP/GAP-road/GAP-road.grb
	./build/gbperf_trianglecount /raid/GAP/GAP-kron/GAP-kron.grb

just: library
	./build/gbperf_build 100 100 20 2

quick: library
	./build/gbperf_build 1000 1000 10000 2
