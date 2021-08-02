CXX=c++
CXX_FLAGS= -O3 -std=c++17 -lm -Wall -Wextra -Xpreprocessor -fopenmp -mavx
OPENMP = -Xpreprocessor -fopenmp 

MPICXX = mpicxx
MPICXX_FLAGS = --std=c++17 -mavx -O3 -Wall -Wextra -g -DOMPI_SKIP_MPICXX
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gnu compilers from version 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157





#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
hybrid: gcn_hybrid.cpp Model.cpp Model.hpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o hybrid gcn_hybrid.cpp Model.cpp Node.cpp

run_hybrid:
	mpirun -np 2 ./hybrid
#-----------------------------------------------------------------------------------------#


clean:
	rm -rf *.o sequential omp mpi hybrid
