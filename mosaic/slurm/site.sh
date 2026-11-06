# Site-specific setup, sourced at the start of every job. Edit this for your cluster.
# This version is an example for Grendel (Aarhus), assuming the conda
# environment from mosaic/environment.yml (see the README for using the
# cluster's own MPI instead); change the module names as needed.

# Python with numpy, numba, mpi4py and mpi4py-fft, for the split and the power spectrum
setup_python() {
    module load anaconda3
    conda activate mosaic
}

# DTFE, with the libraries it was built against
setup_dtfe() {
    module load gcc
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(dirname "$DTFE_BIN")/external/lib"
}

# How to start an MPI program inside a job. With mpi4py built against the
# cluster's MPI use srun (--export=ALL passes the environment set up above on to
# the ranks); with the MPI from conda use its mpirun.
MPIRUN="srun --export=ALL"
