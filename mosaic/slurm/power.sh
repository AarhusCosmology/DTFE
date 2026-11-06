#!/bin/bash -l
#SBATCH --job-name=mosaic-power
#SBATCH --partition=q48
#SBATCH --mem-per-cpu=25G
#SBATCH --time=4:00:00
# Power spectrum of one field (argument 2: div or vort) of the DTFE mosaic.
# Submitted by run_mosaic.py with the run settings (run.env) as argument 1;
# the number of MPI ranks (--ntasks) is set by run_mosaic.py.
# Memory per rank is about 6x the rank's share of the grid (4*G^3/ranks bytes):
# 19 GB for a 3072^3 grid on 36 ranks, 13 GB for 4096^3 on 128 ranks
# (run_mosaic.py prints the estimate; change the rank number with --power-ranks).

source "$1"
field=$2
source "$SITE_SH"
setup_python

case "$field" in
    div)  suffix=a_velDiv ;;
    vort) suffix=a_velVort ;;
    *)    echo "unknown field '$field'"; exit 1 ;;
esac

echo "=== $(date): power spectrum of $field on $SLURM_NTASKS ranks"
$MPIRUN -n "$SLURM_NTASKS" python "$MOSAIC_DIR/mosaic_power.py" \
    "$DTFE_DIR/mosaic_r_0_0_0.$suffix" "$RUN_DIR/power_${field}.txt" --box "$BOX" --norm "$NORM"
status=$?
echo "=== $(date): finished with exit code $status"
exit $status
