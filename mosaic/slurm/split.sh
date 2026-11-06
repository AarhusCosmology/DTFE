#!/bin/bash -l
#SBATCH --job-name=mosaic-split
#SBATCH --partition=q40,q48
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
# One array task per snapshot file: cut SNAPSHOT.<task id> into NREG^3 regions.
# Submitted by run_mosaic.py with the run settings (run.env) as argument.
# Memory: about 3x the size of one snapshot file (16 GB for 5 GB files).

source "$1"
source "$SITE_SH"
setup_python

echo "=== $(date): split ${SNAPSHOT}.${SLURM_ARRAY_TASK_ID} into ${NREG}^3 regions -> $SPLIT_DIR"
python "$MOSAIC_DIR/split_snapshot_in_regions.py" "${SNAPSHOT}.${SLURM_ARRAY_TASK_ID}" "$SPLIT_DIR" \
    "$NREG" sub "$GHOST" --particles "$PARTICLES"
status=$?
echo "=== $(date): finished with exit code $status"
exit $status
