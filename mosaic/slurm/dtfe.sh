#!/bin/bash -l
#SBATCH --job-name=mosaic-dtfe
#SBATCH --partition=qany
#SBATCH --mem=350G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
# One array task per region: DTFE on the region's sub-snapshot, giving CUBE^3
# grids in DTFE_DIR/mosaic_r_<ix>_<iy>_<iz>.<field suffix>.
# Submitted by run_mosaic.py with the run settings (run.env) as argument.
# A 512^3 region of 512^3 particles plus ghost zone took 9-14 min on 16 cores;
# most regions fit in 180 GB, the densest need more.

source "$1"
source "$SITE_SH"
setup_dtfe

t=$SLURM_ARRAY_TASK_ID
iz=$(( t % NREG ))
iy=$(( (t / NREG) % NREG ))
ix=$(( t / (NREG * NREG) ))
region() { awk -v i="$1" -v n="$NREG" 'BEGIN { printf "%.10f %.10f", i / n, (i + 1) / n }'; }
input="${SPLIT_DIR:?}/sub_r_${ix}_${iy}_${iz}_${SNAPSHOT_NAME}"

echo "=== $(date): DTFE on region $ix $iy $iz (task $t)"
"$DTFE_BIN" "${input}.%i" "$DTFE_DIR/mosaic_r_${ix}_${iy}_${iz}" \
    --grid "$CUBE" "$CUBE" "$CUBE" --field $DTFE_FIELDS --periodic $DTFE_OPTIONS \
    --paddingMpc $PADDING $PADDING $PADDING $PADDING $PADDING $PADDING \
    --region $(region $ix) $(region $iy) $(region $iz) --skipMissingFiles
status=$?
if [ $status -eq 0 ] && [ "$KEEP_SPLIT" != 1 ]; then
    rm -f "${input}."*
fi
echo "=== $(date): finished with exit code $status"
exit $status
