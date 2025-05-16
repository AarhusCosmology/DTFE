#!/bin/bash -l
#SBATCH --job-name=Arrayjob
#SBATCH --partition=q40,q48
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=4:10:00
#SBATCH --export=NONE
###SBATCH --array=1
#SBATCH --array=0-48%10

# a = [0.01, 0.02, 0.17, 0.20, 0.25, 0.33, 0.50, 1.00]
#SCALE_FACTOR="1.00"
BOXSIZE="512"
PARTICLES="2048"
INPUT_SNAPSHOT="/home/camtgs/concept/output/p${PARTICLES}_b${BOXSIZE}_pm_z99_fixed/snapshot_a=${SCALE_FACTOR}/snapshot.$SLURM_ARRAY_TASK_ID"
OUTPUT_FOLDER="p${PARTICLES}_b${BOXSIZE}_pm_z99_a=${SCALE_FACTOR}_fixed"

N_SPLITS="4"
SNAP_SUFFIX="a=${SCALE_FACTOR}_"
GHOSTZONE_WIDTH="5"

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"
cd $SLURM_SUBMIT_DIR
python split_snapshot_in_regions.py $INPUT_SNAPSHOT $SLURM_SUBMIT_DIR/$OUTPUT_FOLDER $N_SPLITS $SNAP_SUFFIX $GHOSTZONE_WIDTH
echo "========= Job finished at `date` =========="

