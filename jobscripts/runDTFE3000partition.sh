#!/bin/bash -l
#SBATCH --job-name=Arrayjob
#SBATCH --partition=qfat
#SBATCH --mem=1300G
#SBATCH --cpus-per-task=32
#SBATCH --time=1:50:00
#SBATCH --export=NONE
##SBATCH --array=0-63%40
#SBATCH --array=58
# Load required modules if needed
ml load gcc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/external/lib

# Simulation config
# scale factors: 0.01, 0.02, 0.17, 0.20, 0.25, 0.33, 0.50, 1.00
a="1.00"
box="128"
particles="2048"
N_splits=4
# Compute ghost zone under the assumption that the hubble parameter is h=0.67556
padding=$(awk "BEGIN {print $box / $particles * 5 / 0.67556}")

extra_args=" --grid 512 512 512 --field divergence_a vorticity_a --periodic --samples 1000 --method 1"
extra_args+=" --paddingMpc $padding $padding $padding $padding $padding $padding"
#extra_args+=" --partition 2 2 2"
echo "Extra arguments:"
echo $extra_args

# Compute 3D indices (r_x, r_y, r_z) from SLURM_ARRAY_TASK_ID
task_id=${SLURM_ARRAY_TASK_ID}
r_z=$(( task_id % N_splits ))
r_y=$(( (task_id / N_splits) % N_splits ))
r_x=$(( task_id / (N_splits * N_splits) ))

input_snapshot="/home/bulow/codes/snapsplit/p${particles}_b${box}_pm_z99_a=${a}_fixed/a=${a}__r_${r_x}_${r_y}_${r_z}_snapshot.%i"
output_dir_and_suffix="output_test/p${particles}_fixed/p${particles}_b${box}_pm_a_${a}"
# You can now compute spatial region bounds here if needed
# For example, assuming simulation box is [0, 1] in each direction:
box_min=0.0
box_max=1.0
delta=$(echo "scale=10; ($box_max - $box_min)/$N_splits" | bc)

xmin=$(echo "$box_min + $r_x * $delta" | bc)
xmax=$(echo "$xmin + $delta" | bc)
ymin=$(echo "$box_min + $r_y * $delta" | bc)
ymax=$(echo "$ymin + $delta" | bc)
zmin=$(echo "$box_min + $r_z * $delta" | bc)
zmax=$(echo "$zmin + $delta" | bc)

echo "========= Job started at $(date) =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Region index: $r_x $r_y $r_z"
echo "Region bounds: x=[$xmin,$xmax], y=[$ymin,$ymax], z=[$zmin,$zmax]"

cd $SLURM_SUBMIT_DIR

# Output folder (optional: create subfolder per region?)
output_dir="${output_dir_and_suffix}_r_${r_x}_${r_y}_${r_z}"

./DTFE "$input_snapshot" "$output_dir" $extra_args --region $xmin $xmax $ymin $ymax $zmin $zmax

echo "========= Job finished at $(date) =========="

