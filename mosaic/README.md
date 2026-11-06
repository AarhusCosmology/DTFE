# Mosaic: DTFE fields and power spectra of very large snapshots

DTFE reads the whole snapshot into memory and runs on a single node, which rules out
snapshots with many billions of particles. The scripts in this folder get around that by
cutting the box into N³ regions, running DTFE on each region separately, and computing
the power spectrum of the resulting mosaic of grids with a distributed FFT, without ever
holding the full grid on one node.

```
snapshot.0 … snapshot.M          (Gadget format, multiple files)
   │  1. split   one Slurm array task per snapshot file          split_snapshot_in_regions.py
   ▼
sub_r_i_j_k_snapshot.0 … .M      N³ regions, each with a ghost zone of 5 particle separations
   │  2. dtfe    one Slurm array task per region                  DTFE --region … --skipMissingFiles
   ▼
mosaic_r_i_j_k.a_velDiv, …       N³ cubes of 512³ cells (DTFE binary output)
   │  3. power   one MPI job per field                            mosaic_power.py
   ▼
power_div.txt, power_vort.txt    P(k) of the full (512 N)³ grid
```

`run_mosaic.py` submits the three steps as Slurm jobs that wait for each other.
It can also run them one after another on a single machine (`--local`), which is
useful for small snapshots and for testing.

## Setup

1. **DTFE** from this repository. The `--skipMissingFiles` option is needed because a
   region gets no file from snapshot files that have no particles near it.
2. **Python** with numpy, numba, mpi4py and mpi4py-fft:
   ```
   conda env create -f mosaic/environment.yml
   ```
   This is enough on a workstation. For jobs spanning several nodes, mpi4py has to use
   the cluster's MPI. Either replace `openmpi` in `environment.yml` by conda-forge's
   placeholder for an external MPI (`openmpi=4.1.*=external_*`, with the version of
   the cluster's Open MPI), or build mpi4py yourself with the cluster's MPI loaded
   (`pip install --no-binary mpi4py mpi4py`). Check with
   `srun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.size)"`, which
   should print 2 twice.
3. **Cluster settings**: edit `slurm/site.sh` (modules, Python environment, MPI launcher)
   and the `#SBATCH` lines of `slurm/split.sh`, `slurm/dtfe.sh` and `slurm/power.sh`
   (partitions, memory, time). The defaults are for Grendel in Aarhus.
4. Run the tests: `bash mosaic/tests/run_tests.sh` (about a minute, see below).

## Usage

```
python mosaic/run_mosaic.py SNAPSHOT RUNDIR --box L --particles NP [--fields div vort] [--norm 3e5]
```

For example, the velocity divergence and vorticity of a CONCEPT simulation with 4096³
particles in a 2048 Mpc/h box:

```
python mosaic/run_mosaic.py /home/camtgs/concept/output/p4096_b2048_pm_z99_fixed/snapshot_a=1.00/snapshot \
       /scratch/mosaic/p4096_b2048_a1.00 --box 2048 --particles 4096 --norm 3e5
```

Everything for the run ends up in `RUNDIR`: the settings (`run.env`), the sub-snapshots
(`split/`), the DTFE output (`dtfe/`), the spectra (`power_div.txt`, `power_vort.txt`),
the Slurm logs and `launcher.log`. Useful options:

* `--dry-run` prints the settings and the `sbatch` commands without doing anything.
* `--start-at dtfe` or `--start-at power` redoes the later steps. With `power` it first
  checks that all DTFE cubes are there.
* `--power-ranks P` sets the number of MPI ranks. The default is N², doubled until the
  estimated memory per rank fits in `--power-mem` (24 GB).
* `--keep-split` keeps the sub-snapshots; by default each region's sub-snapshots are
  deleted as soon as its DTFE run has succeeded.
* `--local --site slurm/site_local.sh` runs everything on this machine.

The grid has one cell per particle per dimension (`--grid` to change) and each region is
512³ cells (`--cube`), so N = particles/512. If a step fails for some array tasks, the
later steps never start (Slurm reports `DependencyNeverSatisfied`). Rerun the failed tasks
by hand, e.g. `sbatch --array=17,42 --export=NONE mosaic/slurm/dtfe.sh RUNDIR/run.env`,
and then continue with `--start-at power`. Don't rerun the whole DTFE step, because
the sub-snapshots of the regions that succeeded have already been deleted.

`mosaic_power.py` can also be used on its own on any mosaic of DTFE binary files:
```
mpirun -n 16 python mosaic/mosaic_power.py dtfe/mosaic_r_0_0_0.a_velDiv power_div.txt --box 1024
```

## Output

`power_<field>.txt` has three columns: k [h/Mpc], P(k) and the number of modes in the bin.
For the vorticity, P is the sum over its three components. P is divided by `norm`²; with
velocities in km/s, `--norm 3e5` expresses them in units of c. The binning is the same as
nbodykit's `FFTPower(mode='1d')` with default settings, and the results agree with
nbodykit to float32 precision. Bins have width 2π/L and start at k = 0, k is the mean |k|
of the modes in a bin, and the k = 0 mode is counted in the first bin with zero power.

## Resources

Measured on Grendel for 3072³ particles in a 1024 Mpc/h box (6³ regions, 163 snapshot
files of 5 GB each):

| step  | per task                                   | resources            |
|-------|--------------------------------------------|----------------------|
| split | 5–34 min per snapshot file (median 7)      | 1 core, 16 GB        |
| dtfe  | 9–14 min per region                        | 16 cores, ≤ 350 GB   |

The power spectrum needs about 6 × 4 G³ / P bytes per rank for a G³ grid on P ranks,
e.g. 19 GB for 3072³ on 36 ranks and 13 GB for 4096³ on 128 ranks; `run_mosaic.py`
prints the estimate. Disk space: the sub-snapshots take about 25 bytes per particle
(1.75 TB for 4096³ particles) until DTFE has processed them, and the DTFE output takes
4 bytes per cell and component (0.27 TB for the divergence and 0.82 TB for the
vorticity of a 4096³ grid).

## Assumptions

* Periodic box, Gadget binary snapshot in multiple files, dark-matter particles of
  type 1 with equal masses given in the header. Only positions and velocities are
  copied to the sub-snapshots.
* Positions in kpc/h, so that DTFE's default 1 Mpc = 1000 position units holds, and
  `--box` in Mpc/h.
* The ghost zone (`--ghost`, default 5 mean particle separations) has to be wide enough
  for the Delaunay tessellation inside a region to be the same as in the full box. DTFE's
  `--paddingMpc` is set to the same width.

## Tests and validation

`tests/run_tests.sh` checks:

* the power spectrum of a synthetic Gaussian mosaic, with 1 and 3 components, is the
  same on 1, 4 and 6 ranks, agrees with a single-process numpy FFT and recovers the
  input power spectrum;
* the whole chain (split, DTFE, power) on a small snapshot with 64³ particles in 2³
  regions, run through `run_mosaic.py --local`, against DTFE on the whole box. The
  spectra agree to 0.3%. The remaining difference comes from DTFE's Monte Carlo
  sampling, since the sample points depend on the order of the particles. The
  non-averaged divergence of the mosaic is identical to that of the whole box in every
  cell.

In addition, `mosaic_power.py` was compared with nbodykit 0.3.15 (`FFTPower` on the full
field, and the earlier nbodykit/pmesh-based mosaic script). The mode counts and k values
are identical in every bin, and P agrees to better than 4e-7, for several grid sizes,
box sizes and process grids.

## Files

* `run_mosaic.py`: submits (or runs) the whole chain.
* `split_snapshot_in_regions.py`: cuts one snapshot file into N³ sub-snapshots.
* `mosaic_power.py`: power spectrum of a mosaic of DTFE cubes (MPI, mpi4py-fft).
* `g3read.py`: Gadget reader/writer by Antonio Ragagnin (GPL-3.0), based on pynbody.
* `slurm/`: job scripts and site setup.
* `tests/`: synthetic data generators and the test script.
