#!/usr/bin/env python3
"""Run the split -> DTFE -> power spectrum chain for a large Gadget snapshot.

    python run_mosaic.py SNAPSHOT RUNDIR --box L --particles NP [options]

SNAPSHOT is the multi-file snapshot without its file number: the files are
SNAPSHOT.0, SNAPSHOT.1, ... RUNDIR is created and receives everything for this
run: run.env (the settings, read by the job scripts), split/ (sub-snapshots),
dtfe/ (DTFE output, N^3 cubes per field), power_<field>.txt, the Slurm logs and
launcher.log.

  1. split   (Slurm array, one task per snapshot file, slurm/split.sh):
             cut each file into N^3 regions with a ghost zone of GHOST mean
             particle separations; N = grid/cube.
  2. dtfe    (Slurm array, one task per region, slurm/dtfe.sh):
             DTFE on each region's sub-snapshot -> cube^3 grids. The region's
             sub-snapshots are deleted afterwards unless --keep-split.
  3. power   (one MPI job per field, slurm/power.sh):
             P(k) of the full grid with mosaic_power.py, on N^2 ranks or more if
             they would need more than --power-mem GB each.

Each step waits for the previous one (afterok). With --start-at you can redo
later steps, and with --local the same job scripts run one after another on
this machine instead of through Slurm. Site-specific settings (modules, Python
environment, MPI launcher) are in slurm/site.sh; partitions, memory and time
limits are the #SBATCH lines of the job scripts.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

MOSAIC_DIR = os.path.dirname(os.path.abspath(__file__))
FIELDS = {  # field name -> (DTFE --field option, DTFE output suffix, values per cell)
    'div': ('divergence_a', 'a_velDiv', 1),
    'vort': ('vorticity_a', 'a_velVort', 3),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snapshot', help='snapshot path without the file number, e.g. .../snapshot_a=1.00/snapshot')
    p.add_argument('rundir', help='directory for this run (created)')
    p.add_argument('--box', type=float, required=True,
                   help='box size in units of 1000 x the position unit, i.e. Mpc/h for positions in kpc/h '
                        '(DTFE assumes 1 Mpc = 1000 position units)')
    p.add_argument('--particles', type=int, required=True, help='particles per dimension in the snapshot')
    p.add_argument('--grid', type=int, help='grid cells per dimension of the full box (default: particles)')
    p.add_argument('--cube', type=int, default=512, help='grid cells per dimension of one region (default 512)')
    p.add_argument('--ghost', type=float, default=5.0,
                   help='ghost zone around each region in mean particle separations (default 5)')
    p.add_argument('--fields', nargs='+', choices=sorted(FIELDS), default=['div', 'vort'],
                   help='fields for DTFE and the power spectrum (default: div vort)')
    p.add_argument('--norm', type=float, default=1.0,
                   help='divide P(k) by norm^2, e.g. 3e5 for velocities in units of c (default 1)')
    p.add_argument('--dtfe', default=os.path.join(MOSAIC_DIR, '..', 'DTFE'),
                   help='DTFE executable (default: the one built in the repository)')
    p.add_argument('--dtfe-options', default='--samples 1000 --method 1',
                   help="extra DTFE options (default: '--samples 1000 --method 1')")
    p.add_argument('--power-ranks', type=int,
                   help='MPI ranks for the power spectrum (default: N^2, doubled until it fits in --power-mem)')
    p.add_argument('--power-mem', type=float, default=24.0,
                   help='memory per rank in GB available to the power spectrum; keep it below '
                        '--mem-per-cpu in slurm/power.sh (default 24)')
    p.add_argument('--keep-split', action='store_true', help='keep the sub-snapshots after DTFE')
    p.add_argument('--start-at', choices=['split', 'dtfe', 'power'], default='split',
                   help='skip earlier steps whose output already exists (default: split)')
    p.add_argument('--array-limit', type=int, default=32, help='max simultaneous array tasks (default 32)')
    p.add_argument('--site', default=os.path.join(MOSAIC_DIR, 'slurm', 'site.sh'),
                   help='site setup script sourced by the jobs (default slurm/site.sh)')
    p.add_argument('--local', action='store_true', help='run the job scripts here, one after another, without Slurm')
    p.add_argument('--dry-run', action='store_true', help='print what would be done without doing it')
    return p.parse_args(argv)


class Launcher:
    def __init__(self, args, logfile):
        self.args = args
        self.logfile = logfile
        self.fake_id = 0

    def log(self, *msg):
        text = ' '.join(str(m) for m in msg)
        print(text, flush=True)
        if not self.args.dry_run:
            with open(self.logfile, 'a') as f:
                f.write(text + '\n')

    def submit(self, script, script_args, ntasks=None, array=None, dependency=None):
        """Submit to Slurm (or run locally) and return the job id."""
        a = self.args
        script = os.path.join(MOSAIC_DIR, 'slurm', script)
        if a.local:
            return self.run_local(script, script_args, ntasks, array)
        cmd = ['sbatch', '--parsable', '--export=NONE']
        if array is not None:
            cmd.append(f'--array=0-{array - 1}%{a.array_limit}')
        if ntasks is not None:
            cmd.append(f'--ntasks={ntasks}')
        if dependency:
            cmd.append(f'--dependency=afterok:{dependency}')
        cmd += [script] + script_args
        self.log('Submitting:', ' '.join(shlex.quote(c) for c in cmd))
        if a.dry_run:
            self.fake_id += 1
            return f'DRYRUN{self.fake_id}'
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        job_id = out.strip().split(';')[0]
        self.log(f'  -> job {job_id}')
        return job_id

    def run_local(self, script, script_args, ntasks, array):
        tasks = range(array) if array is not None else [None]
        for t in tasks:
            env = dict(os.environ)
            if t is not None:
                env['SLURM_ARRAY_TASK_ID'] = str(t)
            if ntasks is not None:
                env['SLURM_NTASKS'] = str(ntasks)
            cmd = ['bash', script] + script_args
            self.log('Running:', ' '.join(shlex.quote(c) for c in cmd), '' if t is None else f'(task {t})')
            if self.args.dry_run:
                continue
            name = os.path.basename(script).replace('.sh', '')
            logname = '_'.join([f'local-{name}'] + script_args[1:] + ([] if t is None else [str(t)])) + '.out'
            with open(logname, 'w') as f:
                r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                sys.exit(f'{script} failed (exit {r.returncode}), see {os.path.abspath(logname)}')
        return 'local'


def count_snapshot_files(snapshot):
    folder, base = os.path.split(os.path.abspath(snapshot))
    if not os.path.isdir(folder):
        return 0
    pattern = re.compile(re.escape(base) + r'\.(\d+)$')
    numbers = sorted(int(m.group(1)) for m in map(pattern.match, os.listdir(folder)) if m)
    if numbers and numbers != list(range(len(numbers))):
        sys.exit(f'snapshot files {snapshot}.N are not numbered 0..{len(numbers) - 1}')
    return len(numbers)


def missing_cubes(dtfe_dir, nreg, cube, field):
    _, suffix, ncomp = FIELDS[field]
    size = ncomp * cube**3 * 4
    bad = []
    for ix in range(nreg):
        for iy in range(nreg):
            for iz in range(nreg):
                fn = os.path.join(dtfe_dir, f'mosaic_r_{ix}_{iy}_{iz}.{suffix}')
                if not os.path.isfile(fn) or os.path.getsize(fn) != size:
                    bad.append(fn)
    return bad


def main(argv=None):
    a = parse_args(argv)
    a.snapshot = os.path.abspath(a.snapshot)  # before changing into the run directory
    grid = a.grid or a.particles
    if grid % a.cube:
        sys.exit(f'grid {grid} is not a multiple of the region size --cube {a.cube}')
    nreg = grid // a.cube
    rundir = os.path.abspath(a.rundir)
    split_dir = os.path.join(rundir, 'split')
    dtfe_dir = os.path.join(rundir, 'dtfe')
    dtfe_bin = os.path.abspath(a.dtfe)
    def power_memory(ranks):                 # GB per rank, measured: ~6x the rank's share of the grid
        return 6 * 4 * grid**3 / ranks / 1e9 + 0.2
    ranks = a.power_ranks or nreg**2
    while not a.power_ranks and power_memory(ranks) > a.power_mem:
        ranks *= 2
    padding = a.ghost * a.box / a.particles   # ghost zone in DTFE's 'Mpc' (1000 position units)

    settings = {
        'MOSAIC_DIR': MOSAIC_DIR,
        'SITE_SH': os.path.abspath(a.site),
        'RUN_DIR': rundir,
        'SNAPSHOT': os.path.abspath(a.snapshot),
        'SNAPSHOT_NAME': os.path.basename(a.snapshot),
        'SPLIT_DIR': split_dir,
        'DTFE_DIR': dtfe_dir,
        'DTFE_BIN': dtfe_bin,
        'DTFE_FIELDS': ' '.join(FIELDS[f][0] for f in a.fields),
        'DTFE_OPTIONS': a.dtfe_options,
        'NREG': nreg,
        'CUBE': a.cube,
        'PARTICLES': a.particles,
        'BOX': a.box,
        'GHOST': a.ghost,
        'PADDING': f'{padding:.8g}',
        'KEEP_SPLIT': int(a.keep_split),
        'NORM': a.norm,
    }
    env_file = os.path.join(rundir, 'run.env')
    env_text = ''.join(f'{k}={shlex.quote(str(v))}\n' for k, v in settings.items())

    if not a.dry_run:
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(dtfe_dir, exist_ok=True)
        with open(env_file, 'w') as f:
            f.write(env_text)
        os.chdir(rundir)                     # Slurm logs end up here
    L = Launcher(a, os.path.join(rundir, 'launcher.log'))
    L.log(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")}: {" ".join(shlex.quote(x) for x in sys.argv)}')
    L.log(f'grid {grid}^3 = {nreg}^3 regions of {a.cube}^3, ghost zone {padding:.6g} (box units), '
          f'fields {a.fields}, start at {a.start_at}' + (', local' if a.local else ''))
    L.log(f'power spectrum on {ranks} ranks needs about {power_memory(ranks):.1f} GB per rank '
          f'(change with --power-ranks)')
    if power_memory(ranks) > a.power_mem:
        L.log(f'WARNING: that is more than --power-mem {a.power_mem} GB')
    if a.dry_run:
        L.log(f'--- {env_file} would contain:\n{env_text}---')

    if a.start_at in ('split', 'dtfe') and not os.access(dtfe_bin, os.X_OK):
        L.log(f'WARNING: DTFE executable {dtfe_bin} not found')

    job = None
    if a.start_at == 'split':
        nfiles = count_snapshot_files(a.snapshot)
        if nfiles == 0:
            if not a.dry_run:
                sys.exit(f'no files {a.snapshot}.0, {a.snapshot}.1, ... found')
            L.log(f'WARNING: no files {a.snapshot}.N found; using 1 for the dry run')
            nfiles = 1
        L.log(f'{nfiles} snapshot files')
        job = L.submit('split.sh', [env_file], array=nfiles)

    if a.start_at in ('split', 'dtfe'):
        job = L.submit('dtfe.sh', [env_file], array=nreg**3, dependency=job)

    for field in a.fields:
        if a.start_at == 'power':
            bad = missing_cubes(dtfe_dir, nreg, a.cube, field)
            if bad:
                L.log(f'{len(bad)} of {nreg**3} DTFE {field} cubes are missing or have the wrong size, e.g. {bad[0]}')
                if not a.dry_run:
                    sys.exit('Not starting the power spectrum. Redo DTFE with --start-at dtfe (or split).')
            else:
                L.log(f'all {nreg**3} DTFE {field} cubes present in {dtfe_dir}')
        L.submit('power.sh', [env_file, field], ntasks=ranks, dependency=job)

    if a.local and not a.dry_run:
        L.log('Done: ' + ', '.join(os.path.join(rundir, f'power_{f}.txt') for f in a.fields))


if __name__ == '__main__':
    main()
