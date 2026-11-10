#!/usr/bin/env python3
"""Power spectrum of a field that DTFE wrote as a mosaic of cubes.

The mosaic is N^3 binary files ..._r_{ix}_{iy}_{iz}.<suffix>, each holding a
cube^3 grid of float32 values (C order, x slowest) with 1 value per cell
(e.g. a_velDiv, a_den) or 3 (e.g. a_velVort, a_vel). The full (N*cube)^3 grid
is never held by one process: every MPI rank reads its pencil (a block in x
and y, all of z) from the cubes straight into its part of a distributed
real-to-complex FFT (mpi4py-fft). For vector fields the power of the three
components is summed.

The binning follows nbodykit's FFTPower(mode='1d') with its defaults, so the
output can be compared directly with spectra computed with nbodykit:
bins of width 2*pi/L starting at k=0, k is the mean |k| of the modes in a bin,
modes on the kz=0 and Nyquist planes count once and all others twice, and the
k=0 mode has zero power but is counted.

Units: k is in 1/box unit and P(k) in box unit^3 x field^2 / norm^2, i.e. h/Mpc
and (Mpc/h)^3 for a box in Mpc/h. With --hubble h (box in Mpc/h), lengths are
converted to Mpc at the end: k -> k h and P -> P h^(2m-3), where the field has
m powers of 1/length (m = 1 for the velocity divergence and vorticity, whose
DTFE values are in km/s per Mpc/h, and m = 0 for the velocity in km/s).
For the divergence and vorticity with --norm 3e5 this gives k in 1/Mpc and
P(k) in Mpc, i.e. k h and P/h of the spectrum in h/Mpc and Mpc/h.

Usage
    mpirun -n P python mosaic_power.py CUBE0 OUTPUT --box L [--hubble h] [options]

CUBE0 is any file name that contains r_0_0_0 (or {ix}, {iy}, {iz}
placeholders). The P ranks form a Px x Py process grid (default: as square as
possible); a rank can hold whole cubes or parts of them. Memory per rank is
about 6x its share of the grid (4*G^3/P bytes), e.g. 19 GB for 3072^3 on 36
ranks and 13 GB for 4096^3 on 128 ranks.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from mpi4py import MPI

# values per cell for DTFE's binary output files (file suffix -> components)
COMPONENTS = {'den': 1, 'a_den': 1, 'velDiv': 1, 'a_velDiv': 1,
              'vel': 3, 'a_vel': 3, 'velVort': 3, 'a_velVort': 3}
# powers of 1/length in the velocity fields (km/s and km/s per Mpc/h), for --hubble; the density is
# left out because its unit also depends on the mass unit
INVERSE_LENGTHS = {'velDiv': 1, 'a_velDiv': 1, 'velVort': 1, 'a_velVort': 1, 'vel': 0, 'a_vel': 0}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cube0', help="path of cube (0,0,0), containing 'r_0_0_0' or {ix}/{iy}/{iz} placeholders")
    p.add_argument('output', help='output text file')
    p.add_argument('--box', type=float, required=True,
                   help='side length of the full box; k is in units of 1/box unit (h/Mpc for Mpc/h) '
                        'unless --hubble is given')
    p.add_argument('--hubble', type=float,
                   help='h = H0/(100 km/s/Mpc). With the box in Mpc/h, write k in 1/Mpc and P(k) with lengths in '
                        'Mpc: k*h and P/h for the velocity divergence and vorticity (P/h^3 for the velocity; '
                        'not for the density). Default: no conversion')
    p.add_argument('--components', type=int, choices=[1, 3],
                   help='values per cell (default: from the DTFE file suffix, else from the file size)')
    p.add_argument('--norm', type=float, default=1.0,
                   help='divide P(k) by norm^2, e.g. 3e5 to express velocities in units of c (default 1)')
    p.add_argument('--grid', type=int, nargs=2, metavar=('PX', 'PY'),
                   help='MPI process grid; PX*PY must equal the number of ranks (default: as square as possible)')
    return p.parse_args(argv)


def file_pattern(cube0):
    """Turn the name of cube (0,0,0) into a format string with {ix}, {iy}, {iz}."""
    if '{ix}' in cube0:
        return cube0
    i = cube0.rfind('r_0_0_0')
    if i < 0:
        raise SystemExit(f"'{cube0}' contains neither 'r_0_0_0' nor {{ix}}/{{iy}}/{{iz}} placeholders")
    return cube0[:i] + 'r_{ix}_{iy}_{iz}' + cube0[i + len('r_0_0_0'):]


def inspect_mosaic(pattern, components=None):
    """Return (N, cube, components) and check that all N^3 files exist with the same size."""
    files = glob.glob(pattern.format(ix='*', iy='*', iz='*'))
    N = round(len(files) ** (1 / 3))
    if N < 1 or N**3 != len(files):
        raise SystemExit(f'found {len(files)} files matching {pattern.format(ix="*", iy="*", iz="*")}, '
                         f'which is not a cube number')
    expected = {pattern.format(ix=ix, iy=iy, iz=iz) for ix in range(N) for iy in range(N) for iz in range(N)}
    missing = sorted(expected - set(files))
    if missing:
        raise SystemExit(f'{len(missing)} cubes missing, e.g. {missing[0]}')
    sizes = {os.path.getsize(f) for f in files}
    if len(sizes) != 1:
        raise SystemExit(f'the cubes have different sizes: {sorted(sizes)[:5]} bytes')
    nvalues = sizes.pop() // 4
    if components is None:
        suffix = pattern.rsplit('.', 1)[-1]
        components = COMPONENTS.get(suffix)
    if components is None:  # fall back on the file size: n^3 or 3*n^3 values, never both
        components = 1 if round(nvalues ** (1 / 3)) ** 3 == nvalues else 3
    cube = round((nvalues / components) ** (1 / 3))
    if components * cube**3 != nvalues:
        raise SystemExit(f'a cube has {nvalues} float32 values, which is not {components} x n^3')
    return N, cube, components


def process_grid(nranks, G, grid=None):
    """Choose Px*Py = nranks, as square as possible (or check the given grid)."""
    if grid is not None:
        px, py = grid
        if px * py != nranks:
            raise SystemExit(f'--grid {px} {py}: PX*PY must equal the number of ranks, {nranks}')
    else:
        px = max(d for d in range(1, int(np.sqrt(nranks)) + 1) if nranks % d == 0)
        py = nranks // px
    if max(px, py) > G // 2:
        raise SystemExit(f'too many ranks ({px} x {py}) for a {G}^3 grid')
    return px, py


def kgrid(start, stop, G, L):
    """Wavenumbers of FFT indices start..stop-1 as pmesh computes them (float32, Nyquist negative).

    Copying pmesh's float32 arithmetic makes |k|^2 bit-identical to nbodykit's,
    so modes that sit exactly on a bin edge land in the same bin.
    """
    w = np.arange(start, stop).astype(np.float32)
    w[w >= G // 2] -= G
    w = w * np.float32(2 * np.pi / G)
    return (w * np.float32(G)) / np.float32(L)


def bin_power(field_hat, kslice, G, L, kedges, comm):
    """Accumulate nbodykit-style 1d binning of |field_hat|^2 * L^3 on this rank."""
    nbins = len(kedges) + 1
    x2edges = kedges**2
    kx = kgrid(kslice[0].start, kslice[0].stop, G, L)
    ky = kgrid(kslice[1].start, kslice[1].stop, G, L)
    kz = kgrid(kslice[2].start, kslice[2].stop, G, L)
    # square whole arrays: numpy's float32 *scalar* x**2 is not always correctly rounded
    kx2 = kx**2
    ky2 = (ky**2)[:, None]
    kz2 = (kz**2)[None, :]
    # half-complex axis: modes with 0 < kz < Nyquist stand for themselves and their conjugate
    weight = np.where(kz > 0, np.float32(2), np.float32(1))[None, :] * np.ones_like(ky2)
    ksum = np.zeros(nbins)
    psum = np.zeros(nbins)
    nsum = np.zeros(nbins)
    volume = float(L) ** 3
    has_dc = kslice[0].start == 0 and kslice[1].start == 0 and kslice[2].start == 0
    for i in range(field_hat.shape[0]):
        k2 = (kx2[i] + ky2) + kz2                   # same float32 order as nbodykit's slab.norm2()
        dig = np.digitize(k2.ravel(), x2edges)
        p = field_hat[i].real.astype(np.float64)**2 + field_hat[i].imag.astype(np.float64)**2
        if has_dc and i == 0:
            p[0, 0] = 0.0                           # nbodykit clears the k=0 mode
        nsum += np.bincount(dig, weights=weight.ravel(), minlength=nbins)
        ksum += np.bincount(dig, weights=(np.sqrt(k2) * weight).ravel(), minlength=nbins)
        psum += np.bincount(dig, weights=(p * weight).ravel(), minlength=nbins) * volume
    return ksum, psum, nsum


def max_rss_mb():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == 'darwin' else rss / 2**10
    except Exception:
        return float('nan')


def main(argv=None):
    from mpi4py_fft import PFFT

    args = parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    t0 = time.time()

    pattern = file_pattern(args.cube0)
    inverse_lengths = INVERSE_LENGTHS.get(pattern.rsplit('.', 1)[-1])
    if args.hubble is not None:
        if args.hubble <= 0:
            raise SystemExit(f'--hubble must be positive, got {args.hubble}')
        if inverse_lengths is None:
            raise SystemExit(f"--hubble: no conversion to Mpc known for field '{pattern.rsplit('.', 1)[-1]}' "
                             f"(only for {', '.join(INVERSE_LENGTHS)})")
    info = None
    if rank == 0:
        try:
            info = inspect_mosaic(pattern, args.components)
        except SystemExit as e:
            info = str(e)
    info = comm.bcast(info, root=0)
    if isinstance(info, str):
        if rank == 0:
            print('ERROR:', info, flush=True)
        comm.Abort(1)
    N, cube, ncomp = info
    G = N * cube
    px, py = process_grid(comm.Get_size(), G, args.grid)

    fft = PFFT(comm, (G, G, G), axes=(0, 1, 2), dtype=np.float32, grid=(px, py, 1))
    u = fft.forward.input_array                     # fill the FFT's own input buffer: saves a copy of the grid
    xs, ys, zs = fft.local_slice(False)
    assert u.shape == tuple(s.stop - s.start for s in (xs, ys, zs))
    if (zs.start, zs.stop) != (0, G):
        print(f'ERROR rank {rank}: expected all of z locally, got {xs}, {ys}, {zs}', flush=True)
        comm.Abort(1)
    kslice = fft.local_slice(True)

    dk = 2 * np.pi / args.box
    kedges = np.arange(0., np.pi * G / args.box + dk / 2, dk)   # nbodykit's default edges
    ksum = psum = nsum = 0.0

    if rank == 0:
        print(f'{pattern}\n  {N}^3 cubes of {cube}^3 cells -> {G}^3 grid, {ncomp} component(s), '
              f'box {args.box}, process grid {px} x {py}', flush=True)

    for c in range(ncomp):
        t = time.time()
        # copy the part of every cube that overlaps this rank's block [xs) x [ys) x all z
        for ix in range(xs.start // cube, (xs.stop - 1) // cube + 1):
            x0, x1 = max(xs.start, ix * cube), min(xs.stop, (ix + 1) * cube)
            for iy in range(ys.start // cube, (ys.stop - 1) // cube + 1):
                y0, y1 = max(ys.start, iy * cube), min(ys.stop, (iy + 1) * cube)
                for iz in range(N):
                    data = np.memmap(pattern.format(ix=ix, iy=iy, iz=iz), dtype=np.float32, mode='r',
                                     shape=(cube, cube, cube, ncomp))
                    u[x0 - xs.start:x1 - xs.start, y0 - ys.start:y1 - ys.start, iz * cube:(iz + 1) * cube] = \
                        data[x0 - ix * cube:x1 - ix * cube, y0 - iy * cube:y1 - iy * cube, :, c]
                    del data
        comm.Barrier()
        t_read = time.time() - t
        t = time.time()
        u_hat = fft.forward(normalize=True)         # (1/G^3) sum_x f(x) exp(-ikx), as pmesh
        t_fft = time.time() - t
        t = time.time()
        k_c, p_c, n_c = bin_power(u_hat, kslice, G, args.box, kedges, comm)
        ksum, psum, nsum = k_c, psum + p_c, n_c
        comm.Barrier()
        if rank == 0:
            print(f'  component {c}: read {t_read:.1f} s, FFT {t_fft:.1f} s, binning {time.time() - t:.1f} s',
                  flush=True)

    ksum = comm.allreduce(ksum)
    psum = comm.allreduce(psum)
    nsum = comm.allreduce(nsum)
    rss = comm.reduce(max_rss_mb(), op=MPI.MAX, root=0)

    if rank == 0:
        sl = slice(1, len(kedges))                  # drop the under- and overflow bins
        with np.errstate(invalid='ignore', divide='ignore'):
            k = ksum[sl] / nsum[sl]
            power = psum[sl] / nsum[sl] / args.norm**2
        header = (f'power spectrum of {pattern}\n'
                  f'{N}^3 cubes of {cube}^3 cells, grid {G}^3, box {args.box}, {ncomp} component(s) summed, '
                  f'P divided by norm^2 with norm = {args.norm}\n')
        if args.hubble is None:
            header += 'k [1/box unit]   P(k) [box unit^3 * field^2 / norm^2]   modes'
        else:
            p = 2 * inverse_lengths - 3
            k = k * args.hubble
            power = power * args.hubble**p
            mpc = 'Mpc' if p == -1 else f'Mpc^{-p}'
            header += (f'box in Mpc/h converted to Mpc with h = {args.hubble}: k -> k h, P -> P h^{p}\n'
                       f'k [1/Mpc]   P(k) [{mpc} (km/s)^2 / norm^2]   modes')
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savetxt(args.output, np.c_[k, power, nsum[sl]], header=header)
        print(f'Wrote {args.output} ({len(k)} bins) in {time.time() - t0:.1f} s; '
              f'max memory per rank {rss / 1024:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
