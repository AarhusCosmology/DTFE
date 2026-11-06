#!/usr/bin/env python3
"""Write a small multi-file Gadget snapshot for testing the mosaic pipeline.

    python make_snapshot.py OUTDIR [--particles n] [--box L] [--files F] [--seed S]

A lattice of n^3 equal-mass particles is displaced along the gradient of a
Gaussian random potential (rms displacement ~0.3 particle separations), with
velocities proportional to the displacement plus a small rotational flow, so
both the velocity divergence and the vorticity are non-zero. Positions are in
the same units as L (default 64000, i.e. kpc/h for a 64 Mpc/h box). The
particles are distributed over F files OUTDIR/snapshot.0 ... by x-slabs of
the lattice, so most regions of a split receive particles from only some files.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import g3read as g3  # noqa: E402


def displacement(n, box, rng):
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    kvec = np.meshgrid(k1, k1, kz, indexing='ij')
    k2 = kvec[0]**2 + kvec[1]**2 + kvec[2]**2
    k2[0, 0, 0] = 1.0
    kmax = 2 * np.pi / box * n / 4
    phi_k = np.fft.rfftn(rng.standard_normal((n, n, n))) * np.exp(-k2 / kmax**2) / k2
    phi_k[0, 0, 0] = 0
    psi = np.stack([np.fft.irfftn(-1j * kk * phi_k, s=(n, n, n), axes=(0, 1, 2)) for kk in kvec])
    return psi * (0.3 * box / n / psi.std())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outdir')
    p.add_argument('--particles', type=int, default=64)
    p.add_argument('--box', type=float, default=64000.0)
    p.add_argument('--files', type=int, default=4)
    p.add_argument('--seed', type=int, default=3)
    a = p.parse_args()

    n, L = a.particles, a.box
    rng = np.random.default_rng(a.seed)
    q = (np.indices((n, n, n)) + 0.5) * (L / n)
    psi = displacement(n, L, rng)
    pos = np.mod(q + psi, L).reshape(3, -1).T.astype(np.float32)
    vel = 100.0 * psi / (L / n)                                   # ~30 km/s rms
    vel[0] += 20.0 * np.sin(2 * np.pi * q[1] / L)                 # rotational part
    vel[1] += 20.0 * np.sin(2 * np.pi * q[2] / L)
    vel = vel.reshape(3, -1).T.astype(np.float32)

    os.makedirs(a.outdir, exist_ok=True)
    slab = np.repeat(np.arange(n) * a.files // n, n * n)           # file index from the lattice x index
    for i in range(a.files):
        sel = slab == i
        npart = [0, int(sel.sum()), 0, 0, 0, 0]
        header = g3.GadgetHeader(np.array(npart), np.array([0, 1.0, 0, 0, 0, 0]), 1.0, 0.0,
                                 L, 0.3, 0.7, 0.7, num_files=a.files)
        header.npartTotal = np.array([0, n**3, 0, 0, 0, 0], dtype=np.int32)
        fn = os.path.join(a.outdir, f'snapshot.{i}')
        f = g3.GadgetWriteFile(fn, npart, {}, header)
        f.write_header(f.header)
        f.add_file_block('POS ', npart[1] * 12, partlen=12)
        f.write_block('POS ', -1, pos[sel])
        f.add_file_block('VEL ', npart[1] * 12, partlen=12)
        f.write_block('VEL ', -1, vel[sel])
    print(f'wrote {n}^3 particles in box {L} to {a.files} files in {a.outdir}')


if __name__ == '__main__':
    main()
