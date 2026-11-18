#!/usr/bin/env python3
"""Write a Gaussian random field with a known power spectrum as a DTFE-style mosaic.

    python make_mosaic.py OUTDIR [--cubes N] [--cube C] [--box L] [--components 1|3] [--seed S] [--window]

Writes OUTDIR/test_r_{ix}_{iy}_{iz}.a_velDiv (1 component) or .a_velVort
(3 components): N^3 cubes of C^3 float32 values, C order with x slowest and the
components of a cell next to each other, exactly like DTFE's binary output. Also
writes OUTDIR/test_full.npy with the full field, shape (components, G, G, G).

Each component has the power spectrum P(k) = model_power(k, L), with the
convention P(k) = L^3 <|f_k|^2> and f_k = G^-3 sum_x f(x) exp(-ikx).

With --window, the modes of the written field are multiplied by the window of an
average over one grid cell of side d = L/G, W(k) = prod_i sin(k_i d/2)/(k_i d/2),
as for DTFE's cell-averaged (a_*) fields without aliasing. The field before the
window is then written to OUTDIR/test_nowindow.npy.
"""
import argparse
import os

import numpy as np


def model_power(k, box):
    """Power law with a cut-off, in the units of the box."""
    k0 = 2 * np.pi / box
    with np.errstate(divide='ignore'):
        p = np.where(k > 0, box**3 * (k / k0) ** -1.5 * np.exp(-(k / (40 * k0)) ** 2), 0.0)
    return p


def cell_average_window(k, d):
    """sin(k d/2) / (k d/2) along one axis."""
    x = k * d / 2
    return np.where(x == 0, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))


def gaussian_field(G, box, rng, window=False):
    """Return the field, and with window=True also the field before the window was applied."""
    k1 = 2 * np.pi * np.fft.fftfreq(G, d=box / G)
    kz = 2 * np.pi * np.fft.rfftfreq(G, d=box / G)
    k = np.sqrt(k1[:, None, None]**2 + k1[None, :, None]**2 + kz[None, None, :]**2)
    white = np.fft.rfftn(rng.standard_normal((G, G, G))) / G**3        # <|w_k|^2> = G^-3
    f_k = white * np.sqrt(model_power(k, box) * G**3 / box**3)
    to_real = lambda fk: (np.fft.irfftn(fk, s=(G, G, G), axes=(0, 1, 2)) * G**3).astype(np.float32)
    if not window:
        return to_real(f_k)
    w1, wz = cell_average_window(k1, box / G), cell_average_window(kz, box / G)
    return to_real(f_k * w1[:, None, None] * w1[None, :, None] * wz[None, None, :]), to_real(f_k)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outdir')
    p.add_argument('--cubes', type=int, default=4)
    p.add_argument('--cube', type=int, default=16)
    p.add_argument('--box', type=float, default=100.0)
    p.add_argument('--components', type=int, choices=[1, 3], default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--window', action='store_true',
                   help='multiply the modes by the window of a one-cell average (see above)')
    a = p.parse_args()

    G = a.cubes * a.cube
    rng = np.random.default_rng(a.seed)
    fields = [gaussian_field(G, a.box, rng, a.window) for _ in range(a.components)]
    os.makedirs(a.outdir, exist_ok=True)
    if a.window:
        np.save(os.path.join(a.outdir, 'test_nowindow.npy'), np.stack([f[1] for f in fields]))
        fields = [f[0] for f in fields]
    full = np.stack(fields)
    np.save(os.path.join(a.outdir, 'test_full.npy'), full)
    suffix = 'a_velDiv' if a.components == 1 else 'a_velVort'
    C = a.cube
    for ix in range(a.cubes):
        for iy in range(a.cubes):
            for iz in range(a.cubes):
                block = full[:, ix*C:(ix+1)*C, iy*C:(iy+1)*C, iz*C:(iz+1)*C]
                np.ascontiguousarray(np.moveaxis(block, 0, -1)).tofile(
                    os.path.join(a.outdir, f'test_r_{ix}_{iy}_{iz}.{suffix}'))
    print(f'wrote {a.cubes**3} cubes of {C}^3 x {a.components} to {a.outdir} (grid {G}^3, box {a.box})')


if __name__ == '__main__':
    main()
