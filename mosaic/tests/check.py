#!/usr/bin/env python3
"""Checks used by run_tests.sh.

    check.py same A B [--rtol R]            two spectra agree (same bins and modes, P within R)
    check.py model PK BOX NCOMP             spectrum agrees with make_mosaic.model_power
    check.py numpy PK FULL.npy BOX          spectrum agrees with a single-process numpy FFT of the full field
    check.py hubble PK_MPC PK_H H           PK_MPC (written with --hubble H) is k*H, P/H of PK_H (h-units)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))


def fail(msg):
    print('FAIL:', msg)
    sys.exit(1)


def same(a, b, rtol):
    a, b = np.loadtxt(a), np.loadtxt(b)
    if a.shape != b.shape or not np.array_equal(a[:, 2], b[:, 2]) or not np.allclose(a[:, 0], b[:, 0], rtol=1e-6):
        fail('different bins or mode counts')
    err = np.max(np.abs(a[1:, 1] / b[1:, 1] - 1))
    if err > rtol:
        fail(f'P differs by {err:.1e} > {rtol}')
    print(f'ok: same bins and modes, P agrees to {err:.1e}')


def model(pk, box, ncomp):
    from make_mosaic import model_power
    d = np.loadtxt(pk)
    m = d[:, 2] > 500
    ratio = d[m, 1] / (ncomp * model_power(d[m, 0], box))
    sigma = np.sqrt(2 / (ncomp * d[m, 2]))          # cosmic variance of each bin
    chi = (ratio - 1) / sigma
    if np.abs(chi).max() > 5 or abs(np.mean(chi)) * np.sqrt(len(chi)) > 5:
        fail(f'input power not recovered: max |chi| {np.abs(chi).max():.1f}, mean chi {np.mean(chi):.2f}')
    print(f'ok: input P(k) recovered, mean ratio {ratio.mean():.3f} over {m.sum()} bins (max |chi| {np.abs(chi).max():.1f})')


def numpy_reference(pk, full_npy, box):
    from mpi4py import MPI
    from mosaic_power import bin_power
    full = np.load(full_npy)
    G = full.shape[1]
    dk = 2 * np.pi / box
    kedges = np.arange(0., np.pi * G / box + dk / 2, dk)
    kslice = (slice(0, G), slice(0, G), slice(0, G // 2 + 1))
    psum = 0
    for comp in full:
        f_hat = (np.fft.rfftn(comp.astype(np.float64)) / G**3).astype(np.complex64)
        ksum, p, nsum = bin_power(f_hat, kslice, G, box, kedges, MPI.COMM_SELF)
        psum = psum + p
    sl = slice(1, len(kedges))
    ref = np.c_[ksum[sl] / nsum[sl], psum[sl] / nsum[sl], nsum[sl]]
    d = np.loadtxt(pk)
    if not np.array_equal(d[:, 2], ref[:, 2]):
        fail('mode counts differ from the numpy reference')
    err = np.max(np.abs(d[1:, 1] / ref[1:, 1] - 1))
    if err > 1e-5:
        fail(f'P differs from the numpy reference by {err:.1e}')
    print(f'ok: agrees with single-process numpy FFT to {err:.1e}')


def hubble(pk_mpc, pk_h, h):
    a, b = np.loadtxt(pk_mpc), np.loadtxt(pk_h)
    if a.shape != b.shape or not np.array_equal(a[:, 2], b[:, 2]):
        fail('different bins or mode counts')
    if not np.allclose(a[:, 0], b[:, 0] * h, rtol=1e-12, atol=0):
        fail(f'k is not k*h with h = {h}')
    err = np.max(np.abs(a[1:, 1] / (b[1:, 1] / h) - 1))
    if err > 1e-6:                                  # float32 FFTs of two runs differ at ~1e-8
        fail(f'P is not P/h with h = {h} (relative difference {err:.1e})')
    print(f'ok: k -> k h and P -> P/h with h = {h} (P to {err:.1e})')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'same':
        same(args[0], args[1], float(args[3]) if len(args) > 3 and args[2] == '--rtol' else 1e-5)
    elif cmd == 'model':
        model(args[0], float(args[1]), int(args[2]))
    elif cmd == 'hubble':
        hubble(args[0], args[1], float(args[2]))
    elif cmd == 'numpy':
        numpy_reference(args[0], args[1], float(args[2]))
    else:
        fail(f'unknown check {cmd}')
