#!/usr/bin/env python3
"""Cut one file of a Gadget snapshot into N^3 sub-snapshots with ghost zones.

    python split_snapshot_in_regions.py SNAPSHOT_FILE OUTPUT_DIR N PREFIX GHOST [--particles NP]

For every region (ix, iy, iz) of an N x N x N division of the (periodic) box,
writes OUTPUT_DIR/PREFIX_r_{ix}_{iy}_{iz}_<basename of SNAPSHOT_FILE> with the
dark-matter (type 1) particles of SNAPSHOT_FILE that lie in the region or in a
ghost zone of GHOST mean particle separations around it. Running this on every
file snapshot.0, snapshot.1, ... gives each region its own multi-file snapshot
PREFIX_r_{ix}_{iy}_{iz}_snapshot.%i that DTFE can read. Regions that get no
particles from a file get no file, so DTFE needs --skipMissingFiles.

Only positions and velocities are written; particle masses must be in the header
(equal-mass particles). Memory use is about three times the input file size.
"""
import argparse
import os

import numpy as np
from numba import njit

import g3read as g3


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filename", type=str, help="input Gadget snapshot file")
    parser.add_argument("output_dir", type=str, help="directory for the sub-snapshots")
    parser.add_argument("N", type=int, help="number of regions per dimension (N^3 in total)")
    parser.add_argument("prefix", type=str, help="prefix of the output file names")
    parser.add_argument("ghost_zone_multiplier", type=float,
                        help="ghost zone thickness in units of the mean particle separation")
    parser.add_argument("--particles", type=int,
                        help="particles per dimension in the full snapshot; only needed if the header "
                             "does not contain the total particle number")
    return parser.parse_args()


def write_subsnapshots(all_pos, all_vel, original_header, N, outprefix, outsuffix, output_dir='.'):
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                idx = ix * N * N + iy * N + iz
                pos = all_pos[idx]
                vel = all_vel[idx]
                npart = len(pos)
                # no file for empty regions: there can be N^3 x (number of files) of them
                if npart == 0:
                    continue
                npart_array = [0, npart, 0, 0, 0, 0]

                filename = f"{output_dir}/{outprefix}_r_{ix}_{iy}_{iz}_{outsuffix}"
                if os.path.exists(filename):
                    print(f"File {filename} already exists! Removing.")
                    os.remove(filename)

                f = g3.GadgetWriteFile(filename, npart_array, {}, original_header)
                f.write_header(f.header)

                f.add_file_block('POS ', npart * 3 * 4, partlen=12)
                f.write_block('POS ', -1, pos)

                f.add_file_block('VEL ', npart * 3 * 4, partlen=12)
                f.write_block('VEL ', -1, vel)

                print(f"Wrote {filename} with {npart} particles")


@njit
def index(ix, iy, iz, N):
    return ix * N * N + iy * N + iz


@njit
def in_bounds(x, x0, x1):
    if x0 < x1:
        return x0 <= x <= x1
    else:
        return x >= x0 or x <= x1


@njit
def assign_particles_to_regions(pos, vel, box_size, N, ghost_zone):
    npart = pos.shape[0]
    cell_size = box_size / N
    n_cells = N * N * N
    # neighbouring regions along each axis; for N < 3 the offsets -1 and +1 are
    # the same region and must only be visited once
    if N >= 3:
        offsets = np.array([-1, 0, 1])
    elif N == 2:
        offsets = np.array([0, 1])
    else:
        offsets = np.array([0])

    # Allocate buffers
    pos_buffers = [np.empty((1000, 3), dtype=np.float32) for _ in range(n_cells)]
    vel_buffers = [np.empty((1000, 3), dtype=np.float32) for _ in range(n_cells)]
    counters = np.zeros(n_cells, dtype=np.int64)

    for i in range(npart):
        px, py, pz = pos[i]
        vx, vy, vz = vel[i]
        cx = int(px / cell_size) % N
        cy = int(py / cell_size) % N
        cz = int(pz / cell_size) % N

        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    nx = (cx + dx) % N
                    ny = (cy + dy) % N
                    nz = (cz + dz) % N

                    xmin = (nx * cell_size - ghost_zone) % box_size
                    xmax = (nx * cell_size + cell_size + ghost_zone) % box_size
                    ymin = (ny * cell_size - ghost_zone) % box_size
                    ymax = (ny * cell_size + cell_size + ghost_zone) % box_size
                    zmin = (nz * cell_size - ghost_zone) % box_size
                    zmax = (nz * cell_size + cell_size + ghost_zone) % box_size

                    if (in_bounds(px, xmin, xmax) and
                        in_bounds(py, ymin, ymax) and
                        in_bounds(pz, zmin, zmax)):
                        idx = index(nx, ny, nz, N)
                        if counters[idx] >= pos_buffers[idx].shape[0]:
                            # double the buffer
                            pos_buffers[idx] = np.concatenate((
                                pos_buffers[idx][:counters[idx]],
                                np.empty((pos_buffers[idx].shape[0], 3), dtype=np.float32)
                            ))
                            vel_buffers[idx] = np.concatenate((
                                vel_buffers[idx][:counters[idx]],
                                np.empty((pos_buffers[idx].shape[0], 3), dtype=np.float32)
                            ))
                        pos_buffers[idx][counters[idx]] = np.array([px, py, pz], dtype=np.float32)
                        vel_buffers[idx][counters[idx]] = np.array([vx, vy, vz], dtype=np.float32)
                        counters[idx] += 1

    # Final compact output
    all_pos = [pos_buffers[i][:counters[i]] for i in range(n_cells)]
    all_vel = [vel_buffers[i][:counters[i]] for i in range(n_cells)]

    return all_pos, all_vel


def main():
    args = parse_arguments()

    filename = args.filename
    # keep the file number: 'snapshot.15' -> outputs ..._r_i_j_k_snapshot.15
    suffix = filename.split('/')[-1]
    output_dir = args.output_dir
    N = args.N
    prefix = args.prefix
    ghost_zone_multiplier = args.ghost_zone_multiplier

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading snapshot from {filename}")
    f = g3.GadgetFile(filename)
    header = f.header

    box_size = header.BoxSize
    npart = sum(header.npart)

    total_particles = header.NallHW.astype(np.uint64) * 2**32 + header.npartTotal.astype(np.uint64)
    npart_total = sum(total_particles)
    print(f"Number of particles in snapshot: {npart}")
    print(f"Total number of particles: {npart_total}")
    if npart_total == 0:
        if args.particles is None:
            raise SystemExit("The header has no total particle number; give it with --particles")
        npart_total = args.particles**3
    if header.mass[1] == 0:
        raise SystemExit("Particle masses are not in the header; only equal-mass particles are supported")
    avg_particle_sep = box_size / npart_total**(1 / 3)
    ghost_zone = ghost_zone_multiplier * avg_particle_sep
    print(f"Box size: {box_size} (position units)")
    print(f"Average particle separation: {avg_particle_sep}")
    print(f"Ghost zone size: {ghost_zone}")

    pos = f.read_new("POS ", 1)  # DM = parttype 1
    vel = f.read_new("VEL ", 1)

    print("Assigning particles to regions...")
    all_pos, all_vel = assign_particles_to_regions(pos, vel, box_size, N, ghost_zone)

    print("Writing subsnapshots...")
    write_subsnapshots(all_pos, all_vel, header, N, prefix, suffix, output_dir)


if __name__ == "__main__":
    main()
