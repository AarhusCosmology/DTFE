import argparse
import os
import numpy as np
from numba import njit
import g3read as g3

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Partition a Gadget snapshot into N^3 subregions with ghost zones."
    )
    parser.add_argument("filename", type=str, help="Input Gadget snapshot filename")
    parser.add_argument("output_dir", type=str, help="Directory to save partitioned snapshots")
    parser.add_argument("N", type=int, help="Number of partitions per dimension (total = N^3)")
    parser.add_argument("prefix", type=str, help="Prefix for output snapshot filenames")
    parser.add_argument("ghost_zone_multiplier", type=float,
                        help="Ghost zone thickness in units of average particle spacing")

    return parser.parse_args()

def _assign_particles_to_regions(pos, vel, box_size, N, ghost_zone):
    npart = pos.shape[0]

    cell_size = box_size / N

    all_pos = [[] for _ in range(N**3)]
    all_vel = [[] for _ in range(N**3)]

    def index(ix, iy, iz):
        return ix * N * N + iy * N + iz

    for i in range(npart):
        px, py, pz = pos[i]
        cx = int(px / cell_size) % N
        cy = int(py / cell_size) % N
        cz = int(pz / cell_size) % N

        for dx_i in [-1, 0, 1]:
            for dy_i in [-1, 0, 1]:
                for dz_i in [-1, 0, 1]:
                    nx = (cx + dx_i) % N
                    ny = (cy + dy_i) % N
                    nz = (cz + dz_i) % N

                    # bounding box
                    xmin = (nx * cell_size - ghost_zone) % box_size
                    xmax = (nx * cell_size + cell_size + ghost_zone) % box_size
                    ymin = (ny * cell_size - ghost_zone) % box_size
                    ymax = (ny * cell_size + cell_size + ghost_zone) % box_size
                    zmin = (nz * cell_size - ghost_zone) % box_size
                    zmax = (nz * cell_size + cell_size + ghost_zone) % box_size

                    def in_bounds(x, x0, x1):
                        if x0 < x1:
                            return x0 <= x <= x1
                        else:
                            return x >= x0 or x <= x1

                    if in_bounds(px, xmin, xmax) and in_bounds(py, ymin, ymax) and in_bounds(pz, zmin, zmax):
                        idx = index(nx, ny, nz)
                        all_pos[idx].append(pos[i])
                        all_vel[idx].append(vel[i])

    all_pos = [np.array(p, dtype=np.float32) for p in all_pos]
    all_vel = [np.array(v, dtype=np.float32) for v in all_vel]

    return all_pos, all_vel


def write_subsnapshots(all_pos, all_vel, original_header, N, outprefix, outsuffix, output_dir='.'):
    # Create the output directory if it doesn't exist
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                idx = ix * N * N + iy * N + iz
                pos = all_pos[idx]
                vel = all_vel[idx]
                npart = len(pos)
                npart_array = [n for n in original_header.npart]
                npart_array[1] = npart

                # We need to write an empty file, because otherwise we will 
                # possibly screw up multi-file snapshots.
                #if npart == 0:
                #    continue
                
                filename = f"{output_dir}/{outprefix}_r_{ix}_{iy}_{iz}_{outsuffix}"
                if os.path.exists(filename):
                    print(f"File {filename} already exists!!! Removing.")
                    os.remove(filename)
                    #continue

                f = g3.GadgetWriteFile(filename, npart_array, {}, original_header)
                f.write_header(f.header)

                if npart > 0:

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

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
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
                            # Concatenate buffer with np.empty((growth, 3), dtype=np.float32)                                                 
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
    # Create suffix from filename. Most importantly, if it ends with .NUM, the
    # .NUM should be stored in the suffix.
    # Otherwise, the suffix should be the last part of the filename.

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
        # Estimate npart_total based on assumption of 512^3 subsnaps
        npart_total = (N*512)**3
    avg_particle_sep = box_size/npart_total**(1/3)
    ghost_zone = ghost_zone_multiplier * avg_particle_sep
    print(f"Box size: {box_size} kpc/h")
    print(f"Average particle separation: {avg_particle_sep} kpc/h")
    print(f"Ghost zone size: {ghost_zone} kpc/h = {ghost_zone/1e3/header.HubbleParam} Mpc")

    pos = f.read_new("POS ", 1)  # DM = parttype 1
    vel = f.read_new("VEL ", 1)

    print("Assigning particles to regions...")
    all_pos, all_vel = assign_particles_to_regions(pos, vel, box_size, N, ghost_zone)

    print("Writing subsnapshots...")
    write_subsnapshots(all_pos, all_vel, header, N, prefix, suffix, output_dir)

if __name__ == "__main__":
    main()
