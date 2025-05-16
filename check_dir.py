import os
import sys

# Set the directory to check
a = sys.argv[1]
particles = 2048
boxsize = 512
directory = f"/home/bulow/codes/DTFE/output_test/p{particles}_fixed"  # <-- CHANGE THIS

# Expected range
nxyz = int(particles/512)
ix_range = range(nxyz)
iy_range = range(nxyz)
iz_range = range(nxyz)

# Generate expected filenames
expected_files = {
    f"p{particles}_b{boxsize}_pm_a_{a}_r_{ix}_{iy}_{iz}.a_velVort"
    for ix in ix_range
    for iy in iy_range
    for iz in iz_range
}

# List actual files in the directory
actual_files = set(os.listdir(directory))

# Find missing files
missing_files = expected_files - actual_files

# Output
if missing_files:
    print(f"{len(missing_files)} files are missing:\n")
    for fname in sorted(missing_files):
        print(fname)
    snapshotnumbers = set(tmp.split('.')[-1] for tmp in missing_files)
    print(','.join(snapshotnumbers))
else:
    print("All files are present.")

