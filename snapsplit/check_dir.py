import os
import sys

# Set the directory to check
a = sys.argv[1]
directory = f"/home/bulow/codes/snapsplit/p3072_b128_pm_z99_a={a}_fixed"  # <-- CHANGE THIS

# Expected range
ix_range = range(6)
iy_range = range(6)
iz_range = range(6)
N_range = range(163)

# Generate expected filenames
expected_files = {
    f"a={a}" + r'__' + f"r_{ix}_{iy}_{iz}_snapshot.{N}"
    for ix in ix_range
    for iy in iy_range
    for iz in iz_range
    for N in N_range
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

