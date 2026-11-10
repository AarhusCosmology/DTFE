#!/bin/bash
# Tests of the mosaic pipeline on small synthetic data; run from anywhere:
#     bash mosaic/tests/run_tests.sh [WORKDIR]
# Needs python with numpy, numba, mpi4py and mpi4py-fft, and mpirun, on the PATH.
# The full-chain test also needs the DTFE executable (built in the repository root).
# Set MPIRUN to change how MPI programs are started (default: mpirun --oversubscribe).

HERE=$(cd "$(dirname "$0")" && pwd)
MOSAIC=$(dirname "$HERE")
WORK=${1:-$(mktemp -d)}
MPIRUN=${MPIRUN:-"mpirun --oversubscribe"}
DTFE=${DTFE:-$MOSAIC/../DTFE}
mkdir -p "$WORK" && cd "$WORK" || exit 1
echo "working in $WORK"
failed=0
check() { python "$HERE/check.py" "$@" || failed=1; }

echo "=== power spectrum of a synthetic mosaic (4^3 cubes of 16^3, 1 and 3 components)"
for nc in 1 3; do
    suffix=$([ $nc = 1 ] && echo a_velDiv || echo a_velVort)
    python "$HERE/make_mosaic.py" "mosaic$nc" --cubes 4 --cube 16 --box 100 --components $nc > /dev/null || failed=1
    for np in 1 4 6; do
        $MPIRUN -n $np python "$MOSAIC/mosaic_power.py" "mosaic$nc/test_r_0_0_0.$suffix" "pk${nc}_np$np.txt" \
            --box 100 > "log_power${nc}_np$np.txt" 2>&1 || { echo "FAIL: mosaic_power.py on $np ranks"; failed=1; }
    done
    echo "--- $nc component(s): 4 and 6 ranks vs 1 rank"
    check same "pk${nc}_np4.txt" "pk${nc}_np1.txt"
    check same "pk${nc}_np6.txt" "pk${nc}_np1.txt"
    echo "--- $nc component(s): vs numpy FFT of the full field, and vs the input power spectrum"
    check numpy "pk${nc}_np4.txt" "mosaic$nc/test_full.npy" 100
    check model "pk${nc}_np4.txt" 100 $nc
    echo "--- $nc component(s): --hubble converts to k in 1/Mpc and P in Mpc"
    $MPIRUN -n 4 python "$MOSAIC/mosaic_power.py" "mosaic$nc/test_r_0_0_0.$suffix" "pk${nc}_mpc.txt" \
        --box 100 --hubble 0.7 > "log_power${nc}_mpc.txt" 2>&1 || { echo "FAIL: mosaic_power.py --hubble"; failed=1; }
    check hubble "pk${nc}_mpc.txt" "pk${nc}_np4.txt" 0.7
done

if [ -x "$DTFE" ]; then
    echo "=== full chain: split -> DTFE -> power (run_mosaic.py --local) vs DTFE on the whole box"
    python "$HERE/make_snapshot.py" snap --particles 64 --box 64000 --files 8 > /dev/null || failed=1
    MPIRUN="$MPIRUN" python "$MOSAIC/run_mosaic.py" snap/snapshot run --box 64 --particles 64 --cube 32 \
        --hubble 0.7 --norm 3e5 --local --site "$HERE/site_test.sh" --dtfe "$DTFE" > log_chain.txt 2>&1 \
        || { echo "FAIL: run_mosaic.py, see $WORK/log_chain.txt"; failed=1; }
    grep -h "Skipped" run/local-dtfe_0.out
    mkdir -p full
    "$DTFE" "snap/snapshot.%i" full/whole_r_0_0_0 --grid 64 64 64 --field divergence_a vorticity_a --periodic \
        --samples 1000 --method 1 > log_dtfe_whole.txt 2>&1 || { echo "FAIL: DTFE on the whole box"; failed=1; }
    for f in div vort; do
        suffix=$([ $f = div ] && echo a_velDiv || echo a_velVort)
        $MPIRUN -n 1 python "$MOSAIC/mosaic_power.py" "full/whole_r_0_0_0.$suffix" "pk_whole_$f.txt" --box 64 \
            --hubble 0.7 --norm 3e5 > /dev/null 2>&1 || failed=1
        echo "--- $f: mosaic vs whole box (differences are DTFE's Monte Carlo sampling noise)"
        check same "run/power_$f.txt" "pk_whole_$f.txt" --rtol 0.02
    done
else
    echo "=== skipping the full-chain test: no DTFE executable at $DTFE"
fi

if [ $failed = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $failed
