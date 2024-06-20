#!/bin/bash
# Store the starting directory
START_DIR=$(pwd)
PREFIX="$START_DIR/../external"

# Function to clean external directory
clean_external() {
    rm -rf "$PREFIX" && mkdir "$PREFIX"
}

# Function to install GSL
install_gsl() {
    echo "Installing GSL..."
    cd "$START_DIR/gsl" || exit
    rm -rf build && mkdir build && cd build || exit
    cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX" -G"Unix Makefiles" -DNO_AMPL_BINDINGS=1
    make -j && make install
}

# Function to install boost
install_boost() {
    echo "Installing boost..."
    cd "$START_DIR/boost" || exit
    ./bootstrap.sh --prefix="$PREFIX"
    ./b2 install --prefix="$PREFIX" -j4
}

# Function to install GMP
install_gmp() {
    echo "Installing GMP..."
    wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
    tar -xf gmp-6.3.0.tar.xz
    cd "$START_DIR/gmp-6.3.0" || exit
    # Find the directory containing the m4 executable
    ./configure --prefix="$PREFIX"
    make -j && make install
}

# Function to install MPFR
install_mpfr() {
    echo "Installing MPFR..."
    cd "$START_DIR/MPFR" || exit
    ./configure --prefix="$PREFIX" --with-gmp="$PREFIX"
    make -j && make install
}

# Function to install CGAL
install_cgal() {
    echo "Installing CGAL..."
    cd "$START_DIR/cgal" || exit
    rm -rf build && mkdir build && cd build || exit
    cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX"
    make install
}

# Function to install HDF5
install_hdf5() {
    echo "Installing HDF5..."
    cd "$START_DIR/hdf5" || exit
    git checkout hdf5_1.14.4.3
    rm -rf build && mkdir build && cd build || exit
    cmake .. -DCMAKE_INSTALL_PREFIX="$PREFIX" -DHDF5_BUILD_CPP_LIB=ON
    make -j && make install
}
# Call installation functions
#clean_external
#install_gsl
#install_boost
#install_gmp
#install_mpfr
#install_cgal
install_hdf5