#!/bin/bash

set -e
set -x

mkdir -p build
cd build

if [ ${cudatoolkit} == "8.0" ]; then
    # CXXFLAGS is used to compile C code. -fvisibility-inlines-hidden
    # is not valid for C which causes an issue when -Werror included
    CXXFLAGS="${CXXFLAGS//-fvisibility-inlines-hidden/}"
fi
CXXFLAGS="-std=c++11"
LDFLAGS=${LDFLAGS}" -L/usr/local/cuda/lib64"
LIBRT=$(find ${BUILD_PREFIX} -name "librt.so")
cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DUSE_CUDA=ON \
    -DCUDA_rt_LIBRARY=${LIBRT} \
    -DUSE_NCCL=ON \
    -DNCCL_ROOT=/usr/local/cuda \
    ..
make -j${CPU_COUNT} ${VERBOSE_CM}
