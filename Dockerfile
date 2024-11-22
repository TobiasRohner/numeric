FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV HOME /root

SHELL ["/bin/bash", "-c"]

RUN apt update && apt -y --no-install-recommends install build-essential gcc git libssl-dev

RUN git clone --branch v3.24.3 https://github.com/Kitware/CMake.git \
    && cd CMake \
    && ./bootstrap --parallel=$(nproc) \
    && make -j$(nproc) \
    && make install
    
RUN apt -y --no-install-recommends install libopenmpi-dev python3 python3-numpy python3-sympy libhdf5-dev

RUN git clone -b rocm-6.2.4 https://github.com/ROCm/llvm-project.git \
    && cd llvm-project/amd/hipcc \
    && mkdir build; cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install

RUN git clone -b rocm-6.2.4 https://github.com/ROCm/clr.git \
    && git clone -b rocm-6.2.4 https://github.com/ROCm/HIP.git \
    && git clone -b rocm-6.2.4 https://github.com/ROCm/hipother.git \
    && cd clr \
    && mkdir -p build; cd build \
    && cmake -DHIP_COMMON_DIR=/HIP -DHIPCC_BIN_DIR=/usr/local/bin -DHIP_PLATFORM=nvidia -DHIP_CATCH_TEST=0 -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIPNV_DIR=/hipother/hipnv .. \
    && make -j$(nproc) \
    && make install

RUN rm -rf ./numeric
COPY benchmark/ ./numeric//benchmark/
COPY cmake/ ./numeric/cmake/
COPY data/ ./numeric/data/
COPY doc/ ./numeric/doc/
COPY examples/ ./numeric/examples/
COPY include/ ./numeric/include/
COPY scripts/ ./numeric/scripts/
COPY src/ ./numeric/src/
COPY test/ ./numeric/test/
COPY CMakeLists.txt ./numeric/

RUN cmake -DCMAKE_BUILD_TYPE=Release -DHIP_DIR=/opt/rocm/lib/cmake/hip -DHIP_ROOT_DIR=/opt/rocm/bin -DNUMERIC_BUILD_DOCS=OFF -B build numeric && cmake --build build --parallel=$(nproc)

ENTRYPOINT ["./build/test/test"]
