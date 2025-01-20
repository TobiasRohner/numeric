FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV HOME /root

SHELL ["/bin/bash", "-c"]

RUN apt update && apt -y --no-install-recommends install build-essential gcc git libssl-dev libopenmpi-dev python3 python3-numpy python3-sympy libhdf5-dev

RUN git clone --branch v3.24.3 https://github.com/Kitware/CMake.git \
    && cd CMake \
    && ./bootstrap --parallel=$(nproc) \
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

RUN cmake -DCMAKE_BUILD_TYPE=Release -B build numeric && cmake --build build --parallel=$(nproc)

ENTRYPOINT ["./build/test/test"]
