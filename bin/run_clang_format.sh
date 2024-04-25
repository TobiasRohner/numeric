#! /usr/bin/env bash

root="$(realpath "$(dirname "$(readlink -f "$0")")"/..)"

find ${root}/include -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${root}/src -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${root}/test -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${root}/benchmark -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${root}/examples -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
find ${root}/doc -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \;
