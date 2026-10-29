#!/bin/sh
# Copyright 2026 Grant McLendon
# SPDX-License-Identifier: MIT
#
# usage: ./check.sh [aco-lds-repro args...]
# Dumps the shader with and without ACO's scheduler and reports reordered LDS
# spills/reloads, then runs it with the scheduler, with ACO_DEBUG=nosched, and
# with --fill-lds, against the CPU. Exit status 1 if the scheduled run is wrong
# or a reordered pair is found, else 0 (also when the shader doesn't spill to
# LDS, as before Mesa commit 133ef9f9).
set -u
cd "$(dirname "$0")"
bin=target/release/aco-lds-repro
mkdir -p out
export MESA_SHADER_CACHE_DISABLE=true
RADV_DEBUG=shaders,shaderstats,nocache "$bin" --runs 1 --dump out "$@" 2>out/dump.txt >/dev/null
ACO_DEBUG=nosched RADV_DEBUG=shaders,nocache "$bin" --runs 1 "$@" 2>out/dump-nosched.txt >/dev/null
grep -E '^ *(VGPRs|SGPRs|Spilled VGPRs|Spilled SGPRs|LDS size|Scratch size|Pre-Sched VGPRs):' out/dump.txt | sed 's/^ */  /'
python3 find_reordered_spills.py out/dump.txt out/dump-nosched.txt
pairs=$?
echo "--- scheduler on"
"$bin" "$@" | tail -n 3
echo "--- ACO_DEBUG=nosched"
ACO_DEBUG=nosched "$bin" "$@" | tail -n 1
echo "--- --fill-lds (spills to scratch)"
"$bin" --fill-lds "$@" | tail -n 1
[ "$pairs" != 1 ] && "$bin" --runs 1 "$@" >/dev/null && exit 0
exit 1
