# radv-aco-lds-spill-repro

On RADV, ACO's scheduler can move a VGPR spill to LDS above an earlier
reload of the same LDS slot. The reload then returns the wrong value, and the
compute shader gives wrong results. This starts with Mesa commit 133ef9f9
("aco: spill VGPRs to LDS if it doesn't further limit occupancy", Mesa 26.1).
It is fixed by giving the LDS spills and reloads a `memory_sync_info`, as the
scratch spills already have (see `patches/`).

Mesa issue: not filed yet.

## Run it

Needs an AMD GPU on RADV, Rust 1.87 or newer, and a Vulkan loader.

```
cargo build --release
./check.sh
```

Affected: Mesa 26.1 and later (tested on 26.2.4 and on main at 087e1fe). There,
`check.sh` exits 1 and prints:

```
BB3 offset 2560: spill above the slot's previous reload
  out/dump.txt:20121: ds_write_addtid_b32 %13405:m0, %6352:v[5] offset0:2560
  out/dump.txt:20131: v1: %13286:v[3] = ds_read_addtid_b32 %13405:m0 offset0:2560
1 same-slot spill/reload pairs reordered by the scheduler (390 LDS reloads)
--- scheduler on
run 8: 15379 of 16384 lanes differ from the CPU (first: lane 0, got 0x47eccf21, expected 0x7f89d10d)
run 9: 15379 of 16384 lanes differ from the CPU (first: lane 0, got 0x47eccf21, expected 0x7f89d10d)
10 of 10 runs wrong
--- ACO_DEBUG=nosched
0 of 10 runs wrong
--- --fill-lds (spills to scratch)
0 of 10 runs wrong
```

With a driver you built: `VK_DRIVER_FILES=/path/to/radeon_icd.x86_64.json ./check.sh`.

## Contents

- `src/main.rs`: the Vulkan program. It generates the shader, runs it and
  checks every lane against the CPU.
- `shader/`: the default shader, as WGSL and SPIR-V.
- `check.sh`: dumps the shader with and without ACO's scheduler and runs it
  with the scheduler on, with `ACO_DEBUG=nosched`, and with `--fill-lds`.
- `find_reordered_spills.py`: finds LDS spills and reloads the scheduler reordered.
- `patches/`: a proposed Mesa fix and four ACO unit tests.
- [`EVIDENCE.md`](EVIDENCE.md): the full write-up. It covers the mechanism,
  the controls, the regression builds, the fix and its measured cost.

## How this was made

Made with Claude Code, an AI coding assistant. It did the investigation and
wrote the Vulkan program, the scripts, EVIDENCE.md and the patches. Every
result in EVIDENCE.md comes from running them on the RX 6800 described there,
and EVIDENCE.md gives the commands to rerun each one.

## License

MIT, the same license as Mesa (`SPDX-License-Identifier: MIT`); see
[LICENSE](LICENSE). The files in `patches/` are changes to Mesa and stay
under Mesa's MIT license.
