# Evidence: ACO's scheduler reorders LDS spills against their reloads

The full write-up behind [README.md](README.md).

A standalone reproduction for RADV/ACO. Since 133ef9f9 ("aco: spill VGPRs to
LDS if it doesn't further limit occupancy"), ACO can spill the VGPRs of a
single-wave compute shader to LDS with `ds_write_addtid_b32` /
`ds_read_addtid_b32`. `spill_vgpr()` and `reload_vgpr()` in `aco_spill.cpp`
build these without a `memory_sync_info` (the scratch path uses
`memory_sync_info(storage_vgpr_spill, semantic_private)`), so their storage is
`storage_none`. The scheduler runs after the spiller, and its hazard query
(`sync.storage & aliasing_storage`) then sees nothing ordering a spill and a
reload of the same slot, which have no register dependency on each other
either. It moves one past the other, and the reload reads the wrong value.

## Result

AMD Radeon RX 6800 (NAVI21, GFX10.3), Mesa 26.2.4 (Fedora
mesa-vulkan-drivers-26.2.4-1.fc44), Linux 7.2:

| run                                   | spilled VGPRs (ACO's stat) | result                          |
|---------------------------------------|----------------------------|---------------------------------|
| default                               | 388, to LDS                | 15379 of 16384 lanes wrong, every run |
| `ACO_DEBUG=nosched`                   | 388, to LDS                | all lanes match the CPU         |
| `--fill-lds` (no LDS left for spills) | 388, to scratch            | all lanes match the CPU         |

The wrong values are the slot's other value, not a race: in the default
shader the one reordered pair (below) makes the reload of per-lane value 287
return the value the scheduler wrote over it, `gid.x + it * 290`, and a CPU
model with that one substitution matches the GPU in all 16384 lanes (the same
substitution at values 286 or 288 misses in about 16300).

Other controls: `ACO_DEBUG=nosched-ilp,nosched-vopd` (the post-RA schedulers
off) still gives the 15379 wrong lanes, as do `ACO_DEBUG=force-waitcnt` and
`force-waitdeps,force-waitcnt`, so it is the pre-RA scheduler and not a
missing wait. `ACO_DEBUG=validateir,validatera` reports nothing.

Mesa main (27.0.0-devel, 087e1fe, 2026-10-28) gives the same result: the
same reordered pair and the same 15379 wrong lanes. The spill and scheduler
code quoted here is from the mesa-26.2.4 tag; main's is unchanged but for
formatting. The SPIR-V passes `spirv-val --target-env vulkan1.1`, and a run
under `VK_LAYER_KHRONOS_validation` with synchronization validation on
reports nothing.

It starts with 133ef9f9. Built at its parent, 56a65287, the shader spills the
same 388 VGPRs to scratch and every run is correct; built at 133ef9f9, it
spills to LDS and gives the same 15379 wrong lanes, with the same reordered
pair.

## Running it

Needs Rust (1.87+, naga's minimum) and a Vulkan loader; no SDK or shader
compiler (the WGSL is compiled to SPIR-V in-process by naga).

```
cargo build --release
./target/release/aco-lds-repro                       # exits 1: lanes differ from the CPU
ACO_DEBUG=nosched ./target/release/aco-lds-repro     # exits 0
./target/release/aco-lds-repro --fill-lds            # exits 0
./check.sh                                           # all three, plus the dump analysis below; exits 1 if the bug shows
```

It picks the first AMD device. Each lane's result is checked against the
same arithmetic done on the CPU.

`shader/repro.spv` (and `shader/repro.wgsl`, its source) is the default
shader, for use with other tools. Its interface: set 0, binding 0 a `u32`
storage buffer of per-invocation seeds, binding 1 the `u32` results, binding 2
a uniform `uvec4` (x = loop trip count, 4; y = 0x12345678; z = 0), binding 3
a storage buffer of 1024 `u32` coefficients. `local_size` 64.

## The shader

One 64-invocation workgroup (one wave64), no workgroup memory, so LDS is
free for spilling. It carries 290 per-lane `u32`s and 200 workgroup-uniform
`u32`s around a loop whose trip count is a uniform, so all are live at once:
ACO's pre-scheduling demand is about 295 VGPRs, and SGPRs spill too (362).
Each update of a per-lane value loads a coefficient with a per-lane address
(a `buffer_load_dword`); the scheduler's moves around those loads are what
reorder the spills. `--fill-lds` adds a 32 KiB workgroup array: at 256 VGPRs
ACO lets the spills use at most 16 KiB of LDS, so none is left and they go to
scratch.

Sizes: `--vars N` (per-lane values), `--uniforms M`, `--iters`, `--groups`,
`--runs`. On the RX 6800 (16 workgroups, 2 runs each), vars 260–330 in steps
of 5 against uniforms 50, 100, 150, 175, 200, 225, 250 and 300: 35 of the 120
sizes are wrong on Fedora's driver (from 175–200 uniforms up to 250–300, vars
260–305), 0 with the fix. With no uniforms, all 143 sizes from 258 to 400 per-lane values are
correct. The scheduler limits itself to 254 VGPRs while the spiller fills
256, so in a block held at the limit the moves fail the pressure check; the
SGPR spills appear to give it room, but which room is a guess.

## What the scheduler does

`find_reordered_spills.py` compares the "After RA" IR from
`RADV_DEBUG=shaders` with and without `ACO_DEBUG=nosched`. The spiller's SSA
ids are the same in both (bar a few accesses at some sizes, which it skips
with a warning: 2–4 at 9 of the 120 sizes above), so each LDS spill (by the
value it stores) and reload (by the value it defines) can be matched up, and
any spill/reload pair of one LDS offset whose order differs is reported. For
the default shader there is one of 390 reloads, in the loop body (BB3), at
offset 2560. Over the 120 sizes, the sizes where it finds a pair are exactly
the ones the GPU gets wrong.

`ACO_DEBUG=nosched` (the spiller's order): the slot holds `%9492`, which is
reloaded as `%13286`, and only then is `%6352` spilled to it.

```
ds_write_addtid_b32 %13405:m0, %9492:v[82] offset0:2560          <- spill %9492
v1: %9493:v[150] = v_add_u32 0x11f, %6352:v[5]
...
v1: %9496:v[150] = buffer_load_dword %6353:s[4-7], %9495:v[150], 0 offen storage:buffer semantics:private,reorder
v1: %9497:v[150] = v_mul_lo_u32 %5521:v[233], %9496:v[150]
v1: %9498:v[40] = v_xor_b32 %13282:v[40], %9497:v[150]
s2: %9499:s[48-49] = v_cmp_class_f32 %6360:s[69], %5521:v[233]
v1: %13286:v[3] = ds_read_addtid_b32 %13405:m0 offset0:2560      <- reload %9492
v1: %9500:v[3] = v_cndmask_b32 %13286:v[3], %9498:v[40], %9499:s[48-49]
v1: %9501:v[3] = v_add_u32 %6011:s[83], %9500:v[3]
v1: %9502:v[40] = v_add_u32 %5521:v[233], %5347:v[2] ...
ds_write_addtid_b32 %13405:m0, %6352:v[5] offset0:2560          <- spill %6352
v1: %9503:v[5] = v_add_u32 0x120, %6352:v[5]
```

Scheduled: the spill of `%6352` is above the reload, which now reads
`%6352`, not `%9492`.

```
ds_write_addtid_b32 %13405:m0, %9492:v[82] offset0:2560          <- spill %9492
v1: %9493:v[150] = v_add_u32 0x11f, %6352:v[5]
v1: %9494:v[150] = v_and_b32 %13285:s[89], %9493:v[150]
v1: %9495:v[150] = v_lshlrev_b32 2, (is16bit)%9494:v[150]
ds_write_addtid_b32 %13405:m0, %6352:v[5] offset0:2560          <- spill %6352
v1: %9503:v[5] = v_add_u32 0x120, %6352:v[5]
...
v1: %9496:v[150] = buffer_load_dword %6353:s[0-3], %9495:v[150], 0 offen storage:buffer semantics:private,reorder
v1: %9506:v[5] = buffer_load_dword %6353:s[0-3], %9505:v[5], 0 offen storage:buffer semantics:private,reorder
s2: %9499:s[50-51] = v_cmp_class_f32 %6360:s[70], %5521:v[233]
v1: %9497:v[150] = v_mul_lo_u32 %5521:v[233], %9496:v[150]
v1: %9498:v[40] = v_xor_b32 %13282:v[40], %9497:v[150]
v1: %13286:v[3] = ds_read_addtid_b32 %13405:m0 offset0:2560      <- reload: gets %6352
v1: %9500:v[3] = v_cndmask_b32 %13286:v[3], %9498:v[40], %9499:s[50-51]
```

Two moves do it (traced with a logging build of the scheduler). First the
reload's own `schedule_LDS` upward pass pulls the later spill of `%6352` and
the next load's address (`%9503`–`%9505`) up to just below the reload. Then
`schedule_VMEM`'s downward pass for the next load (`%9506`) moves the uses of
the first load, and the reload with them, below `%9506`, while the spill of
`%6352` can't move: `%9503`, the next load's address, is `%6352`'s last use.
The reload passes the spill because with `storage_none` neither is in the
hazard query's aliasing classes.

The other order, a reload moved above its own spill, was what first showed
up (in a larger shader, not this one). The upward passes of `schedule_VMEM`,
`schedule_SMEM` and `schedule_LDS` move DS instructions too (only the SMEM and
LDS downward passes skip them), and for a value spilled right after a load and
reloaded later, `schedule_VMEM`'s upward pass moves the reload above the spill
(the unit test `scheduler.lds_spill_order_up` builds that case). The analyzer
reports both orders.

## Fix

`patches/aco-spill-order-lds-spills.patch` gives the LDS spills and reloads
`memory_sync_info(storage_shared, semantic_private)`: LDS, accessed by this
wave alone. The hazard query then keeps them in order against each other
(and against the shader's own LDS accesses, which they never alias:
conservative, but cheap), as `storage_vgpr_spill` does for scratch spills.
`semantic_private` keeps them out of the scheduler's and the waitcnt pass's
barrier logic, as today: the waitcnt pass's release tracking only keeps
private stores of `storage_scratch` and `storage_vgpr_spill`, so the code
around barriers and the GFX11+ VGPR deallocation is unchanged (an ACO unit
test of the deallocation shows the same code with this sync as with none).

The other choice, the scratch spills' `memory_sync_info(storage_vgpr_spill,
semantic_private)`, also fixes the reproduction but costs more:

- a conflict between two `storage_vgpr_spill` accesses is a VMEM/SMEM-class
  hazard, which ends `schedule_VMEM`'s and `schedule_SMEM`'s downward scans
  where `storage_shared` only pins the instruction, and it also orders the
  LDS spills against the scratch ones;
- the waitcnt pass then tracks LDS spill stores for the GFX11+ VGPR
  deallocation and adds an `s_waitcnt lgkmcnt(0)` before
  `s_sendmsg dealloc_vgprs` when a VMEM store is still pending.

Measured with `fossilize_replay` and ACO's pipeline statistics, over the
compute pipelines of the 18 Steam games with Fossilize archives on this
machine (7,625 pipelines; RADV built from mesa-26.2.4 with asserts and IR
validation on, unpatched and with each fix): only 5 pipelines change, all
copies of one DOOM: The Dark Ages shader that spills 130 VGPRs to LDS and
scratch (the unpatched driver doesn't reorder any of its pairs). With
`storage_vgpr_spill` its instructions go 743 → 735 and ACO's latency estimate
9585 → 9948 (+3.8%), inverse throughput 4792 → 4974; with `storage_shared`
742 instructions, latency 9586, inverse throughput 4793. A second unpatched
replay differs from the first in one pipeline (a shader that compiles
nondeterministically), which sets the noise. No replay crashed or asserted.
On this reproduction's default shader both cost about 0.3% of the latency
estimate. The ray-tracing pipelines (50, all DOOM: The Dark Ages; LDS
spilling also applies to ray-tracing shaders) replayed the same way: none
change, none crash. Not measured: run time.

The patches are against mesa-26.2.4 and also apply to main as of 2026-10-28
(087e1fe; offsets of 1–4 lines). On main, the two spill tests fail without
the fix and all four pass with it; the full `aco_tests` suite's other
failures (`assembler.*`, which needs `clrxdisasm`, `d3d11_derivs.*` and three
`isel.*` tests, in this toolbox) are the same with and without the fix.

Tested on the RX 6800, with RADV built from mesa-26.2.4 (release, RADV only,
the configuration below, in a Fedora 44 toolbox) and loaded through
`VK_DRIVER_FILES`:

| driver                          | default shader                         | 120-size grid | `--uniforms 0`, 258–400 |
|---------------------------------|----------------------------------------|---------------|-------------------------|
| Fedora's mesa 26.2.4            | 15379 of 16384 lanes wrong, every run  | 35 wrong      | all 143 correct         |
| mesa-26.2.4, as built           | 15379 of 16384 lanes wrong, every run  | not run       | not run                 |
| mesa-26.2.4 + the patch         | all lanes match, 3 of 3 runs           | 0 wrong       | not run                 |
| 56a65287 (133ef9f9's parent)    | all lanes match, 10 of 10 runs         | not run       | not run                 |
| 133ef9f9                        | 15379 of 16384 lanes wrong, every run  | not run       | not run                 |
| main (087e1fe), as built        | 15379 of 16384 lanes wrong, every run  | not run       | not run                 |
| main (087e1fe) + the patch      | all lanes match, 10 of 10 runs         | not run       | not run                 |

With the patch the spills are the same (388 VGPRs, LDS size 12288, no
scratch) and the analyzer finds no reordered pairs.

To build and test it:

```
git clone --depth 1 --branch mesa-26.2.4 https://gitlab.freedesktop.org/mesa/mesa.git
cd mesa && git apply ../patches/aco-spill-order-lds-spills.patch
meson setup build -Dbuildtype=release -Dvulkan-drivers=amd -Dgallium-drivers= \
  -Dplatforms= -Dllvm=disabled -Damd-use-llvm=false -Dopengl=false -Degl=disabled \
  -Dgbm=disabled -Dglx=disabled -Dgles1=disabled -Dgles2=disabled -Dvideo-codecs= \
  -Dvulkan-layers= -Dtools=
ninja -C build src/amd/vulkan/libvulkan_radeon.so src/amd/vulkan/radeon_icd.x86_64.json
# an ICD manifest whose library_path is the built library:
mkdir -p ../out
sed "s#\"library_path\": *\"[^\"]*\"#\"library_path\": \"$PWD/build/src/amd/vulkan/libvulkan_radeon.so\"#" \
  build/src/amd/vulkan/radeon_icd.x86_64.json > ../out/radeon_built_icd.json
cd .. && VK_DRIVER_FILES=$PWD/out/radeon_built_icd.json ./check.sh
```

On main, leave out `-Damd-use-llvm=false`: main replaced it with
`amd-options` and rejects it.

`patches/aco-tests-lds-spill-order.patch` adds four ACO unit tests (apply it
after the fix; build with `-Dllvm=enabled -Dtools=drm-shim
-Dbuild-aco-tests=true`, then run `aco_tests spill. scheduler.` with
`LD_PRELOAD=build/src/amd/drm-shim/libamdgpu_noop_drm_shim.so`):

- `spill.lds_sync` and `spill.lds_sync_multi_dword` run `spill()` on a
  one-wave compute shader with 260 VGPRs live (as `v1`, and as `v2` values
  that are split and reassembled) and check that its LDS spills and reloads
  carry the sync info. Both fail without the fix.
- `scheduler.lds_spill_order` (a spill and reload of one slot, then of
  another value to the same slot, around two buffer loads) and
  `scheduler.lds_spill_order_up` (a loaded value spilled at once and
  reloaded later) check that `schedule_program()` keeps spills and reloads
  built with that sync in order. They don't depend on the fix; built without
  the sync, both are reordered and fail.

All four pass with the fix; the rest of `aco_tests` gives the same results
with and without it (in this build the `assembler`, `d3d11_derivs` and `isel`
groups fail either way).

(Build dependencies on Fedora: meson ninja-build gcc-c++ python3-mako
python3-pyyaml python3-packaging libdrm-devel expat-devel
zlib-ng-compat-devel elfutils-libelf-devel libzstd-devel glslang bison
flex.)
