// Copyright 2026 Grant McLendon
// SPDX-License-Identifier: MIT

//! Reproduction for ACO's scheduler reordering VGPR spills to LDS against their reloads.
//!
//! The shader is one 64-lane workgroup with no workgroup memory, carrying
//! `VARS` u32s around a loop whose trip count comes from a uniform, so all of
//! them are live at once and the register demand is well over 256 VGPRs. ACO
//! then spills VGPRs to LDS (single-wave compute, LDS free). Every lane's
//! result is checked against the same arithmetic on the CPU.
//!
//! `--uniforms M` adds M loop-carried workgroup-uniform u32s, mixed into the
//! updates, so that SGPRs spill too; ACO then reserves VGPRs for the SGPR
//! spills and spills VGPRs to below the scheduler's limit.
//!
//! `--fill-lds` declares a 32 KiB workgroup array, more than the LDS ACO lets
//! this shader's spills use, so ACO has no LDS left and spills to scratch.
//!
//! Usage: aco-lds-repro [--vars N] [--uniforms M] [--fill-lds] [--iters N] [--groups N] [--runs N] [--dump DIR]
//! Exit status: 0 if every run matched the CPU, 1 if any lane differed.

use ash::vk;
use std::ffi::CStr;
use std::fmt::Write as _;

const WG: u32 = 64;
/// Entries in the coefficient buffer each update loads from (a power of two).
const COEFS: u32 = 1024;
const USAGE: &str = "usage: aco-lds-repro [--vars N] [--uniforms M] [--fill-lds] [--iters N] [--groups N] [--runs N] [--dump DIR]";

struct Args {
    vars: usize,
    iters: u32,
    groups: u32,
    uniforms: usize,
    fill_lds: bool,
    runs: u32,
    dump: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args { vars: 290, iters: 4, groups: 256, uniforms: 200, fill_lds: false, runs: 10, dump: None };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--vars" => a.vars = v().parse().unwrap(),
            "--iters" => a.iters = v().parse().unwrap(),
            "--groups" => a.groups = v().parse().unwrap(),
            "--uniforms" => a.uniforms = v().parse().unwrap(),
            "--fill-lds" => a.fill_lds = true,
            "--runs" => a.runs = v().parse().unwrap(),
            "--dump" => a.dump = Some(v()),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown argument {k}\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    assert!(a.vars >= 2);
    a
}

// Per-variable constants, the same on both sides.
fn k(i: usize) -> u32 { (i as u32).wrapping_mul(0x9E37_79B9) | 1 }
fn sh(i: usize) -> u32 { 1 + (i as u32 * 7) % 29 }
fn ku(j: usize) -> u32 { (j as u32).wrapping_mul(0x85EB_CA6B) ^ 0x2545_F491 }
const USEED: u32 = 0x1234_5678;

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^ (x >> 16)
}

/// The shader's arithmetic, on the CPU.
fn reference(lane: u32, seed: u32, coef: &[u32], vars: usize, uniforms: usize, iters: u32) -> u32 {
    let mut v: Vec<u32> = (0..vars).map(|i| hash(seed ^ k(i))).collect();
    let mut u: Vec<u32> = (0..uniforms).map(|j| hash(USEED ^ ku(j))).collect();
    for it in 0..iters {
        for j in 0..uniforms {
            u[j] = u[j].wrapping_mul(0x0100_0193) ^ u[(j + 1) % uniforms];
        }
        for i in 0..vars {
            let n = v[(i + 1) % vars];
            let c = coef[(lane.wrapping_add(it.wrapping_mul(vars as u32)).wrapping_add(i as u32) & (COEFS - 1)) as usize];
            v[i] = if n & 4 != 0 { v[i] ^ n.wrapping_mul(c) } else { v[i].wrapping_add(n >> sh(i)) };
            if uniforms > 0 {
                v[i] = v[i].wrapping_add(u[i % uniforms]);
            }
        }
    }
    v.iter().fold(0u32, |h, &x| h.wrapping_mul(31).wrapping_add(x))
}

fn wgsl(vars: usize, uniforms: usize, fill_lds: bool) -> String {
    let mut s = String::new();
    if fill_lds {
        // 32 KiB: at 256 VGPRs, ACO lets a one-wave shader's spills use LDS up to
        // 64 KiB / (2 waves x 2 SIMDs) = 16 KiB on GFX10.3 (aco_spill.cpp).
        s.push_str("var<workgroup> fill: array<u32, 8192>;\n");
    }
    s.push_str(
        "@group(0) @binding(0) var<storage, read> seeds: array<u32>;\n\
         @group(0) @binding(1) var<storage, read_write> results: array<u32>;\n\
         @group(0) @binding(2) var<uniform> params: vec4<u32>;\n\
         @group(0) @binding(3) var<storage, read> coef: array<u32>;\n\n\
         fn hash(x0: u32) -> u32 {\n\
         \x20   var x = x0;\n\
         \x20   x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu;\n\
         \x20   return x ^ (x >> 16u);\n\
         }\n\n\
         @compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
         \x20   let seed = seeds[gid.x];\n",
    );
    for i in 0..vars {
        writeln!(s, "    var v{i}: u32 = hash(seed ^ {}u);", k(i)).unwrap();
    }
    for j in 0..uniforms {
        writeln!(s, "    var w{j}: u32 = hash(params.y ^ {}u);", ku(j)).unwrap();
    }
    s.push_str("    for (var it = 0u; it < params.x; it++) {\n");
    for j in 0..uniforms {
        writeln!(s, "        w{j} = (w{j} * 0x01000193u) ^ w{};", (j + 1) % uniforms).unwrap();
    }
    for i in 0..vars {
        let n = (i + 1) % vars;
        writeln!(
            s,
            "        v{i} = select(v{i} + (v{n} >> {}u), v{i} ^ (v{n} * coef[(gid.x + it * {vars}u + {i}u) & {}u]), (v{n} & 4u) != 0u);",
            sh(i),
            COEFS - 1
        )
        .unwrap();
        if uniforms > 0 {
            writeln!(s, "        v{i} += w{};", i % uniforms).unwrap();
        }
    }
    s.push_str("    }\n    var h = 0u;\n");
    for i in 0..vars {
        writeln!(s, "    h = h * 31u + v{i};").unwrap();
    }
    if fill_lds {
        s.push_str("    let slot = (gid.x & 63u) + params.z; // params.z is 0, unknown to the compiler: all of fill stays\n    fill[slot] = h;\n    workgroupBarrier();\n    h = fill[slot];\n");
    }
    s.push_str("    results[gid.x] = h;\n}\n");
    s
}

fn spirv(src: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(src).expect("WGSL parse");
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .expect("WGSL validate");
    let options = naga::back::spv::Options {
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &options, None).expect("SPIR-V out")
}

struct Gpu {
    _entry: ash::Entry,
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    qfam: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    name: String,
}

impl Gpu {
    unsafe fn new() -> Gpu {
        unsafe {
            let entry = ash::Entry::load().expect("libvulkan");
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).unwrap();
            let pdevs = instance.enumerate_physical_devices().unwrap();
            // The first AMD device (RADV), else the first device.
            let pdev = *pdevs
                .iter()
                .find(|&&p| instance.get_physical_device_properties(p).vendor_id == 0x1002)
                .unwrap_or(&pdevs[0]);
            let props = instance.get_physical_device_properties(pdev);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
            let qfam = instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .unwrap() as u32;
            let prio = [1.0];
            let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfam).queue_priorities(&prio)];
            let device = instance.create_device(pdev, &vk::DeviceCreateInfo::default().queue_create_infos(&qinfo), None).unwrap();
            let queue = device.get_device_queue(qfam, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdev);
            Gpu { _entry: entry, _instance: instance, device, queue, qfam, mem_props, name }
        }
    }

    unsafe fn buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> (vk::Buffer, vk::DeviceMemory, *mut u32) {
        unsafe {
            let buf = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buf);
            let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = (0..self.mem_props.memory_type_count)
                .find(|&t| req.memory_type_bits & (1 << t) != 0 && self.mem_props.memory_types[t as usize].property_flags.contains(want))
                .unwrap();
            let mem = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None).unwrap();
            self.device.bind_buffer_memory(buf, mem, 0).unwrap();
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).unwrap() as *mut u32;
            (buf, mem, ptr)
        }
    }
}

fn main() {
    let args = parse_args();
    let src = wgsl(args.vars, args.uniforms, args.fill_lds);
    let spv = spirv(&src);
    if let Some(dir) = &args.dump {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(format!("{dir}/repro.wgsl"), &src).unwrap();
        std::fs::write(format!("{dir}/repro.spv"), spv.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }

    let n = (args.groups * WG) as usize;
    let seeds: Vec<u32> = (0..n as u32).map(|i| hash(i ^ 0xA5A5_0000)).collect();
    let coef: Vec<u32> = (0..COEFS).map(|i| hash(i ^ 0x5A5A_0000) | 1).collect();
    let expected: Vec<u32> = (0..n).map(|i| reference(i as u32, seeds[i], &coef, args.vars, args.uniforms, args.iters)).collect();

    unsafe {
        let gpu = Gpu::new();
        let d = &gpu.device;
        println!(
            "device: {}  vars: {}  uniforms: {}  fill-lds: {}  iters: {}  invocations: {}  ACO_DEBUG={}",
            gpu.name,
            args.vars,
            args.uniforms,
            args.fill_lds,
            args.iters,
            n,
            std::env::var("ACO_DEBUG").unwrap_or_default()
        );

        let bytes = (n * 4) as u64;
        let (seed_buf, _m0, seed_ptr) = gpu.buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER);
        let (out_buf, _m1, out_ptr) = gpu.buffer(bytes, vk::BufferUsageFlags::STORAGE_BUFFER);
        let (par_buf, _m2, par_ptr) = gpu.buffer(16, vk::BufferUsageFlags::UNIFORM_BUFFER);
        let (coef_buf, _m3, coef_ptr) = gpu.buffer(COEFS as u64 * 4, vk::BufferUsageFlags::STORAGE_BUFFER);
        std::ptr::copy_nonoverlapping(coef.as_ptr(), coef_ptr, COEFS as usize);
        std::ptr::copy_nonoverlapping(seeds.as_ptr(), seed_ptr, n);
        std::ptr::copy_nonoverlapping([args.iters, USEED, 0, 0].as_ptr(), par_ptr, 4);

        let bindings = [
            (0, vk::DescriptorType::STORAGE_BUFFER),
            (1, vk::DescriptorType::STORAGE_BUFFER),
            (2, vk::DescriptorType::UNIFORM_BUFFER),
            (3, vk::DescriptorType::STORAGE_BUFFER),
        ]
        .map(|(b, t)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(t)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let dsl = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layouts = [dsl];
        let pl = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None).unwrap();
        let sm = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None).unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(sm).name(c"main");
        let pipe = d
            .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl)], None)
            .unwrap()[0];

        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
        ];
        let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).unwrap();
        let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).unwrap()[0];
        let infos = [seed_buf, out_buf, par_buf, coef_buf].map(|b| [vk::DescriptorBufferInfo { buffer: b, offset: 0, range: vk::WHOLE_SIZE }]);
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(bindings[i].descriptor_type)
                    .buffer_info(info)
            })
            .collect();
        d.update_descriptor_sets(&writes, &[]);

        let cpool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(gpu.qfam), None).unwrap();
        let cb = d
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cpool).command_buffer_count(1))
            .unwrap()[0];
        d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).unwrap();
        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
        d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pl, 0, &[set], &[]);
        d.cmd_dispatch(cb, args.groups, 1, 1);
        let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        d.end_command_buffer(cb).unwrap();

        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let mut failed_runs = 0;
        for run in 0..args.runs {
            std::ptr::write_bytes(out_ptr, 0, n);
            let cbs = [cb];
            d.queue_submit(gpu.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence).unwrap();
            d.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            d.reset_fences(&[fence]).unwrap();
            let got = std::slice::from_raw_parts(out_ptr, n);
            let bad: Vec<usize> = (0..n).filter(|&i| got[i] != expected[i]).collect();
            if bad.is_empty() {
                println!("run {run}: all {n} lanes match the CPU");
            } else {
                failed_runs += 1;
                let i = bad[0];
                println!(
                    "run {run}: {} of {n} lanes differ from the CPU (first: lane {i}, got {:#010x}, expected {:#010x})",
                    bad.len(),
                    got[i],
                    expected[i]
                );
            }
        }
        d.device_wait_idle().unwrap();
        println!("{failed_runs} of {} runs wrong", args.runs);
        std::process::exit(if failed_runs > 0 { 1 } else { 0 });
    }
}
