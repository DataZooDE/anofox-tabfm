//===----------------------------------------------------------------------===//
// tabfm_devices.cpp — device discovery, tabfm_devices(), device resolution
// (WS-C, HLD §9, SQL-API §3/§4)
//
// The cpu row always exists. The GPU probes are compiled into every build,
// because every build drives a GPU through a dlopen'd backend plugin (the ggml
// GPU plugin, MLX) — a build that cannot even see the card would make "one
// artifact, backends resolved at runtime" impossible.
// This costs no GPU dependency: NVML is dlopen'd by name and ROCm is read out
// of the KFD sysfs topology, so neither probe needs a CUDA or ROCm SDK header
// and the extension still links nothing vendor-specific (community-extension
// eligibility, HLD D9, is about what is LINKED, which is unchanged).
//===----------------------------------------------------------------------===//

#include "tabfm_session.hpp"
#include "tabfm_registration.hpp"
#include "anofox_function_alias.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb_compat.hpp"

#include "telemetry.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include "anofox_tabfm_banner.hpp"


#ifdef _WIN32
// NOMINMAX and WIN32_LEAN_AND_MEAN before <windows.h>: without them it defines
// min/max as macros (which collide with std::min/std::max and DuckDB's
// MinValue/MaxValue) and pulls in the winsock/RPC headers this file has no use
// for. Guarded rather than defined outright so a toolchain that already sets
// them does not warn about redefinition.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
// KFD topology lives under /sys and there is no Windows equivalent, so the rocm
// probe is POSIX-only rather than flavor-only.
#include <dirent.h>
#include <sys/stat.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace duckdb {
namespace anofox {

namespace {

//===----------------------------------------------------------------------===//
// CPU discovery
//===----------------------------------------------------------------------===//

string CpuModelName() {
#if defined(__linux__)
	std::ifstream cpuinfo("/proc/cpuinfo");
	string line;
	while (std::getline(cpuinfo, line)) {
		if (StringUtil::StartsWith(line, "model name")) {
			auto colon = line.find(':');
			if (colon != string::npos) {
				return StringUtil::Replace(line.substr(colon + 1), "\t", " ").substr(1);
			}
		}
	}
#elif defined(__APPLE__)
	char brand[256];
	size_t size = sizeof(brand);
	if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) {
		return string(brand);
	}
#endif
	return "cpu";
}

TabFMDeviceInfo MakeCpuDevice() {
	TabFMDeviceInfo device;
	device.device_id = "cpu";
	device.ep = "ggml";
	device.name = CpuModelName();
	device.arch = "";
	device.vram_total = -1;
	device.vram_free = -1;
	device.driver = "";
	device.usable = true;
	device.device_ordinal = 0;
	return device;
}


//===----------------------------------------------------------------------===//
// CUDA discovery (compiled into every build — see the file header)
//
// Device enumeration goes through NVML, loaded at runtime — NVML ships with every
// NVIDIA driver install and has a stable C ABI, so the extension links nothing
// NVIDIA-specific. No driver == no NVML == only the cpu row, which is exactly
// the "missing driver is a diagnosable one-liner" behavior HLD §9 asks for.
//
// Linux and Windows both; only the loader differs.
//===----------------------------------------------------------------------===//

struct NvmlMemory {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
};

// NVML is the same C ABI on both platforms; only the loader differs. It ships
// with every NVIDIA driver install — libnvidia-ml.so.1 on Linux, nvml.dll in
// System32 on Windows — so the extension still links nothing NVIDIA-specific.
#ifdef _WIN32
using NvmlHandle = HMODULE;
static const char *const kNvmlLibraries[] = {"nvml.dll"};
NvmlHandle NvmlOpen(const char *name) {
	return LoadLibraryA(name);
}
void *NvmlSymbol(NvmlHandle lib, const char *symbol) {
	return reinterpret_cast<void *>(GetProcAddress(lib, symbol));
}
void NvmlClose(NvmlHandle lib) {
	FreeLibrary(lib);
}
#else
using NvmlHandle = void *;
static const char *const kNvmlLibraries[] = {"libnvidia-ml.so.1", "libnvidia-ml.so"};
NvmlHandle NvmlOpen(const char *name) {
	return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}
void *NvmlSymbol(NvmlHandle lib, const char *symbol) {
	return dlsym(lib, symbol);
}
void NvmlClose(NvmlHandle lib) {
	dlclose(lib);
}
#endif

} // anonymous namespace

bool CudaComputeCapabilityUsable(int major, int minor) {
	return major * 10 + minor >= 75; // the plugin is built for sm_75 (Turing) and newer; an older card has no kernel image
}

namespace {

void ProbeCudaDevices(vector<TabFMDeviceInfo> &devices) {
	NvmlHandle nvml = nullptr;
	for (auto *candidate : kNvmlLibraries) {
		nvml = NvmlOpen(candidate);
		if (nvml) {
			break;
		}
	}
	if (!nvml) {
		return; // no NVIDIA driver -> no cuda rows; tabfm_devices() shows cpu only
	}

	using InitFn = int (*)();
	using CountFn = int (*)(unsigned int *);
	using HandleFn = int (*)(unsigned int, void **);
	using NameFn = int (*)(void *, char *, unsigned int);
	using MemFn = int (*)(void *, NvmlMemory *);
	using CcFn = int (*)(void *, int *, int *);
	using DriverFn = int (*)(char *, unsigned int);

	auto nvml_init = reinterpret_cast<InitFn>(NvmlSymbol(nvml, "nvmlInit_v2"));
	auto nvml_shutdown = reinterpret_cast<InitFn>(NvmlSymbol(nvml, "nvmlShutdown"));
	auto nvml_count = reinterpret_cast<CountFn>(NvmlSymbol(nvml, "nvmlDeviceGetCount_v2"));
	auto nvml_handle = reinterpret_cast<HandleFn>(NvmlSymbol(nvml, "nvmlDeviceGetHandleByIndex_v2"));
	auto nvml_name = reinterpret_cast<NameFn>(NvmlSymbol(nvml, "nvmlDeviceGetName"));
	auto nvml_memory = reinterpret_cast<MemFn>(NvmlSymbol(nvml, "nvmlDeviceGetMemoryInfo"));
	auto nvml_cc = reinterpret_cast<CcFn>(NvmlSymbol(nvml, "nvmlDeviceGetCudaComputeCapability"));
	auto nvml_driver = reinterpret_cast<DriverFn>(NvmlSymbol(nvml, "nvmlSystemGetDriverVersion"));

	if (!nvml_init || !nvml_count || !nvml_handle || nvml_init() != 0) {
		NvmlClose(nvml);
		return;
	}

	char driver_version[96] = {0};
	if (nvml_driver) {
		nvml_driver(driver_version, sizeof(driver_version));
	}

	unsigned int count = 0;
	if (nvml_count(&count) == 0) {
		for (unsigned int i = 0; i < count; i++) {
			void *handle = nullptr;
			if (nvml_handle(i, &handle) != 0) {
				continue;
			}
			TabFMDeviceInfo device;
			device.device_id = "cuda:" + std::to_string(i);
			device.ep = "ggml-cuda";
			device.device_ordinal = static_cast<int>(i);
			char name[96] = {0};
			if (nvml_name && nvml_name(handle, name, sizeof(name)) == 0) {
				device.name = name;
			}
			int cc_major = 0, cc_minor = 0;
			if (nvml_cc && nvml_cc(handle, &cc_major, &cc_minor) == 0) {
				device.arch = "sm_" + std::to_string(cc_major) + std::to_string(cc_minor);
			}
			NvmlMemory memory {};
			if (nvml_memory && nvml_memory(handle, &memory) == 0) {
				device.vram_total = static_cast<int64_t>(memory.total);
				device.vram_free = static_cast<int64_t>(memory.free);
			}
			device.driver = driver_version;
			// usable == the card can actually be driven, which since phase 3 no
			// longer means "the CUDA EP is compiled into this binary's ORT".
			// CUDA inference runs in a dlopen'd plugin carrying its OWN shared
			// ORT-GPU runtime (a statically-linked ORT cannot host ORT's provider
			// libraries at all), so a discovered card is usable whenever the
			// plugin ships kernels for it (compute capability 7.5+). If the
			// plugin turns out not to be installed, TryGgmlGpuBackend throws a
			// message naming the fix rather than quietly running on CPU.
			device.usable = CudaComputeCapabilityUsable(cc_major, cc_minor);
			devices.push_back(std::move(device));
		}
	}
	if (nvml_shutdown) {
		nvml_shutdown();
	}
	NvmlClose(nvml);
}


//===----------------------------------------------------------------------===//
// ROCm discovery
//
// Design: no ROCm library dependency at all — the kernel's KFD topology under
// /sys/class/kfd/kfd/topology/nodes/* is the source of truth (present iff the
// amdgpu KFD stack is up; /dev/kfd is the existence gate). Per node:
//   gfx_target_version = major*10000 + minor*100 + step  -> "gfx%d%x%x"
//   (0 or missing => CPU node, skipped)
//   mem_banks/*/properties: heap_type 1|2 (FB public|private) sizes summed
//   -> vram_total; vram_free stays NULL (needs the ROCm SMI library).
// `usable` = the released ggml ROCm plugin carries code objects for the gfx
// arch (GgmlRocmPluginArchs; HLD §9 support-matrix honesty: an unsupported
// card is refused by name, not chosen and then failing inside HIP). The KFD
// ordinal counts every GPU node, usable or not, because that is the HIP
// device index the plugin is handed.
//===----------------------------------------------------------------------===//

#ifndef _WIN32

bool ReadKfdProperty(const string &properties_path, const string &key, uint64_t &value) {
	std::ifstream file(properties_path);
	string line;
	while (std::getline(file, line)) {
		if (StringUtil::StartsWith(line, key + " ")) {
			value = std::strtoull(line.c_str() + key.size() + 1, nullptr, 10);
			return true;
		}
	}
	return false;
}

string GfxArchFromTargetVersion(uint64_t version) {
	const auto major = version / 10000;
	const auto minor = (version / 100) % 100;
	const auto step = version % 100;
	// gfx names use hex for minor/step: 9.0.10 -> gfx90a, 10.3.0 -> gfx1030
	char buffer[32];
	snprintf(buffer, sizeof(buffer), "gfx%llu%llx%llx", static_cast<unsigned long long>(major),
	         static_cast<unsigned long long>(minor), static_cast<unsigned long long>(step));
	return string(buffer);
}


int64_t SumVramBanks(const string &node_path) {
	int64_t total = 0;
	auto banks_path = node_path + "/mem_banks";
	DIR *dir = opendir(banks_path.c_str());
	if (!dir) {
		return -1;
	}
	while (auto *entry = readdir(dir)) {
		if (entry->d_name[0] == '.') {
			continue;
		}
		auto properties = banks_path + "/" + entry->d_name + "/properties";
		uint64_t heap_type = 0, size_in_bytes = 0;
		if (ReadKfdProperty(properties, "heap_type", heap_type) &&
		    ReadKfdProperty(properties, "size_in_bytes", size_in_bytes)) {
			// HSA heap types 1 (FB public) and 2 (FB private) are VRAM
			if (heap_type == 1 || heap_type == 2) {
				total += static_cast<int64_t>(size_in_bytes);
			}
		}
	}
	closedir(dir);
	return total > 0 ? total : -1;
}

string AmdgpuDriverVersion() {
	std::ifstream version_file("/sys/module/amdgpu/version");
	string version;
	if (version_file && std::getline(version_file, version)) {
		return version;
	}
	return "";
}

void ProbeRocmDevices(vector<TabFMDeviceInfo> &devices) {
	struct stat kfd_stat;
	if (stat("/dev/kfd", &kfd_stat) != 0) {
		return; // no KFD -> no ROCm rows
	}
	for (auto &device : DiscoverRocmDevicesFrom("/sys/class/kfd/kfd/topology/nodes", AmdgpuDriverVersion())) {
		devices.push_back(std::move(device));
	}
}

} // anonymous namespace

const vector<string> &GgmlRocmPluginArchs() {
	static const vector<string> archs = {"gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"};
	return archs;
}

bool RocmArchUsable(const string &arch) {
	const auto &archs = GgmlRocmPluginArchs();
	return std::find(archs.begin(), archs.end(), arch) != archs.end();
}

vector<TabFMDeviceInfo> DiscoverRocmDevicesFrom(const string &nodes_path, const string &driver) {
	vector<TabFMDeviceInfo> devices;
	DIR *dir = opendir(nodes_path.c_str());
	if (!dir) {
		return devices;
	}
	// Sort node ids for stable ordinals.
	vector<string> node_names;
	while (auto *entry = readdir(dir)) {
		if (entry->d_name[0] == '.') {
			continue;
		}
		node_names.push_back(entry->d_name);
	}
	closedir(dir);
	std::sort(node_names.begin(), node_names.end(),
	          [](const string &a, const string &b) { return std::stoll(a) < std::stoll(b); });

	int ordinal = 0;
	for (auto &node : node_names) {
		const string node_path = nodes_path + "/" + node;
		uint64_t gfx_target_version = 0;
		if (!ReadKfdProperty(node_path + "/properties", "gfx_target_version", gfx_target_version) ||
		    gfx_target_version == 0) {
			continue; // CPU node
		}
		uint64_t simd_count = 0;
		ReadKfdProperty(node_path + "/properties", "simd_count", simd_count);
		if (simd_count == 0) {
			continue; // not a compute-capable GPU node
		}
		TabFMDeviceInfo device;
		device.device_id = "rocm:" + std::to_string(ordinal);
		device.ep = "ggml-rocm";
		device.device_ordinal = ordinal;
		ordinal++;
		device.arch = GfxArchFromTargetVersion(gfx_target_version);
		std::ifstream name_file(node_path + "/name");
		std::getline(name_file, device.name);
		if (device.name.empty()) {
			device.name = device.arch;
		}
		device.vram_total = SumVramBanks(node_path);
		device.vram_free = -1; // needs rocm_smi; NULL is honest
		device.driver = driver;
		device.usable = RocmArchUsable(device.arch); // the plugin carries code objects for these architectures only
		devices.push_back(std::move(device));
	}
	return devices;
}

namespace {

#endif // !_WIN32

//===----------------------------------------------------------------------===//
// Apple MLX discovery (docs/MLX_PLAN.md)
//
// Unlike the CUDA and ROCm probes there is no driver to interrogate and no SDK
// to dlopen: MLX runs on the SoC's own GPU, so "is there hardware" reduces to
// "is this Apple Silicon". That is a compile-time platform test plus a runtime
// SoC-name check, which is why this probe is compiled into every build rather
// than platform-gated -- the plugin that does the work is fetched separately, but
// discovery must be able to say the hardware is there.
//
// One logical `mlx:0` row. Rosetta is the case worth being careful
// about: an x86_64 process on an Apple Silicon machine reads an Apple SoC name
// while being unable to load an arm64 plugin, so usable is gated on the
// compiled architecture, not just the reported chip.
//===----------------------------------------------------------------------===//
#if defined(__APPLE__)
void ProbeMlxDevices(vector<TabFMDeviceInfo> &devices) {
	const string soc = CpuModelName(); // e.g. "Apple M3"
	const bool apple_silicon = StringUtil::Contains(StringUtil::Lower(soc), "apple");
#if defined(__aarch64__) || defined(__arm64__)
	const bool arm_process = true;
#else
	const bool arm_process = false; // x86_64 build (or under Rosetta): no arm64 plugin
#endif
	if (!apple_silicon) {
		return; // Intel Mac -> no mlx row at all; there is no such hardware
	}
	TabFMDeviceInfo device;
	device.device_id = "mlx:0";
	device.ep = "mlx";
	device.name = soc;
	device.arch = soc;
	device.vram_total = -1; // unified memory; no discrete VRAM semantics
	device.vram_free = -1;
	device.driver = "";
	device.usable = arm_process;
	device.device_ordinal = 0;
	devices.push_back(std::move(device));
}
#endif // __APPLE__

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Discovery + resolution (declared in tabfm_session.hpp)
//===----------------------------------------------------------------------===//

vector<TabFMDeviceInfo> DiscoverDevices() {
	vector<TabFMDeviceInfo> devices;
	devices.push_back(MakeCpuDevice());
	ProbeCudaDevices(devices);
#ifndef _WIN32
	ProbeRocmDevices(devices);
#endif
#if defined(__APPLE__)
	ProbeMlxDevices(devices);
#endif
	return devices;
}

TabFMDeviceInfo ResolveDevice(const string &setting_value, const vector<TabFMDeviceInfo> &devices) {
	auto setting = StringUtil::Lower(setting_value);
	if (setting == "migraphx") { // validator normalizes, but stay forgiving
		setting = "rocm";
	}
	auto cpu_device = [&]() {
		for (auto &device : devices) {
			if (device.device_id == "cpu") {
				return device;
			}
		}
		return MakeCpuDevice();
	};
	if (setting == "cpu" || setting == "auto") {
		// 'auto' picks a GPU per model (ResolveDeviceForModel in tabfm_engine.cpp), and only where its plugin is installed and serves that model; the
		// session-wide answer is the CPU, which every model runs on.
		return cpu_device();
	}
	if (setting == "coreml") {
		// CoreML was an ONNX Runtime execution provider compiled into the extension; ONNX Runtime is gone, and MLX (a plugin) serves Apple GPUs.
		throw InvalidInputException("anofox_tabfm: device 'coreml' is no longer available: it was an ONNX Runtime provider, and this build contains no "
		                            "ONNX Runtime. On Apple Silicon use 'mlx': CALL tabfm_download_runtime('mlx'); SET anofox_tabfm_device='mlx'. "
		                            "Otherwise SET anofox_tabfm_device='cpu'.");
	}
	if (setting == "cuda" || setting == "rocm" || setting == "mlx") {
		// Every build carries every GPU lane: each runs in a dlopen'd plugin (the ggml GPU plugin for cuda and rocm, MLX for Apple), built and shipped
		// separately. A missing plugin is reported by the lane that loads it, naming the fix (TryGgmlGpuBackend, TryMlxBackend).
		for (auto &device : devices) {
			if (StringUtil::StartsWith(device.device_id, setting) && device.usable) {
				return device;
			}
		}
		// A card that is there but unusable is the common case worth explaining: say which, and what the plugin runs on.
		string found;
		for (auto &device : devices) {
			if (StringUtil::StartsWith(device.device_id, setting)) {
				found += (found.empty() ? "" : ", ") + device.device_id + " (" + (device.arch.empty() ? device.name : device.arch) + ")";
			}
		}
		if (!found.empty() && setting != "mlx") {
			const string supported = setting == "cuda" ? "compute capability 7.5 or newer (sm_75+)"
			                                           : "the architectures " + StringUtil::Join(GgmlRocmPluginArchs(), ", ");
			throw InvalidInputException("anofox_tabfm: no usable '" + setting + "' device: found " + found +
			                            ", which the ggml GPU plugin has no kernels for; it supports " + supported +
			                            ". SELECT * FROM tabfm_devices(); shows every device found; SET anofox_tabfm_device='cpu' (or 'auto') to run "
			                            "on the CPU.");
		}
		throw InvalidInputException("anofox_tabfm: no usable '" + setting +
		                            "' device was discovered — check SELECT * FROM tabfm_devices(); for the "
		                            "detected hardware/driver state, or SET anofox_tabfm_device='cpu'");
	}
	throw InvalidInputException("anofox_tabfm: unknown device setting '" + setting_value +
	                            "' — anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm', 'mlx'");
}


//===----------------------------------------------------------------------===//
// tabfm_devices() table function (SQL-API §3)
//===----------------------------------------------------------------------===//

namespace {

struct DevicesBindData : public TableFunctionData {};

struct DevicesGlobalState : public GlobalTableFunctionState {
	vector<TabFMDeviceInfo> devices;
	idx_t offset = 0;
};

unique_ptr<FunctionData> DevicesBind(ClientContext &context, TableFunctionBindInput &input,
                                     vector<LogicalType> &return_types, duckdb_compat::BindNames &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_devices");

	names = {"device_id", "ep", "name", "arch", "vram_total", "vram_free", "driver", "usable"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::BOOLEAN};
	return make_uniq<DevicesBindData>();
}

unique_ptr<GlobalTableFunctionState> DevicesInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<DevicesGlobalState>();
	state->devices = DiscoverDevices();
	return std::move(state);
}

void DevicesFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<DevicesGlobalState>();
	idx_t count = 0;
	while (state.offset < state.devices.size() && count < STANDARD_VECTOR_SIZE) {
		auto &device = state.devices[state.offset++];
		output.SetValue(0, count, Value(device.device_id));
		output.SetValue(1, count, Value(device.ep));
		output.SetValue(2, count, Value(device.name));
		output.SetValue(3, count, Value(device.arch));
		output.SetValue(4, count, device.vram_total < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(device.vram_total));
		output.SetValue(5, count, device.vram_free < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(device.vram_free));
		output.SetValue(6, count, Value(device.driver));
		output.SetValue(7, count, Value::BOOLEAN(device.usable));
		count++;
	}
	duckdb_compat::SetChildCardinality(output, count);
}

} // anonymous namespace

void RegisterDevicesFunctions(ExtensionLoader &loader) {
	TableFunction func("anofox_tabfm_devices", {}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, DevicesFunction),
	                   DATAZOO_GUARD(ANOFOX_TABFM_BANNER, DevicesBind), DevicesInitGlobal);
	FunctionDescription fd;
	fd.description =
	    "List the inference devices this build can see (device_id, ep, name, arch, vram, driver, usable). The cpu "
	    "row always exists; a GPU row appears for every NVIDIA (NVML), AMD (KFD) or Apple Silicon GPU found, with "
	    "usable=false when the GPU plugin has no kernels for it; ep names the engine that serves it.";
	fd.examples = {"SELECT * FROM tabfm_devices();"};
	fd.categories = {"tabfm", "meta"};
	RegisterTableFunctionWithAlias(loader, std::move(func), "tabfm_devices", {std::move(fd)});
}

} // namespace anofox
} // namespace duckdb
