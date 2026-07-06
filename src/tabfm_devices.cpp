//===----------------------------------------------------------------------===//
// tabfm_devices.cpp — device discovery, tabfm_devices(), device resolution,
// MIGraphX shape buckets (WS-C, HLD §9, SQL-API §3/§4)
//
// The cpu row always exists. GPU probes are compiled only into their flavor
// (TABFM_EP_CUDA / TABFM_EP_MIGRAPHX) so the cpu flavor carries zero GPU code
// paths (community-extension eligibility, HLD D9).
//===----------------------------------------------------------------------===//

#include "tabfm_ort_engine.hpp"
#include "tabfm_registration.hpp"
#include "anofox_function_alias.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"

#include "telemetry.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>

#if defined(TABFM_EP_CUDA) || defined(TABFM_EP_MIGRAPHX) || defined(TABFM_EP_COREML)
#include <onnxruntime_cxx_api.h>
#endif

#if defined(TABFM_EP_CUDA) && !defined(_WIN32)
#include <dlfcn.h>
#endif

#ifdef TABFM_EP_MIGRAPHX
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
	device.ep = "CPUExecutionProvider";
	device.name = CpuModelName();
	device.arch = "";
	device.vram_total = -1;
	device.vram_free = -1;
	device.driver = "";
	device.usable = true;
	device.device_ordinal = 0;
	return device;
}

#if defined(TABFM_EP_CUDA) || defined(TABFM_EP_MIGRAPHX) || defined(TABFM_EP_COREML)
bool OrtProviderAvailable(const char *provider_name) {
	for (auto &provider : Ort::GetAvailableProviders()) {
		if (provider == provider_name) {
			return true;
		}
	}
	return false;
}
#endif

//===----------------------------------------------------------------------===//
// CUDA discovery (cuda flavor only)
//
// Design: EP presence comes from Ort::GetAvailableProviders(); device
// enumeration goes through NVML (libnvidia-ml.so.1), dlopen'd at runtime —
// NVML ships with every NVIDIA driver install and has a stable C ABI, so the
// extension links nothing NVIDIA-specific. No driver == no NVML == only the
// cpu row, which is exactly the "missing driver is a diagnosable one-liner"
// behavior HLD §9 asks for.
//===----------------------------------------------------------------------===//

#if defined(TABFM_EP_CUDA) && !defined(_WIN32)

struct NvmlMemory {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
};

void ProbeCudaDevices(vector<TabFMDeviceInfo> &devices) {
	const bool ep_available = OrtProviderAvailable("CUDAExecutionProvider");

	void *nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
	if (!nvml) {
		nvml = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
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

	auto nvml_init = reinterpret_cast<InitFn>(dlsym(nvml, "nvmlInit_v2"));
	auto nvml_shutdown = reinterpret_cast<InitFn>(dlsym(nvml, "nvmlShutdown"));
	auto nvml_count = reinterpret_cast<CountFn>(dlsym(nvml, "nvmlDeviceGetCount_v2"));
	auto nvml_handle = reinterpret_cast<HandleFn>(dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2"));
	auto nvml_name = reinterpret_cast<NameFn>(dlsym(nvml, "nvmlDeviceGetName"));
	auto nvml_memory = reinterpret_cast<MemFn>(dlsym(nvml, "nvmlDeviceGetMemoryInfo"));
	auto nvml_cc = reinterpret_cast<CcFn>(dlsym(nvml, "nvmlDeviceGetCudaComputeCapability"));
	auto nvml_driver = reinterpret_cast<DriverFn>(dlsym(nvml, "nvmlSystemGetDriverVersion"));

	if (!nvml_init || !nvml_count || !nvml_handle || nvml_init() != 0) {
		dlclose(nvml);
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
			device.ep = "CUDAExecutionProvider";
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
			// usable == the CUDA EP is actually registered in this ORT build;
			// a discovered card without the EP (or vice versa) stays diagnosable.
			device.usable = ep_available;
			devices.push_back(std::move(device));
		}
	}
	if (nvml_shutdown) {
		nvml_shutdown();
	}
	dlclose(nvml);
}

#endif // TABFM_EP_CUDA && !_WIN32

//===----------------------------------------------------------------------===//
// ROCm / MIGraphX discovery (rocm flavor only)
//
// Design: no ROCm library dependency at all — the kernel's KFD topology under
// /sys/class/kfd/kfd/topology/nodes/* is the source of truth (present iff the
// amdgpu KFD stack is up; /dev/kfd is the existence gate). Per node:
//   gfx_target_version = major*10000 + minor*100 + step  -> "gfx%d%x%x"
//   (0 or missing => CPU node, skipped)
//   mem_banks/*/properties: heap_type 1|2 (FB public|private) sizes summed
//   -> vram_total; vram_free stays NULL (needs the ROCm SMI library).
// `usable` = MIGraphX EP registered in this ORT build AND MIGraphX claims the
// gfx arch (HLD §9 support-matrix honesty: unsupported consumer cards fail
// explicitly, not mysteriously).
//===----------------------------------------------------------------------===//

#ifdef TABFM_EP_MIGRAPHX

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

//! MIGraphX official coverage (state 2026-07): CDNA data-center cards and
//! recent Radeon on Linux. Kept intentionally explicit — an unsupported gfx
//! arch shows up as usable=false in tabfm_devices() with the arch named.
bool MIGraphXClaimsArch(const string &gfx) {
	static const char *SUPPORTED[] = {"gfx900",  "gfx906",  "gfx908",  "gfx90a",  "gfx940",  "gfx941", "gfx942",
	                                  "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"};
	for (auto *arch : SUPPORTED) {
		if (gfx == arch) {
			return true;
		}
	}
	return false;
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
	const bool ep_available = OrtProviderAvailable("MIGraphXExecutionProvider");
	const string driver = AmdgpuDriverVersion();

	const string nodes_path = "/sys/class/kfd/kfd/topology/nodes";
	DIR *dir = opendir(nodes_path.c_str());
	if (!dir) {
		return;
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
		device.ep = "MIGraphXExecutionProvider";
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
		device.usable = ep_available && MIGraphXClaimsArch(device.arch);
		devices.push_back(std::move(device));
	}
}

#endif // TABFM_EP_MIGRAPHX

//===----------------------------------------------------------------------===//
// CoreML discovery (coreml flavor only)
//
// The CoreML EP is compiled into the prebuilt osx-universal2 ORT core, so its
// availability is a build fact, not a driver probe. We emit a single logical
// `coreml:0` row for the Apple SoC. `usable` is honest: true only on Apple
// Silicon (arm64, where the ANE/GPU actually accelerate) with the EP registered
// — an Intel Mac reports the row usable=false (CoreML would run CPU-only there).
//===----------------------------------------------------------------------===//
#ifdef TABFM_EP_COREML
void ProbeCoreMLDevices(vector<TabFMDeviceInfo> &devices) {
	const bool ep_available = OrtProviderAvailable("CoreMLExecutionProvider");
	const string soc = CpuModelName(); // e.g. "Apple M3"
	const bool apple_silicon = StringUtil::Contains(StringUtil::Lower(soc), "apple");

	TabFMDeviceInfo device;
	device.device_id = "coreml:0";
	device.ep = "CoreMLExecutionProvider";
	device.name = soc;
	device.arch = soc;
	device.vram_total = -1; // unified memory; no discrete VRAM semantics
	device.vram_free = -1;
	device.driver = "";
	device.usable = ep_available && apple_silicon;
	device.device_ordinal = 0;
	devices.push_back(std::move(device));
}
#endif // TABFM_EP_COREML

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Discovery + resolution (declared in tabfm_ort_engine.hpp)
//===----------------------------------------------------------------------===//

vector<TabFMDeviceInfo> DiscoverDevices() {
	vector<TabFMDeviceInfo> devices;
	devices.push_back(MakeCpuDevice());
#if defined(TABFM_EP_CUDA) && !defined(_WIN32)
	ProbeCudaDevices(devices);
#endif
#ifdef TABFM_EP_MIGRAPHX
	ProbeRocmDevices(devices);
#endif
#ifdef TABFM_EP_COREML
	ProbeCoreMLDevices(devices);
#endif
	return devices;
}

TabFMDeviceInfo ResolveDevice(const string &setting_value, const vector<TabFMDeviceInfo> &devices, bool flavor_has_cuda,
                              bool flavor_has_rocm, bool flavor_has_coreml) {
	auto setting = StringUtil::Lower(setting_value);
	if (setting == "migraphx") { // validator normalizes, but stay forgiving
		setting = "rocm";
	}

	// Flavor name as implied by the capability flags (== compiled flavor in
	// production; explicit flags keep this a pure, unit-testable function).
	const string flavor = flavor_has_cuda ? "cuda" : (flavor_has_rocm ? "rocm" : (flavor_has_coreml ? "coreml" : "cpu"));

	auto find_usable = [&](const string &prefix) -> const TabFMDeviceInfo * {
		for (auto &device : devices) {
			if (StringUtil::StartsWith(device.device_id, prefix) && device.usable) {
				return &device;
			}
		}
		return nullptr;
	};

	if (setting == "cpu") {
		for (auto &device : devices) {
			if (device.device_id == "cpu") {
				return device;
			}
		}
		return MakeCpuDevice();
	}
	if (setting == "cuda" || setting == "rocm" || setting == "coreml") {
		const bool carried =
		    setting == "cuda" ? flavor_has_cuda : (setting == "rocm" ? flavor_has_rocm : flavor_has_coreml);
		if (!carried) {
			throw InvalidInputException(
			    "anofox_tabfm: this build is the '" + flavor + "' flavor and does not carry '" + setting +
			    "'; install the '" + setting +
			    "' flavor from the anofox extension repository (SET custom_extension_repository = "
			    "'https://ext.anofox.com/tabfm/" +
			    setting + "') or SET anofox_tabfm_device='cpu'");
		}
		if (auto *device = find_usable(setting)) {
			return *device;
		}
		throw InvalidInputException("anofox_tabfm: no usable '" + setting +
		                            "' device was discovered — check SELECT * FROM tabfm_devices(); for the "
		                            "detected hardware/driver state, or SET anofox_tabfm_device='cpu'");
	}
	if (setting == "auto") {
		// Best discovered device: the flavor's GPU lane if a usable device
		// exists, else the cpu fallback lane (HLD §9).
		if (flavor_has_cuda) {
			if (auto *device = find_usable("cuda")) {
				return *device;
			}
		}
		if (flavor_has_rocm) {
			if (auto *device = find_usable("rocm")) {
				return *device;
			}
		}
		if (flavor_has_coreml) {
			if (auto *device = find_usable("coreml")) {
				return *device;
			}
		}
		for (auto &device : devices) {
			if (device.device_id == "cpu") {
				return device;
			}
		}
		return MakeCpuDevice();
	}
	throw InvalidInputException("anofox_tabfm: unknown device setting '" + setting_value +
	                            "' — anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm', 'coreml'");
}

//===----------------------------------------------------------------------===//
// MIGraphX shape buckets (HLD §9)
//===----------------------------------------------------------------------===//

TabFMShapeBucket MIGraphXShapeBucket(int64_t rows_t, int64_t features_h) {
	if (rows_t <= 0 || features_h <= 0) {
		throw InvalidInputException("anofox_tabfm: shape bucket needs positive dimensions, got T=" +
		                            std::to_string(rows_t) + " H=" + std::to_string(features_h));
	}
	static constexpr int64_t T_BUCKETS[] = {128, 512, 1024, 2048, 4096, 10000};
	static constexpr int64_t H_BUCKETS[] = {16, 64, 128, 256, 512};

	auto pad_up = [](int64_t value, const int64_t *buckets, size_t count) {
		for (size_t i = 0; i < count; i++) {
			if (value <= buckets[i]) {
				return buckets[i];
			}
		}
		// Above the largest bucket: return unchanged; callers guard via the
		// anofox_tabfm_max_rows / anofox_tabfm_max_features settings.
		return value;
	};

	TabFMShapeBucket bucket;
	bucket.padded_t = pad_up(rows_t, T_BUCKETS, sizeof(T_BUCKETS) / sizeof(T_BUCKETS[0]));
	bucket.padded_h = pad_up(features_h, H_BUCKETS, sizeof(H_BUCKETS) / sizeof(H_BUCKETS[0]));
	return bucket;
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
                                     vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_devices");

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
	output.SetCardinality(count);
}

} // anonymous namespace

void RegisterDevicesFunctions(ExtensionLoader &loader) {
	TableFunction func("anofox_tabfm_devices", {}, DevicesFunction, DevicesBind, DevicesInitGlobal);
	FunctionDescription fd;
	fd.description =
	    "List the inference devices this build can see (device_id, ep, name, arch, vram, driver, usable). The cpu "
	    "row always exists; GPU rows appear only in the matching flavor (cuda/rocm) and report usable=false when a "
	    "device is present but unsupported.";
	fd.examples = {"SELECT * FROM tabfm_devices();"};
	RegisterTableFunctionWithAlias(loader, std::move(func), "tabfm_devices", {std::move(fd)});
}

} // namespace anofox
} // namespace duckdb
