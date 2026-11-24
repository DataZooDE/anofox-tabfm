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
#include "anofox_ggml_adapter/devices.hpp"
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
// probe is POSIX-only.
#include <dirent.h>
#include <sys/stat.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace duckdb {
namespace anofox {

namespace {

namespace ga = ::anofox::ggml_adapter;

TabFMDeviceInfo FromAdapter(const ga::DeviceInfo &d) {
	TabFMDeviceInfo device;
	device.device_id = d.device_id;
	device.ep = d.backend;
	device.name = d.name;
	device.arch = d.arch;
	device.vram_total = d.vram_total;
	device.vram_free = d.vram_free;
	device.driver = d.driver;
	device.usable = d.usable;
	device.device_ordinal = d.device_ordinal;
	return device;
}

TabFMDeviceInfo MakeCpuDevice() {
	for (const auto &d : ga::DiscoverDevices()) {
		if (d.device_id == "cpu") {
			return FromAdapter(d);
		}
	}
	TabFMDeviceInfo cpu;
	cpu.device_id = "cpu";
	cpu.ep = "ggml";
	cpu.name = "cpu";
	cpu.usable = true;
	return cpu;
}

} // namespace

//===----------------------------------------------------------------------===//
// Discovery (anofox-ggml-adapter's devices.hpp: NVML dlopened, the KFD topology, Apple Silicon; no vendor SDK), declared in
// tabfm_session.hpp. Which cards are usable is the plugins' property, so it comes from the adapter too.
//===----------------------------------------------------------------------===//

bool CudaComputeCapabilityUsable(int major, int minor) {
	return ga::CudaComputeCapabilityUsable(major, minor);
}

const vector<string> &GgmlRocmPluginArchs() {
	static const vector<string> archs(ga::GgmlRocmPluginArchs().begin(), ga::GgmlRocmPluginArchs().end());
	return archs;
}

bool RocmArchUsable(const string &arch) {
	return ga::RocmArchUsable(arch);
}

vector<TabFMDeviceInfo> DiscoverRocmDevicesFrom(const string &nodes_path, const string &driver) {
	vector<TabFMDeviceInfo> devices;
	for (const auto &d : ga::DiscoverRocmDevicesFrom(nodes_path, driver)) {
		devices.push_back(FromAdapter(d));
	}
	return devices;
}

vector<TabFMDeviceInfo> DiscoverDevices() {
	vector<TabFMDeviceInfo> devices;
	for (const auto &d : ga::DiscoverDevices()) {
		devices.push_back(FromAdapter(d));
	}
	return devices;
}

TabFMDeviceInfo ResolveDevice(const string &setting_value, const vector<TabFMDeviceInfo> &devices) {
	auto setting = NormalizeDeviceSetting(setting_value);
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
		// CoreML (deprecated) was an ONNX Runtime execution provider compiled into the extension; ONNX Runtime was removed, and MLX (a plugin) serves Apple GPUs.
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
