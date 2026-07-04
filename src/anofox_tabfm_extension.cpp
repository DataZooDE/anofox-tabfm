#define DUCKDB_EXTENSION_MAIN

#include "anofox_tabfm_extension.hpp"
#include "tabfm_registration.hpp"
#include "telemetry.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

#include <cstdlib>

namespace duckdb {

namespace {

// True if the DATAZOO_DISABLE_TELEMETRY environment variable requests an opt-out.
// This mirrors the check posthog-telemetry performs at send time, but evaluating it
// here lets us disable the singleton before any event is enqueued (and before the
// background queue thread is ever started).
bool IsTelemetryDisabledByEnv() {
	const char *env = std::getenv("DATAZOO_DISABLE_TELEMETRY");
	if (!env) {
		return false;
	}
	const std::string value(env);
	return value == "1" || value == "true" || value == "yes";
}

// Telemetry is pointless (and noisy) on CI runners.
bool IsRunningInCI() {
	for (const char *var : {"CI", "GITHUB_ACTIONS", "GITLAB_CI", "TRAVIS", "CIRCLECI", "JENKINS_URL"}) {
		if (std::getenv(var)) {
			return true;
		}
	}
	return false;
}

void OnTelemetryEnabled(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_telemetry_enabled cannot be NULL");
	}
	auto &telemetry = PostHogTelemetry::Instance();
	telemetry.SetEnabled(BooleanValue::Get(parameter));
}

void OnTelemetryKey(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_telemetry_key cannot be NULL");
	}
	auto &telemetry = PostHogTelemetry::Instance();
	telemetry.SetAPIKey(StringValue::Get(parameter));
}

void RegisterTelemetryOptions(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	// The default reflects the effective pre-load opt-outs so
	// `current_setting('anofox_telemetry_enabled')` is honest about the state.
	config.AddExtensionOption("anofox_telemetry_enabled", "Enable or disable anonymous usage telemetry",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(!IsTelemetryDisabledByEnv() && !IsRunningInCI()),
	                          OnTelemetryEnabled);

	config.AddExtensionOption("anofox_telemetry_key", "PostHog API key for telemetry", LogicalType::VARCHAR,
	                          Value("phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t"), OnTelemetryKey);
}

} // anonymous namespace

static void LoadInternal(ExtensionLoader &loader) {
	loader.SetDescription("Zero-shot classification and regression on tabular data via the TabFM "
	                      "foundation model (TabPFN-style in-context learning): tabfm_predict over any "
	                      "relation, SQL-managed weights, CPU/CUDA/ROCm execution.");

	// Register telemetry options first so they exist before any event is emitted.
	RegisterTelemetryOptions(loader);

	// Determine the effective telemetry opt-out BEFORE emitting the load event.
	// SQL `SET anofox_telemetry_enabled = false` can only run after the extension
	// is loaded, so it cannot suppress this event; the pre-load opt-outs are:
	//   - the DATAZOO_DISABLE_TELEMETRY environment variable,
	//   - a CI environment,
	//   - pre-setting anofox_telemetry_enabled through DBConfig / client config
	//     (with allow_unrecognized_options).
	auto &db = loader.GetDatabaseInstance();
	bool telemetry_enabled = !IsTelemetryDisabledByEnv() && !IsRunningInCI();
	if (telemetry_enabled) {
		Value enabled_value;
		if (db.TryGetCurrentSetting("anofox_telemetry_enabled", enabled_value) && !enabled_value.IsNull()) {
			telemetry_enabled = BooleanValue::Get(enabled_value.DefaultCastAs(LogicalType::BOOLEAN));
		}
	}

	auto &telemetry = PostHogTelemetry::Instance();
	telemetry.SetEnabled(telemetry_enabled);

	std::string version;
#ifdef EXT_VERSION_ANOFOX_TABFM
	version = EXT_VERSION_ANOFOX_TABFM;
#else
	version = "0.1.0";
#endif
	if (telemetry_enabled) {
		Value key_value;
		if (db.TryGetCurrentSetting("anofox_telemetry_key", key_value) && !key_value.IsNull()) {
			telemetry.SetAPIKey(StringValue::Get(key_value.DefaultCastAs(LogicalType::VARCHAR)));
		}
		telemetry.CaptureExtensionLoad("anofox_tabfm", version);
	} else {
		// Still record the extension name so later re-enabling via SQL produces
		// correctly attributed function-execution events.
		telemetry.SetExtensionName("anofox_tabfm");
	}

	anofox::RegisterTabfmSettings(loader);
	anofox::RegisterWeightsFunctions(loader);
	anofox::RegisterDevicesFunctions(loader);
	anofox::RegisterPredictAggFunction(loader);
	anofox::RegisterPredictMacros(loader);
}

void AnofoxTabfmExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string AnofoxTabfmExtension::Name() {
	return "anofox_tabfm";
}

std::string AnofoxTabfmExtension::Version() const {
#ifdef EXT_VERSION_ANOFOX_TABFM
	return EXT_VERSION_ANOFOX_TABFM;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_tabfm, loader) {
	duckdb::LoadInternal(loader);
}
}
