#include "catch.hpp"

#include "tabfm_cpu_budget.hpp"

#ifdef __linux__
#include <sched.h>
#endif

using namespace duckdb;
using namespace duckdb::anofox;

// cgroup v2 exposes both numbers in one file: "<quota> <period>", or "max <period>"
// when uncapped. A 64-core cap on a 100ms period is "6400000 100000".
TEST_CASE("cpu_budget: cgroup v2 cpu.max", "[cpu_budget][tabfm]") {
	SECTION("uncapped reports no limit") {
		REQUIRE(ParseCpuMaxV2("max 100000") == 0);
		REQUIRE(ParseCpuMaxV2("max 100000\n") == 0);
	}
	SECTION("whole cores") {
		REQUIRE(ParseCpuMaxV2("6400000 100000") == 64);
		REQUIRE(ParseCpuMaxV2("100000 100000") == 1);
		REQUIRE(ParseCpuMaxV2("400000 100000\n") == 4);
	}
	SECTION("fractional quota rounds up, never to zero") {
		// 0.5 cores still permits one runnable thread; flooring would yield 0 and
		// then get clamped anyway, so round up and keep the intent visible.
		REQUIRE(ParseCpuMaxV2("50000 100000") == 1);
		REQUIRE(ParseCpuMaxV2("150000 100000") == 2);
		REQUIRE(ParseCpuMaxV2("1 100000") == 1);
	}
	SECTION("malformed input reports no limit rather than guessing") {
		REQUIRE(ParseCpuMaxV2("") == 0);
		REQUIRE(ParseCpuMaxV2("garbage") == 0);
		REQUIRE(ParseCpuMaxV2("100000") == 0);
		REQUIRE(ParseCpuMaxV2("abc def") == 0);
		REQUIRE(ParseCpuMaxV2("100000 0") == 0);
		REQUIRE(ParseCpuMaxV2("-1 100000") == 0);
		REQUIRE(ParseCpuMaxV2("100000 -5") == 0);
	}
}

// cgroup v1 splits them across two files, and spells "uncapped" as quota -1.
TEST_CASE("cpu_budget: cgroup v1 cpu.cfs_quota_us", "[cpu_budget][tabfm]") {
	SECTION("uncapped reports no limit") {
		REQUIRE(ParseCpuQuotaV1("-1", "100000") == 0);
		REQUIRE(ParseCpuQuotaV1("-1\n", "100000\n") == 0);
	}
	SECTION("whole cores") {
		REQUIRE(ParseCpuQuotaV1("6400000", "100000") == 64);
		REQUIRE(ParseCpuQuotaV1("200000", "100000") == 2);
	}
	SECTION("fractional quota rounds up") {
		REQUIRE(ParseCpuQuotaV1("50000", "100000") == 1);
		REQUIRE(ParseCpuQuotaV1("250000", "100000") == 3);
	}
	SECTION("malformed input reports no limit") {
		REQUIRE(ParseCpuQuotaV1("", "100000") == 0);
		REQUIRE(ParseCpuQuotaV1("100000", "") == 0);
		REQUIRE(ParseCpuQuotaV1("100000", "0") == 0);
		REQUIRE(ParseCpuQuotaV1("abc", "100000") == 0);
		REQUIRE(ParseCpuQuotaV1("0", "100000") == 0);
	}
}

// 0 means "this mechanism imposes no limit", which must not win a min().
TEST_CASE("cpu_budget: combining two limits", "[cpu_budget][tabfm]") {
	REQUIRE(CombineCoreLimits(0, 0) == 0);
	REQUIRE(CombineCoreLimits(8, 0) == 8);
	REQUIRE(CombineCoreLimits(0, 8) == 8);
	REQUIRE(CombineCoreLimits(8, 4) == 4);
	REQUIRE(CombineCoreLimits(4, 8) == 4);
	REQUIRE(CombineCoreLimits(4, 4) == 4);
}

// `/sys/fs/cgroup/cpu.max` is the root cgroup and carries no limit; the limit that binds us
// lives on our own cgroup, or on an ancestor when the cgroup namespace is the host's.
TEST_CASE("cpu_budget: locating our own cgroup", "[cpu_budget][tabfm]") {
	SECTION("unified hierarchy is the line with an empty controller field") {
		REQUIRE(ParseUnifiedCgroupPath("0::/user.slice/session-1.scope") ==
		        "/user.slice/session-1.scope");
		REQUIRE(ParseUnifiedCgroupPath("0::/\n") == "/");
		// Containers under a private namespace see themselves at the root.
		REQUIRE(ParseUnifiedCgroupPath("0::/") == "/");
	}
	SECTION("hybrid layouts: the unified line is found among v1 lines") {
		const std::string hybrid = "5:cpu,cpuacct:/kubepods/pod123\n"
		                           "4:memory:/kubepods/pod123\n"
		                           "0::/kubepods/pod123\n";
		REQUIRE(ParseUnifiedCgroupPath(hybrid) == "/kubepods/pod123");
	}
	SECTION("pure v1 has no unified line") {
		REQUIRE(ParseUnifiedCgroupPath("5:cpu,cpuacct:/kubepods/pod123\n").empty());
		REQUIRE(ParseUnifiedCgroupPath("").empty());
		REQUIRE(ParseUnifiedCgroupPath("garbage").empty());
	}
	SECTION("v1 cpu controller is matched exactly, not by prefix") {
		REQUIRE(ParseLegacyCpuCgroupPath("5:cpu,cpuacct:/kubepods/pod123") == "/kubepods/pod123");
		REQUIRE(ParseLegacyCpuCgroupPath("5:cpuacct:/kubepods/pod123").empty());
		REQUIRE(ParseLegacyCpuCgroupPath("3:cpuset:/kubepods").empty());
		REQUIRE(ParseLegacyCpuCgroupPath("9:cpuacct,cpu:/a/b") == "/a/b");
		REQUIRE(ParseLegacyCpuCgroupPath("0::/user.slice").empty());
		REQUIRE(ParseLegacyCpuCgroupPath("").empty());
	}
}

TEST_CASE("cpu_budget: UsableCoreCount is always usable", "[cpu_budget][tabfm]") {
	// Whatever the host, the count has to be something a thread pool can be sized
	// from -- the setting default divides it and clamps at 1.
	REQUIRE(UsableCoreCount() >= 1);
}

#ifdef __linux__
// The whole point of the change: the count follows the cpuset, not the host.
TEST_CASE("cpu_budget: UsableCoreCount follows the affinity mask", "[cpu_budget][tabfm]") {
	cpu_set_t original;
	CPU_ZERO(&original);
	REQUIRE(sched_getaffinity(0, sizeof(original), &original) == 0);
	const auto original_count = CPU_COUNT(&original);
	REQUIRE(original_count >= 1);

	cpu_set_t narrowed;
	CPU_ZERO(&narrowed);
	// Pin to the lowest CPU we are actually allowed on.
	int chosen = -1;
	for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
		if (CPU_ISSET(cpu, &original)) {
			chosen = cpu;
			break;
		}
	}
	REQUIRE(chosen >= 0);
	CPU_SET(chosen, &narrowed);

	REQUIRE(sched_setaffinity(0, sizeof(narrowed), &narrowed) == 0);
	const auto pinned = UsableCoreCount();
	// Restore before asserting, so a failure cannot leave the process pinned.
	REQUIRE(sched_setaffinity(0, sizeof(original), &original) == 0);

	REQUIRE(pinned == 1);

	// And it comes back up once the mask is restored. hardware_concurrency() would
	// have reported original_count both times, which is exactly the bug.
	if (original_count > 1) {
		REQUIRE(UsableCoreCount() > 1);
	}
}
#endif
