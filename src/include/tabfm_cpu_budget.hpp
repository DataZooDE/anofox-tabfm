//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_cpu_budget.hpp — how many cores this process may actually use, as
// opposed to how many the kernel can see. Sizes the `anofox_tabfm_threads`
// default so a container allocation is not oversubscribed (HLD D9).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"

#include <string>

namespace duckdb {
namespace anofox {

//! Cores this process may actually use.
//!
//! `std::thread::hardware_concurrency()` reports what the kernel can see, which inside a
//! container is the host. On a 64-core allocation inside a 256-core host it returns 256, so a
//! default of `hardware_concurrency() / 2` becomes 128 intra-op threads per session -- and the
//! host runs several sessions concurrently, one per DuckDB task. Measured on such a pod: 132
//! threads in one duckdb process and a load average of 143 against 64 usable cores, for a query
//! configured with `SET threads = 4`.
//!
//! A container can impose that allocation two independent ways, and which one it uses is not
//! the application's choice:
//!
//!   * a **cpuset** -- the kernel refuses to schedule us anywhere else, so
//!     `sched_getaffinity` reports it.
//!   * a **CFS bandwidth quota** -- we may run on every CPU, but only for quota/period
//!     CPU-seconds per period. Affinity is untouched and reports the whole host.
//!
//! Kubernetes only pins a cpuset under the static CPU Manager policy with a Guaranteed pod and
//! integer CPU limits; the default enforcement for `limits.cpu` is quota. Honouring just one of
//! the two would leave the other deployment oversubscribed, so this takes the smaller.
//!
//! Falls back to `hardware_concurrency()` when neither limit is discoverable, so behaviour is
//! unchanged off Linux and on an unconstrained host. Never returns 0.
idx_t UsableCoreCount();

//! Parse a cgroup v2 `cpu.max`: `"<quota_us> <period_us>"`, or `"max <period_us>"` when
//! uncapped. Returns the core count rounded up, or 0 for uncapped/unparseable.
idx_t ParseCpuMaxV2(const std::string &contents);

//! Parse a cgroup v1 `cpu.cfs_quota_us` / `cpu.cfs_period_us` pair. A quota of -1 is uncapped.
//! Returns the core count rounded up, or 0 for uncapped/unparseable.
idx_t ParseCpuQuotaV1(const std::string &quota_us, const std::string &period_us);

//! Smaller of two core limits, where 0 means "this mechanism imposes no limit" and so must not
//! win the comparison. Returns 0 only when neither imposes one.
idx_t CombineCoreLimits(idx_t left, idx_t right);

//! This process's cgroup path, from the contents of `/proc/self/cgroup`. The unified (v2)
//! hierarchy is the line with an empty controller field: `0::/a/b`. Returns "" when absent.
//!
//! Needed because `/sys/fs/cgroup/cpu.max` is the *root* cgroup, which carries no limit. Under a
//! private cgroup namespace the container's own cgroup is the root and the two coincide, but
//! with `cgroupns=host` -- still common under Kubernetes -- our cgroup is nested several levels
//! down and the root file does not exist at all.
std::string ParseUnifiedCgroupPath(const std::string &contents);

//! Same, for the legacy (v1) `cpu` controller: `4:cpu,cpuacct:/a/b`. Returns "" when absent.
std::string ParseLegacyCpuCgroupPath(const std::string &contents);

} // namespace anofox
} // namespace duckdb
