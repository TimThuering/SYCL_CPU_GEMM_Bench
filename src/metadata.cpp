#include "bench/metadata.hpp"

#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <ctime>
#include <fstream>
#include <ostream>
#include <string_view>

extern char** environ;

namespace bench {

namespace {

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t");
  const auto e = s.find_last_not_of(" \t\r\n");
  return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// First value of "key : value" in /proc/cpuinfo, empty if not present
std::string cpuinfo_value(const std::string& key) {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    const auto colon = line.find(':');
    if (colon != std::string::npos && trim(line.substr(0, colon)) == key)
      return trim(line.substr(colon + 1));
  }
  return "";
}

// x86: "model name", RISC-V: "uarch", ARM: only implementer / part codes
std::string cpu_model() {
  if (auto v = cpuinfo_value("model name"); !v.empty()) return v;
  if (auto v = cpuinfo_value("uarch"); !v.empty()) return v;
  const auto impl = cpuinfo_value("CPU implementer");
  const auto part = cpuinfo_value("CPU part");
  if (!impl.empty() || !part.empty()) return "CPU implementer " + impl + ", CPU part " + part;
  return "unknown";
}

std::string timestamp_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm);
  return buf;
}

// CPU set -> "0-3,8,10-11"
std::string cpu_list(const cpu_set_t& set) {
  std::string out;
  int start = -1;
  for (int cpu = 0; cpu <= CPU_SETSIZE; ++cpu) {
    const bool in = cpu < CPU_SETSIZE && CPU_ISSET(cpu, &set);
    if (in && start < 0) start = cpu;
    if (!in && start >= 0) {
      if (!out.empty()) out += ',';
      out += std::to_string(start);
      if (cpu - 1 > start) out += "-" + std::to_string(cpu - 1);
      start = -1;
    }
  }
  return out;
}

constexpr std::array<std::string_view, 13> env_prefixes = {
    "OMP_", "GOMP_", "KMP_", "ACPP_", "HIPSYCL_", "POCL_", "SYCL_", "ONEAPI_", "UR_",
    "OCL_ICD", "TBB_", "LD_LIBRARY_PATH", "LD_PRELOAD"};

} // namespace

HostInfo query_host() {
  HostInfo h;
  h.timestamp = timestamp_now();

  char host[256] = {};
  if (gethostname(host, sizeof(host) - 1) == 0) h.hostname = host;

  h.cpu_model = cpu_model();

  utsname u{};
  if (uname(&u) == 0) h.os = std::string(u.sysname) + " " + u.release + " " + u.machine;

  const long online = sysconf(_SC_NPROCESSORS_ONLN);
  h.online_cpus = online > 0 ? static_cast<std::size_t>(online) : 0;

  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    h.affinity_count = static_cast<std::size_t>(CPU_COUNT(&set));
    h.affinity_list = cpu_list(set);
  }
  return h;
}

std::map<std::string, std::string> relevant_env() {
  std::map<std::string, std::string> env;
  for (char** e = environ; *e != nullptr; ++e) {
    const std::string_view entry(*e);
    const auto eq = entry.find('=');
    if (eq == std::string_view::npos) continue;
    const std::string_view name = entry.substr(0, eq);
    for (const auto prefix : env_prefixes)
      if (name.substr(0, prefix.size()) == prefix) {
        env.emplace(std::string(name), std::string(entry.substr(eq + 1)));
        break;
      }
  }
  return env;
}

std::size_t os_thread_count() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("Threads:", 0) == 0) return static_cast<std::size_t>(std::stoul(line.substr(8)));
  return 0;
}

void print_host_info(std::ostream& os, const HostInfo& h) {
  os << "host:          " << h.hostname << " (" << h.cpu_model << ", " << h.online_cpus << " online CPUs, "
     << h.os << ")\n"
     << "affinity:      " << h.affinity_count << " CPUs [" << h.affinity_list << "]\n"
     << "time:          " << h.timestamp << "\n";
}

void print_device_info(std::ostream& os, const DeviceInfo& d) {
  os << "device:        " << d.name << " [" << d.platform_name << "], " << d.max_compute_units
     << " compute units\n";
}

void print_env(std::ostream& os, const std::map<std::string, std::string>& env) {
  os << "environment:  ";
  if (env.empty()) os << " (none of the relevant variables set)";
  for (const auto& [k, v] : env) os << " " << k << "=" << v;
  os << "\n";
}

} // namespace bench
