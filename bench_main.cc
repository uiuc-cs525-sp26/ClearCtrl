#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"

#include "compaction_controller.h"

using namespace ROCKSDB_NAMESPACE;

static std::atomic<bool> g_stop{false};

constexpr bool DEFAULT_CONTROLLER_ENABLED = true;
constexpr uint64_t DEFAULT_RUNNER_SLEEP_MS = 100;
constexpr uint64_t DEFAULT_RUNNER_BATCH_OPS = 4000;
constexpr uint64_t DEFAULT_ROCKSDB_WRITE_BUFFER_SIZE_MB = 4;
constexpr int DEFAULT_ROCKSDB_L0_COMPACTION_TRIGGER = 8;
constexpr int DEFAULT_CTRL_INTERVAL_SEC = 2;
constexpr int DEFAULT_CTRL_COOLDOWN_SEC = 6;
// LinUCB defaults
constexpr double DEFAULT_BANDIT_ALPHA = 1.0;
constexpr int DEFAULT_BANDIT_PROFILE_CYCLES = 10;
constexpr double DEFAULT_BANDIT_REWARD_LAMBDA_STALL = 1.0;
constexpr double DEFAULT_BANDIT_REWARD_LAMBDA_L0 = 0.3;
constexpr double DEFAULT_BANDIT_REWARD_LAMBDA_STOP = 1.0;
constexpr double DEFAULT_BANDIT_REWARD_LAMBDA_DELAY = 0.5;
constexpr double DEFAULT_BANDIT_REWARD_TP_NORM_MB_S = 100.0;
constexpr const char* DEFAULT_BANDIT_FAST_ARMS = "2:1;4:1;4:2;6:2;6:4";
constexpr const char* DEFAULT_BANDIT_PROFILES = "8:12;12:16;16:24";
constexpr int DEFAULT_BANDIT_INITIAL_ARM = 0;
constexpr int DEFAULT_BANDIT_INITIAL_PROFILE = 0;
constexpr const char* DEFAULT_LOG_PATH_PATTERN = "logs/metrics_<run_id>.csv";

struct WorkloadPhase {
    uint64_t batch_ops;
    uint64_t sleep_ms;
    uint64_t rounds;
};

void PrintUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [runner options] [rocksdb options] [ctrl options]\n"
        << "Runner options:\n"
        << "  --controller=on|off                 Enable/disable dynamic controller (default: "
        << (DEFAULT_CONTROLLER_ENABLED ? "on" : "off") << ")\n"
        << "  --runner-schedule=\"OPS:SLEEP:ROUNDS;...\"  Required multi-phase schedule. "
           "Each phase can leave OPS or SLEEP empty to use defaults "
           "(defaults: OPS="
        << DEFAULT_RUNNER_BATCH_OPS << ", SLEEP=" << DEFAULT_RUNNER_SLEEP_MS
        << "ms), e.g. \":100:300;8000::200\"\n"
        << "  --log-path=FILE                     Output CSV path (default: "
        << DEFAULT_LOG_PATH_PATTERN << ")\n"
        << "  --run-id=ID                         Explicit run id (default: auto timestamp)\n"
        << "RocksDB options:\n"
        << "  --rocksdb-increase-parallelism=N    Override IncreaseParallelism thread count (default: auto)\n"
        << "  --rocksdb-write-buffer-size-mb=N    write_buffer_size in MiB (default: "
        << DEFAULT_ROCKSDB_WRITE_BUFFER_SIZE_MB << ")\n"
        << "  --rocksdb-l0-compaction-trigger=N   level0_file_num_compaction_trigger (default: "
        << DEFAULT_ROCKSDB_L0_COMPACTION_TRIGGER << ")\n"
        << "ClearCtrl (LinUCB) options:\n"
        << "  --ctrl-interval-sec=N               Control interval seconds (default: "
        << DEFAULT_CTRL_INTERVAL_SEC << ")\n"
        << "  --ctrl-cooldown-sec=N               Switch cooldown seconds (default: "
        << DEFAULT_CTRL_COOLDOWN_SEC << ")\n"
        << "  --ctrl-bandit-arms=\"BG:SUB;BG:SUB;...\"   Fast-bandit action set "
           "(max_background_jobs:max_subcompactions, default: "
        << DEFAULT_BANDIT_FAST_ARMS << ")\n"
        << "  --ctrl-bandit-profiles=\"SLOW:STOP;...\"   Slow-bandit profile set "
           "for level0_slowdown/stop_writes_trigger (default: "
        << DEFAULT_BANDIT_PROFILES << ")\n"
        << "                                      Each profile must satisfy "
           "stop >= slowdown >= --rocksdb-l0-compaction-trigger.\n"
        << "  --ctrl-bandit-initial-arm=N         Initial fast-arm index into --ctrl-bandit-arms (default: "
        << DEFAULT_BANDIT_INITIAL_ARM << ")\n"
        << "  --ctrl-bandit-initial-profile=N     Initial profile index into --ctrl-bandit-profiles (default: "
        << DEFAULT_BANDIT_INITIAL_PROFILE << ")\n"
        << "  --ctrl-bandit-alpha=N.N             UCB exploration coefficient (default: "
        << DEFAULT_BANDIT_ALPHA << ")\n"
        << "  --ctrl-bandit-profile-cycles=N      Fast cycles between profile updates (default: "
        << DEFAULT_BANDIT_PROFILE_CYCLES << ")\n"
        << "  --ctrl-bandit-reward-lambda-stall=N Reward weight on stall_fraction (default: "
        << DEFAULT_BANDIT_REWARD_LAMBDA_STALL << ")\n"
        << "  --ctrl-bandit-reward-lambda-l0=N    Reward weight on l0/l0_base ratio (default: "
        << DEFAULT_BANDIT_REWARD_LAMBDA_L0 << ")\n"
        << "  --ctrl-bandit-reward-lambda-stop=N  Profile-only weight on write-stop indicator (default: "
        << DEFAULT_BANDIT_REWARD_LAMBDA_STOP << ")\n"
        << "  --ctrl-bandit-reward-lambda-delay=N Profile-only weight on delayed-write fraction (default: "
        << DEFAULT_BANDIT_REWARD_LAMBDA_DELAY << ")\n"
        << "  --ctrl-bandit-reward-tp-norm-mb-s=N Throughput normalization MB/s (default: "
        << DEFAULT_BANDIT_REWARD_TP_NORM_MB_S << ")\n"
        << "  -h, --help            Show help\n";
}

void signal_handler(int) {
    g_stop.store(true);
}

double TimevalToSeconds(const timeval& tv) {
    return static_cast<double>(tv.tv_sec) +
           static_cast<double>(tv.tv_usec) / 1e6;
}

bool GetProcessCpuSeconds(double* user_sec, double* sys_sec) {
    rusage ru {};
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return false;
    }
    *user_sec = TimevalToSeconds(ru.ru_utime);
    *sys_sec = TimevalToSeconds(ru.ru_stime);
    return true;
}

bool ParseRunnerSchedule(const std::string& spec,
                         uint64_t default_batch_ops,
                         uint64_t default_sleep_ms,
                         std::vector<WorkloadPhase>* phases,
                         std::string* error) {
    phases->clear();
    if (spec.empty()) {
        return true;
    }

    std::stringstream ss(spec);
    std::string segment;
    while (std::getline(ss, segment, ';')) {
        if (segment.empty()) {
            continue;
        }
        size_t p1 = segment.find(':');
        size_t p2 = segment.find(':', p1 == std::string::npos ? 0 : p1 + 1);
        if (p1 == std::string::npos || p2 == std::string::npos ||
            segment.find(':', p2 + 1) != std::string::npos) {
            *error = "Invalid phase format: " + segment;
            return false;
        }

        const std::string ops_str = segment.substr(0, p1);
        const std::string sleep_str = segment.substr(p1 + 1, p2 - p1 - 1);
        const std::string rounds_str = segment.substr(p2 + 1);
        try {
            WorkloadPhase phase{};
            phase.batch_ops =
                ops_str.empty() ? default_batch_ops : std::stoull(ops_str);
            phase.sleep_ms =
                sleep_str.empty() ? default_sleep_ms : std::stoull(sleep_str);
            if (rounds_str.empty()) {
                *error = "rounds cannot be empty in phase: " + segment;
                return false;
            }
            phase.rounds = std::stoull(rounds_str);
            if (phase.batch_ops == 0 || phase.rounds == 0) {
                *error = "batch_ops and rounds must be > 0 in phase: " + segment;
                return false;
            }
            phases->push_back(phase);
        } catch (...) {
            *error = "Invalid numeric value in phase: " + segment;
            return false;
        }
    }

    if (phases->empty()) {
        *error = "runner-schedule is empty after parsing";
        return false;
    }
    return true;
}

bool ParseBanditFastArms(const std::string& spec,
                         std::vector<BanditFastArm>* out,
                         std::string* error) {
    out->clear();
    std::stringstream ss(spec);
    std::string segment;
    while (std::getline(ss, segment, ';')) {
        if (segment.empty()) {
            continue;
        }
        size_t p = segment.find(':');
        if (p == std::string::npos ||
            segment.find(':', p + 1) != std::string::npos) {
            *error = "invalid fast arm: " + segment +
                     " (expected BG_JOBS:SUBCOMPACTIONS)";
            return false;
        }
        try {
            BanditFastArm arm{};
            arm.max_background_jobs = std::stoi(segment.substr(0, p));
            arm.max_subcompactions = std::stoi(segment.substr(p + 1));
            if (arm.max_background_jobs <= 0 || arm.max_subcompactions <= 0) {
                *error = "fast arm values must be > 0: " + segment;
                return false;
            }
            out->push_back(arm);
        } catch (...) {
            *error = "fast arm parse error: " + segment;
            return false;
        }
    }
    if (out->empty()) {
        *error = "no fast arms parsed";
        return false;
    }
    return true;
}

bool ParseBanditProfiles(const std::string& spec,
                         std::vector<BanditProfile>* out,
                         std::string* error) {
    out->clear();
    std::stringstream ss(spec);
    std::string segment;
    while (std::getline(ss, segment, ';')) {
        if (segment.empty()) {
            continue;
        }
        size_t p = segment.find(':');
        if (p == std::string::npos ||
            segment.find(':', p + 1) != std::string::npos) {
            *error = "invalid profile: " + segment +
                     " (expected SLOWDOWN:STOP)";
            return false;
        }
        try {
            BanditProfile prof{};
            prof.l0_slowdown_trigger = std::stoi(segment.substr(0, p));
            prof.l0_stop_trigger = std::stoi(segment.substr(p + 1));
            if (prof.l0_slowdown_trigger <= 0 || prof.l0_stop_trigger <= 0 ||
                prof.l0_stop_trigger < prof.l0_slowdown_trigger) {
                *error = "profile must satisfy 0 < slowdown <= stop: " +
                         segment;
                return false;
            }
            out->push_back(prof);
        } catch (...) {
            *error = "profile parse error: " + segment;
            return false;
        }
    }
    if (out->empty()) {
        *error = "no profiles parsed";
        return false;
    }
    return true;
}

// RocksDB requires (typically) compaction_trigger <= slowdown <= stop for L0.
// Enforce for every profile arm: stop >= slowdown >= compaction_trigger.
bool ValidateBanditProfilesL0Ordering(
    const std::vector<BanditProfile>& profiles,
    int l0_compaction_trigger,
    std::string* error) {
    if (l0_compaction_trigger <= 0) {
        *error = "level0_file_num_compaction_trigger must be positive";
        return false;
    }
    for (size_t i = 0; i < profiles.size(); ++i) {
        const BanditProfile& p = profiles[i];
        if (p.l0_stop_trigger < p.l0_slowdown_trigger) {
            *error = "profile index " + std::to_string(i) +
                     ": stop_writes_trigger (" +
                     std::to_string(p.l0_stop_trigger) +
                     ") < slowdown_writes_trigger (" +
                     std::to_string(p.l0_slowdown_trigger) +
                     "); require stop >= slowdown >= compaction";
            return false;
        }
        if (p.l0_slowdown_trigger < l0_compaction_trigger) {
            *error =
                "profile index " + std::to_string(i) +
                ": slowdown_writes_trigger (" +
                std::to_string(p.l0_slowdown_trigger) +
                ") < level0_file_num_compaction_trigger (" +
                std::to_string(l0_compaction_trigger) +
                "); require stop >= slowdown >= compaction";
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const std::string DB_PATH = "./testdb";
    bool controller_enabled = DEFAULT_CONTROLLER_ENABLED;
    std::string runner_schedule;
    std::string log_path;
    std::string run_id;

    int rocksdb_increase_parallelism_override = 0;  // 0 means auto
    uint64_t rocksdb_write_buffer_size_mb = DEFAULT_ROCKSDB_WRITE_BUFFER_SIZE_MB;
    int rocksdb_l0_compaction_trigger = DEFAULT_ROCKSDB_L0_COMPACTION_TRIGGER;

    int ctrl_interval_sec = DEFAULT_CTRL_INTERVAL_SEC;
    int ctrl_cooldown_sec = DEFAULT_CTRL_COOLDOWN_SEC;
    std::string bandit_arms_spec = DEFAULT_BANDIT_FAST_ARMS;
    std::string bandit_profiles_spec = DEFAULT_BANDIT_PROFILES;
    int bandit_initial_arm = DEFAULT_BANDIT_INITIAL_ARM;
    int bandit_initial_profile = DEFAULT_BANDIT_INITIAL_PROFILE;
    double bandit_alpha = DEFAULT_BANDIT_ALPHA;
    int bandit_profile_cycles = DEFAULT_BANDIT_PROFILE_CYCLES;
    double bandit_reward_lambda_stall = DEFAULT_BANDIT_REWARD_LAMBDA_STALL;
    double bandit_reward_lambda_l0 = DEFAULT_BANDIT_REWARD_LAMBDA_L0;
    double bandit_reward_lambda_stop = DEFAULT_BANDIT_REWARD_LAMBDA_STOP;
    double bandit_reward_lambda_delay = DEFAULT_BANDIT_REWARD_LAMBDA_DELAY;
    double bandit_reward_tp_norm_mb_s = DEFAULT_BANDIT_REWARD_TP_NORM_MB_S;

    enum OptionId {
        OPT_CONTROLLER = 1000,
        OPT_RUNNER_SCHEDULE,
        OPT_LOG_PATH,
        OPT_RUN_ID,
        OPT_RDB_INCREASE_PARALLELISM,
        OPT_RDB_WRITE_BUFFER_SIZE_MB,
        OPT_RDB_L0_COMPACTION_TRIGGER,
        OPT_CTRL_INTERVAL_SEC,
        OPT_CTRL_COOLDOWN_SEC,
        OPT_CTRL_BANDIT_ARMS,
        OPT_CTRL_BANDIT_PROFILES,
        OPT_CTRL_BANDIT_INITIAL_ARM,
        OPT_CTRL_BANDIT_INITIAL_PROFILE,
        OPT_CTRL_BANDIT_ALPHA,
        OPT_CTRL_BANDIT_PROFILE_CYCLES,
        OPT_CTRL_BANDIT_LAMBDA_STALL,
        OPT_CTRL_BANDIT_LAMBDA_L0,
        OPT_CTRL_BANDIT_LAMBDA_STOP,
        OPT_CTRL_BANDIT_LAMBDA_DELAY,
        OPT_CTRL_BANDIT_TP_NORM,
    };

    static struct option long_options[] = {
        {"controller", required_argument, nullptr, OPT_CONTROLLER},
        {"runner-schedule", required_argument, nullptr, OPT_RUNNER_SCHEDULE},
        {"log-path", required_argument, nullptr, OPT_LOG_PATH},
        {"run-id", required_argument, nullptr, OPT_RUN_ID},
        {"rocksdb-increase-parallelism", required_argument, nullptr,
         OPT_RDB_INCREASE_PARALLELISM},
        {"rocksdb-write-buffer-size-mb", required_argument, nullptr,
         OPT_RDB_WRITE_BUFFER_SIZE_MB},
        {"rocksdb-l0-compaction-trigger", required_argument, nullptr,
         OPT_RDB_L0_COMPACTION_TRIGGER},
        {"ctrl-interval-sec", required_argument, nullptr, OPT_CTRL_INTERVAL_SEC},
        {"ctrl-cooldown-sec", required_argument, nullptr, OPT_CTRL_COOLDOWN_SEC},
        {"ctrl-bandit-arms", required_argument, nullptr, OPT_CTRL_BANDIT_ARMS},
        {"ctrl-bandit-profiles", required_argument, nullptr,
         OPT_CTRL_BANDIT_PROFILES},
        {"ctrl-bandit-initial-arm", required_argument, nullptr,
         OPT_CTRL_BANDIT_INITIAL_ARM},
        {"ctrl-bandit-initial-profile", required_argument, nullptr,
         OPT_CTRL_BANDIT_INITIAL_PROFILE},
        {"ctrl-bandit-alpha", required_argument, nullptr, OPT_CTRL_BANDIT_ALPHA},
        {"ctrl-bandit-profile-cycles", required_argument, nullptr,
         OPT_CTRL_BANDIT_PROFILE_CYCLES},
        {"ctrl-bandit-reward-lambda-stall", required_argument, nullptr,
         OPT_CTRL_BANDIT_LAMBDA_STALL},
        {"ctrl-bandit-reward-lambda-l0", required_argument, nullptr,
         OPT_CTRL_BANDIT_LAMBDA_L0},
        {"ctrl-bandit-reward-lambda-stop", required_argument, nullptr,
         OPT_CTRL_BANDIT_LAMBDA_STOP},
        {"ctrl-bandit-reward-lambda-delay", required_argument, nullptr,
         OPT_CTRL_BANDIT_LAMBDA_DELAY},
        {"ctrl-bandit-reward-tp-norm-mb-s", required_argument, nullptr,
         OPT_CTRL_BANDIT_TP_NORM},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt = 0;
    auto parse_int_arg = [&](const char* name, const char* value, int* out) {
        try {
            *out = std::stoi(value);
            return true;
        } catch (...) {
            std::cerr << "Invalid " << name << " value: " << value << "\n";
            return false;
        }
    };
    auto parse_u64_arg = [&](const char* name, const char* value,
                             uint64_t* out) {
        try {
            *out = std::stoull(value);
            return true;
        } catch (...) {
            std::cerr << "Invalid " << name << " value: " << value << "\n";
            return false;
        }
    };
    auto parse_double_arg = [&](const char* name, const char* value,
                                double* out) {
        try {
            *out = std::stod(value);
            return true;
        } catch (...) {
            std::cerr << "Invalid " << name << " value: " << value << "\n";
            return false;
        }
    };
    while ((opt = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
        switch (opt) {
            case OPT_CONTROLLER: {
                std::string controller_arg = optarg;
                if (controller_arg == "on") {
                    controller_enabled = true;
                } else if (controller_arg == "off") {
                    controller_enabled = false;
                } else {
                    std::cerr << "Invalid --controller value: " << controller_arg
                              << " (expected on/off)\n";
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            }
            case OPT_RUNNER_SCHEDULE:
                runner_schedule = optarg;
                break;
            case OPT_LOG_PATH:
                log_path = optarg;
                break;
            case OPT_RUN_ID:
                run_id = optarg;
                break;
            case OPT_RDB_INCREASE_PARALLELISM:
                if (!parse_int_arg("--rocksdb-increase-parallelism", optarg,
                                   &rocksdb_increase_parallelism_override)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_RDB_WRITE_BUFFER_SIZE_MB:
                if (!parse_u64_arg("--rocksdb-write-buffer-size-mb", optarg,
                                   &rocksdb_write_buffer_size_mb)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_RDB_L0_COMPACTION_TRIGGER:
                if (!parse_int_arg("--rocksdb-l0-compaction-trigger", optarg,
                                   &rocksdb_l0_compaction_trigger)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_INTERVAL_SEC:
                if (!parse_int_arg("--ctrl-interval-sec", optarg,
                                   &ctrl_interval_sec)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_COOLDOWN_SEC:
                if (!parse_int_arg("--ctrl-cooldown-sec", optarg,
                                   &ctrl_cooldown_sec)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_ARMS:
                bandit_arms_spec = optarg;
                break;
            case OPT_CTRL_BANDIT_PROFILES:
                bandit_profiles_spec = optarg;
                break;
            case OPT_CTRL_BANDIT_INITIAL_ARM:
                if (!parse_int_arg("--ctrl-bandit-initial-arm", optarg,
                                   &bandit_initial_arm)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_INITIAL_PROFILE:
                if (!parse_int_arg("--ctrl-bandit-initial-profile", optarg,
                                   &bandit_initial_profile)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_ALPHA:
                if (!parse_double_arg("--ctrl-bandit-alpha", optarg,
                                      &bandit_alpha)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_PROFILE_CYCLES:
                if (!parse_int_arg("--ctrl-bandit-profile-cycles", optarg,
                                   &bandit_profile_cycles)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_LAMBDA_STALL:
                if (!parse_double_arg("--ctrl-bandit-reward-lambda-stall",
                                      optarg,
                                      &bandit_reward_lambda_stall)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_LAMBDA_L0:
                if (!parse_double_arg("--ctrl-bandit-reward-lambda-l0", optarg,
                                      &bandit_reward_lambda_l0)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_LAMBDA_STOP:
                if (!parse_double_arg("--ctrl-bandit-reward-lambda-stop",
                                      optarg,
                                      &bandit_reward_lambda_stop)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_LAMBDA_DELAY:
                if (!parse_double_arg("--ctrl-bandit-reward-lambda-delay",
                                      optarg,
                                      &bandit_reward_lambda_delay)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_BANDIT_TP_NORM:
                if (!parse_double_arg("--ctrl-bandit-reward-tp-norm-mb-s",
                                      optarg,
                                      &bandit_reward_tp_norm_mb_s)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case 'h':
                PrintUsage(argv[0]);
                return 0;
            default:
                PrintUsage(argv[0]);
                return 1;
        }
    }

    if (run_id.empty()) {
        auto now = std::chrono::system_clock::now();
        auto epoch_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();
        run_id = std::to_string(epoch_ms);
    }
    if (log_path.empty()) {
        log_path = "logs/metrics_" + run_id + ".csv";
    }
    if (runner_schedule.empty()) {
        std::cerr << "Missing required --runner-schedule\n";
        PrintUsage(argv[0]);
        return 1;
    }
    std::vector<WorkloadPhase> workload_phases;
    std::string parse_error;
    if (!ParseRunnerSchedule(runner_schedule, DEFAULT_RUNNER_BATCH_OPS,
                             DEFAULT_RUNNER_SLEEP_MS,
                             &workload_phases, &parse_error)) {
        std::cerr << "Invalid --runner-schedule: " << parse_error << "\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // Always parse arms/profiles: even with controller off, we need them to
    // resolve the initial fast-arm/profile that determines RocksDB's startup
    // configuration (max_background_jobs, max_subcompactions,
    // level0_slowdown/stop_writes_trigger).
    std::vector<BanditFastArm> bandit_fast_arms;
    std::vector<BanditProfile> bandit_profiles;
    if (!ParseBanditFastArms(bandit_arms_spec, &bandit_fast_arms,
                             &parse_error)) {
        std::cerr << "Invalid --ctrl-bandit-arms: " << parse_error << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (!ParseBanditProfiles(bandit_profiles_spec, &bandit_profiles,
                             &parse_error)) {
        std::cerr << "Invalid --ctrl-bandit-profiles: " << parse_error << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (!ValidateBanditProfilesL0Ordering(bandit_profiles,
                                          rocksdb_l0_compaction_trigger,
                                          &parse_error)) {
        std::cerr << "Invalid L0 triggers (need stop >= slowdown >= "
                     "compaction for each profile): "
                  << parse_error << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (bandit_initial_arm < 0 ||
        bandit_initial_arm >= static_cast<int>(bandit_fast_arms.size())) {
        std::cerr << "Invalid --ctrl-bandit-initial-arm=" << bandit_initial_arm
                  << " (valid range: 0.."
                  << (static_cast<int>(bandit_fast_arms.size()) - 1) << ")\n";
        PrintUsage(argv[0]);
        return 1;
    }
    if (bandit_initial_profile < 0 ||
        bandit_initial_profile >= static_cast<int>(bandit_profiles.size())) {
        std::cerr << "Invalid --ctrl-bandit-initial-profile="
                  << bandit_initial_profile << " (valid range: 0.."
                  << (static_cast<int>(bandit_profiles.size()) - 1) << ")\n";
        PrintUsage(argv[0]);
        return 1;
    }

    // Enforce profile_interval_cycles * interval_sec >= 2 * cooldown_sec.
    // If the requested value is too small, raise it to
    // ceil(2 * cooldown_sec / interval_sec) (with interval clamped to >= 1)
    // and warn. A profile window shorter than 2 fast cooldowns leaves the
    // fast bandit too little time to commit a switch within each window,
    // so the per-window aggregated profile reward becomes too noisy.
    const int isec = std::max(1, ctrl_interval_sec);
    const int csec = std::max(0, ctrl_cooldown_sec);
    const int requested = std::max(1, bandit_profile_cycles);
    const long long need = 2LL * static_cast<long long>(csec);
    const long long window =
        static_cast<long long>(requested) * static_cast<long long>(isec);
    if (window < need) {
        const long long isec_ll = static_cast<long long>(isec);
        const int adjusted =
            std::max(1, static_cast<int>((need + isec_ll - 1LL) /
                                            isec_ll));
        std::cerr << "[config][bandit] warning: profile_cycles * "
                        "interval_sec ("
                    << requested << " * " << isec << " = " << window
                    << "s) < 2 * cooldown_sec (2 * " << csec
                    << " = " << need
                    << "s); using profile_cycles=" << adjusted
                    << " (ceil(2*cooldown_sec/interval_sec))\n";
        bandit_profile_cycles = adjusted;
    } else {
        bandit_profile_cycles = requested;
    }

    const int rocksdb_max_background_jobs =
        bandit_fast_arms[bandit_initial_arm].max_background_jobs;
    const int rocksdb_max_subcompactions =
        bandit_fast_arms[bandit_initial_arm].max_subcompactions;
    const int rocksdb_l0_slowdown_trigger =
        bandit_profiles[bandit_initial_profile].l0_slowdown_trigger;
    const int rocksdb_l0_stop_trigger =
        bandit_profiles[bandit_initial_profile].l0_stop_trigger;

    std::filesystem::path log_file_path(log_path);
    std::filesystem::path log_dir = log_file_path.has_parent_path()
                                        ? log_file_path.parent_path()
                                        : std::filesystem::path(".");
    std::filesystem::create_directories(log_dir);
    std::string config_stem = log_file_path.stem().string();
    if (config_stem.empty()) {
        config_stem = "metrics_" + run_id;
    }
    std::filesystem::path run_config_path =
        log_dir / (config_stem + "-run_config.json");
    // Auto increase_parallelism: max(any arm bg_jobs we may use). When the
    // controller is off we still size the worker pool to fit the initial arm.
    int auto_increase_parallelism = rocksdb_max_background_jobs;
    if (controller_enabled) {
        for (const BanditFastArm& a : bandit_fast_arms) {
            auto_increase_parallelism =
                std::max(auto_increase_parallelism, a.max_background_jobs);
        }
    }
    int rocksdb_increase_parallelism =
        rocksdb_increase_parallelism_override
            ? rocksdb_increase_parallelism_override
            : auto_increase_parallelism;

    uint64_t rounds = 0;
    for (const WorkloadPhase& phase : workload_phases) {
        rounds += phase.rounds;
    }

    std::cout << "[config] run_id=" << run_id
              << ", controller=" << (controller_enabled ? "on" : "off")
              << ", default_runner_sleep_ms=" << DEFAULT_RUNNER_SLEEP_MS
              << ", default_runner_batch_ops=" << DEFAULT_RUNNER_BATCH_OPS
              << ", rounds=" << rounds
              << ", runner_schedule="
              << (runner_schedule.empty() ? "<none>" : runner_schedule)
              << ", log_path=" << log_path << "\n"
              << "[config] run_config_path=" << run_config_path.string() << "\n"
              << "[config][rocksdb] increase_parallelism="
              << rocksdb_increase_parallelism
              << " ("
              << (rocksdb_increase_parallelism_override > 0 ? "manual" : "auto")
              << ")"
              << ", max_background_jobs=" << rocksdb_max_background_jobs
              << ", max_subcompactions=" << rocksdb_max_subcompactions
              << ", write_buffer_size_mb=" << rocksdb_write_buffer_size_mb
              << ", l0_compaction_trigger=" << rocksdb_l0_compaction_trigger
              << ", l0_slowdown_trigger=" << rocksdb_l0_slowdown_trigger
              << ", l0_stop_trigger=" << rocksdb_l0_stop_trigger << "\n"
              << "[config][ctrl] interval_sec=" << ctrl_interval_sec
              << ", cooldown_sec=" << ctrl_cooldown_sec;
    std::cout << "\n[config][bandit] arms=\"" << bandit_arms_spec << "\""
              << ", profiles=\"" << bandit_profiles_spec << "\""
              << ", initial_arm=" << bandit_initial_arm
              << ", initial_profile=" << bandit_initial_profile;
    if (controller_enabled) {
        std::cout << ", alpha=" << bandit_alpha
                  << ", profile_cycles=" << bandit_profile_cycles
                  << ", lambda_stall=" << bandit_reward_lambda_stall
                  << ", lambda_l0=" << bandit_reward_lambda_l0
                  << ", lambda_stop=" << bandit_reward_lambda_stop
                  << ", lambda_delay=" << bandit_reward_lambda_delay
                  << ", tp_norm_mb_s=" << bandit_reward_tp_norm_mb_s;
    }
    std::cout << std::endl;

    std::ofstream run_cfg(run_config_path.string(), std::ios::trunc);
    if (run_cfg.is_open()) {
        run_cfg
            << "{\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"runner\": {\n"
            << "    \"controller\": \"" << (controller_enabled ? "on" : "off")
            << "\",\n"
            << "    \"default_sleep_ms\": " << DEFAULT_RUNNER_SLEEP_MS << ",\n"
            << "    \"default_batch_ops\": " << DEFAULT_RUNNER_BATCH_OPS << ",\n"
            << "    \"rounds\": " << rounds << ",\n"
            << "    \"schedule\": \"" << runner_schedule << "\",\n"
            << "    \"log_path\": \"" << log_path << "\"\n"
            << "  },\n"
            << "  \"rocksdb\": {\n"
            << "    \"increase_parallelism\": " << rocksdb_increase_parallelism
            << ",\n"
            << "    \"increase_parallelism_mode\": \""
            << (rocksdb_increase_parallelism_override > 0 ? "manual" : "auto")
            << "\",\n"
            << "    \"max_background_jobs\": " << rocksdb_max_background_jobs << ",\n"
            << "    \"max_subcompactions\": " << rocksdb_max_subcompactions << ",\n"
            << "    \"write_buffer_size_mb\": " << rocksdb_write_buffer_size_mb
            << ",\n"
            << "    \"l0_compaction_trigger\": "
            << rocksdb_l0_compaction_trigger << ",\n"
            << "    \"l0_slowdown_trigger\": " << rocksdb_l0_slowdown_trigger
            << ",\n"
            << "    \"l0_stop_trigger\": " << rocksdb_l0_stop_trigger << "\n"
            << "  },\n"
            << "  \"clearctrl\": {\n"
            << "    \"interval_sec\": " << ctrl_interval_sec << ",\n"
            << "    \"cooldown_sec\": " << ctrl_cooldown_sec << ",\n"
            << "    \"bandit\": {\n"
            << "      \"arms\": \"" << bandit_arms_spec << "\",\n"
            << "      \"profiles\": \"" << bandit_profiles_spec << "\",\n"
            << "      \"initial_arm\": " << bandit_initial_arm << ",\n"
            << "      \"initial_profile\": " << bandit_initial_profile << ",\n"
            << "      \"alpha\": " << bandit_alpha << ",\n"
            << "      \"profile_cycles\": " << bandit_profile_cycles << ",\n"
            << "      \"reward_lambda_stall\": " << bandit_reward_lambda_stall
            << ",\n"
            << "      \"reward_lambda_l0\": " << bandit_reward_lambda_l0
            << ",\n"
            << "      \"reward_lambda_stop\": " << bandit_reward_lambda_stop
            << ",\n"
            << "      \"reward_lambda_delay\": " << bandit_reward_lambda_delay
            << ",\n"
            << "      \"reward_tp_norm_mb_s\": " << bandit_reward_tp_norm_mb_s
            << "\n"
            << "    }\n"
            << "  }\n"
            << "}\n";
    }

    Options options;
    options.create_if_missing = true;

    // 先给一个比较容易观察 compaction/backlog 的配置
    options.IncreaseParallelism(rocksdb_increase_parallelism);
    options.max_background_jobs = rocksdb_max_background_jobs;
    options.max_subcompactions = rocksdb_max_subcompactions;
    options.OptimizeLevelStyleCompaction();
    options.statistics = rocksdb::CreateDBStatistics();

    // 关闭压缩
    options.compression = rocksdb::kNoCompression;

    // 为了更容易让 L0 累积，给一个相对激进一点的配置
    options.write_buffer_size =
        rocksdb_write_buffer_size_mb * 1024ULL * 1024ULL;
    // options.max_write_buffer_number = 3;
    options.level0_file_num_compaction_trigger = rocksdb_l0_compaction_trigger;
    options.level0_slowdown_writes_trigger = rocksdb_l0_slowdown_trigger;
    options.level0_stop_writes_trigger = rocksdb_l0_stop_trigger;

    DB* db = nullptr;
    Status s = DB::Open(options, DB_PATH, &db);
    if (!s.ok()) {
        std::cerr << "Open DB failed: " << s.ToString() << std::endl;
        return 1;
    }

    std::unique_ptr<DB> db_guard(db);

    // 做一点基本读写，确认 DB 正常
    s = db->Put(WriteOptions(), "hello", "world");
    if (!s.ok()) {
        std::cerr << "Put failed: " << s.ToString() << std::endl;
        return 1;
    }

    std::string value;
    s = db->Get(ReadOptions(), "hello", &value);
    if (!s.ok()) {
        std::cerr << "Get failed: " << s.ToString() << std::endl;
        return 1;
    }
    std::cout << "Get(hello) = " << value << std::endl;

    // 控制器开关：on 为 LinUCB 动态调参，off 为 baseline。
    std::unique_ptr<CompactionController> ctl;
    if (controller_enabled) {
        ctl = std::make_unique<CompactionController>(
            db,
            options.statistics,
            &g_stop,
            bandit_fast_arms,
            bandit_profiles,
            bandit_initial_arm,
            bandit_initial_profile,
            ctrl_interval_sec,
            ctrl_cooldown_sec,
            bandit_profile_cycles,
            bandit_alpha,
            bandit_reward_lambda_stall,
            bandit_reward_lambda_l0,
            bandit_reward_lambda_stop,
            bandit_reward_lambda_delay,
            bandit_reward_tp_norm_mb_s,
            rocksdb_l0_compaction_trigger);
        if (!ctl->Start()) {
            std::cerr << "[main] fatal: CompactionController::Start() failed; "
                         "aborting run."
                      << std::endl;
            ctl.reset();
            db_guard.reset();
            return 1;
        }
    }

    std::ofstream log(log_path, std::ios::trunc);
    if (log.is_open()) {
        log << "run_id,phase_idx,timestamp,l0_files,stall_micros_total,stall_micros_delta,"
               "compaction_pending_bytes,is_write_stopped,actual_delayed_write_rate,"
               "num_running_compactions,batch_payload_bytes,batch_latency_ms,bg_jobs,"
               "process_cpu_user_sec,process_cpu_sys_sec,"
               "subcompactions,l0_slowdown_now,l0_stop_now,arm_id,profile_id,"
               "last_fast_reward,last_profile_reward\n";
    }

    // 前台 workload：持续写入，推动 flush/L0/compaction 发生
    uint64_t i = 0;
    uint64_t prev_stall_micros = 0;
    uint64_t rounds_done = 0;
    size_t phase_idx = 0;
    uint64_t rounds_in_phase = 0;
    while (!g_stop.load()) {
        if (phase_idx >= workload_phases.size()) {
            break;
        }
        const WorkloadPhase& phase = workload_phases[phase_idx];
        if (rounds_in_phase >= phase.rounds) {
            ++phase_idx;
            rounds_in_phase = 0;
            continue;
        }
        uint64_t current_batch_ops = phase.batch_ops;
        uint64_t current_sleep_ms = phase.sleep_ms;
        uint64_t phase_idx_for_log = static_cast<uint64_t>(phase_idx + 1);
        WriteBatch batch;
        size_t batch_payload_bytes = 0;
        for (uint64_t j = 0; j < current_batch_ops; ++j) {
            std::string key = "key_" + std::to_string(i++);
            std::string val(1024, 'x');  // 1KB value
            batch.Put(key, val);
            batch_payload_bytes += key.size() + val.size();
        }

        auto write_begin = std::chrono::steady_clock::now();
        s = db->Write(WriteOptions(), &batch);
        auto write_end = std::chrono::steady_clock::now();
        if (!s.ok()) {
            std::cerr << "Write failed: " << s.ToString() << std::endl;
            break;
        }

        uint64_t l0 = CompactionController::GetL0Files(db);
        uint64_t stall_total = CompactionController::GetStallMetric(options.statistics);
        uint64_t stall_delta = stall_total - prev_stall_micros;
        prev_stall_micros = stall_total;
        uint64_t pending_bytes = CompactionController::GetCompactionPendingBytes(db);
        uint64_t is_write_stopped = CompactionController::GetIsWriteStopped(db);
        uint64_t actual_delayed_write_rate =
            CompactionController::GetActualDelayedWriteRate(db);
        uint64_t num_running_compactions =
            CompactionController::GetNumRunningCompactions(db);
        double timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        double batch_latency_ms = std::chrono::duration<double, std::milli>(
            write_end - write_begin).count();
        double process_cpu_user_sec = 0.0;
        double process_cpu_sys_sec = 0.0;
        if (!GetProcessCpuSeconds(&process_cpu_user_sec, &process_cpu_sys_sec)) {
            std::cerr << "Warning: getrusage failed; CPU fields set to 0"
                      << std::endl;
        }
        int bg_jobs = rocksdb_max_background_jobs;
        int subcompactions_now = rocksdb_max_subcompactions;
        int l0_slowdown_now = rocksdb_l0_slowdown_trigger;
        int l0_stop_now = rocksdb_l0_stop_trigger;
        int arm_id = -1;
        int profile_id = -1;
        double last_fast_reward = 0.0;
        double last_profile_reward = 0.0;
        if (ctl) {
            bg_jobs = ctl->CurrentBgJobs();
            subcompactions_now = ctl->CurrentSubcompactions();
            l0_slowdown_now = ctl->CurrentL0Slowdown();
            l0_stop_now = ctl->CurrentL0Stop();
            arm_id = ctl->LastArmId();
            profile_id = ctl->LastProfileId();
            last_fast_reward = ctl->LastFastReward();
            last_profile_reward = ctl->LastProfileReward();
        }

        if (log.is_open()) {
            log << run_id << ","
                << phase_idx_for_log << ","
                << timestamp << ","
                << l0 << ","
                << stall_total << ","
                << stall_delta << ","
                << pending_bytes << ","
                << is_write_stopped << ","
                << actual_delayed_write_rate << ","
                << num_running_compactions << ","
                << batch_payload_bytes << ","
                << batch_latency_ms << ","
                << bg_jobs << ","
                << process_cpu_user_sec << ","
                << process_cpu_sys_sec << ","
                << subcompactions_now << ","
                << l0_slowdown_now << ","
                << l0_stop_now << ","
                << arm_id << ","
                << profile_id << ","
                << last_fast_reward << ","
                << last_profile_reward << "\n";
            log.flush();
        }

        ++rounds_done;
        ++rounds_in_phase;

        std::this_thread::sleep_for(std::chrono::milliseconds(current_sleep_ms));
    }

    if (!g_stop.load()) {
        g_stop.store(true);
    }

    if (ctl) {
        ctl->Stop();
    }

    Status close_status = db->Close();
    if (!close_status.ok()) {
        std::cerr << "Close failed: " << close_status.ToString() << std::endl;
        return 1;
    }
    db_guard.reset();

    Status destroy_status = DestroyDB(DB_PATH, options);
    if (!destroy_status.ok()) {
        std::cerr << "DestroyDB failed: " << destroy_status.ToString()
                  << std::endl;
        return 1;
    }

    return 0;
}