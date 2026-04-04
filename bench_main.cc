#include <chrono>
#include <csignal>
#include <cstdint>
#include <atomic>
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
constexpr int DEFAULT_ROCKSDB_MAX_BACKGROUND_JOBS = 2;
constexpr uint64_t DEFAULT_ROCKSDB_WRITE_BUFFER_SIZE_MB = 4;
constexpr int DEFAULT_ROCKSDB_L0_COMPACTION_TRIGGER = 8;
constexpr int DEFAULT_ROCKSDB_L0_SLOWDOWN_TRIGGER = 12;
constexpr int DEFAULT_ROCKSDB_L0_STOP_TRIGGER = 16;
constexpr int DEFAULT_CTRL_LOW_BG_JOBS = 2;
constexpr int DEFAULT_CTRL_HIGH_BG_JOBS = 6;
constexpr int DEFAULT_CTRL_INTERVAL_SEC = 2;
constexpr int DEFAULT_CTRL_COOLDOWN_SEC = 6;
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
        << "  --rocksdb-max-background-jobs=N     Initial max_background_jobs at startup (default: "
        << DEFAULT_ROCKSDB_MAX_BACKGROUND_JOBS << ")\n"
        << "  --rocksdb-increase-parallelism=N    Override IncreaseParallelism thread count (default: auto: controller?ctrl_high_bg_jobs:rocksdb_max_background_jobs)\n"
        << "  --rocksdb-write-buffer-size-mb=N    write_buffer_size in MiB (default: "
        << DEFAULT_ROCKSDB_WRITE_BUFFER_SIZE_MB << ")\n"
        << "  --rocksdb-l0-compaction-trigger=N   level0_file_num_compaction_trigger (default: "
        << DEFAULT_ROCKSDB_L0_COMPACTION_TRIGGER << ")\n"
        << "  --rocksdb-l0-slowdown-trigger=N     level0_slowdown_writes_trigger (default: "
        << DEFAULT_ROCKSDB_L0_SLOWDOWN_TRIGGER << ")\n"
        << "  --rocksdb-l0-stop-trigger=N         level0_stop_writes_trigger (default: "
        << DEFAULT_ROCKSDB_L0_STOP_TRIGGER << ")\n"
        << "ClearCtrl options:\n"
        << "  --ctrl-low-bg-jobs=N                Low max_background_jobs (default: "
        << DEFAULT_CTRL_LOW_BG_JOBS << ")\n"
        << "  --ctrl-high-bg-jobs=N               High max_background_jobs (default: "
        << DEFAULT_CTRL_HIGH_BG_JOBS << ")\n"
        << "  --ctrl-interval-sec=N               Control interval seconds (default: "
        << DEFAULT_CTRL_INTERVAL_SEC << ")\n"
        << "  --ctrl-cooldown-sec=N               Switch cooldown seconds (default: "
        << DEFAULT_CTRL_COOLDOWN_SEC << ")\n"
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


int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const std::string DB_PATH = "./testdb";
    bool controller_enabled = DEFAULT_CONTROLLER_ENABLED;
    std::string runner_schedule;
    std::string log_path;
    std::string run_id;

    int rocksdb_max_background_jobs = DEFAULT_ROCKSDB_MAX_BACKGROUND_JOBS;
    int rocksdb_increase_parallelism_override = 0;  // 0 means auto
    uint64_t rocksdb_write_buffer_size_mb = DEFAULT_ROCKSDB_WRITE_BUFFER_SIZE_MB;
    int rocksdb_l0_compaction_trigger = DEFAULT_ROCKSDB_L0_COMPACTION_TRIGGER;
    int rocksdb_l0_slowdown_trigger = DEFAULT_ROCKSDB_L0_SLOWDOWN_TRIGGER;
    int rocksdb_l0_stop_trigger = DEFAULT_ROCKSDB_L0_STOP_TRIGGER;

    int ctrl_low_bg_jobs = DEFAULT_CTRL_LOW_BG_JOBS;
    int ctrl_high_bg_jobs = DEFAULT_CTRL_HIGH_BG_JOBS;
    int ctrl_interval_sec = DEFAULT_CTRL_INTERVAL_SEC;
    int ctrl_cooldown_sec = DEFAULT_CTRL_COOLDOWN_SEC;

    enum OptionId {
        OPT_CONTROLLER = 1000,
        OPT_RUNNER_SCHEDULE,
        OPT_LOG_PATH,
        OPT_RUN_ID,
        OPT_RDB_MAX_BACKGROUND_JOBS,
        OPT_RDB_INCREASE_PARALLELISM,
        OPT_RDB_WRITE_BUFFER_SIZE_MB,
        OPT_RDB_L0_COMPACTION_TRIGGER,
        OPT_RDB_L0_SLOWDOWN_TRIGGER,
        OPT_RDB_L0_STOP_TRIGGER,
        OPT_CTRL_LOW_BG_JOBS,
        OPT_CTRL_HIGH_BG_JOBS,
        OPT_CTRL_INTERVAL_SEC,
        OPT_CTRL_COOLDOWN_SEC,
    };

    static struct option long_options[] = {
        {"controller", required_argument, nullptr, OPT_CONTROLLER},
        {"runner-schedule", required_argument, nullptr, OPT_RUNNER_SCHEDULE},
        {"log-path", required_argument, nullptr, OPT_LOG_PATH},
        {"run-id", required_argument, nullptr, OPT_RUN_ID},
        {"rocksdb-max-background-jobs", required_argument, nullptr,
         OPT_RDB_MAX_BACKGROUND_JOBS},
        {"rocksdb-increase-parallelism", required_argument, nullptr,
         OPT_RDB_INCREASE_PARALLELISM},
        {"rocksdb-write-buffer-size-mb", required_argument, nullptr,
         OPT_RDB_WRITE_BUFFER_SIZE_MB},
        {"rocksdb-l0-compaction-trigger", required_argument, nullptr,
         OPT_RDB_L0_COMPACTION_TRIGGER},
        {"rocksdb-l0-slowdown-trigger", required_argument, nullptr,
         OPT_RDB_L0_SLOWDOWN_TRIGGER},
        {"rocksdb-l0-stop-trigger", required_argument, nullptr,
         OPT_RDB_L0_STOP_TRIGGER},
        {"ctrl-low-bg-jobs", required_argument, nullptr, OPT_CTRL_LOW_BG_JOBS},
        {"ctrl-high-bg-jobs", required_argument, nullptr, OPT_CTRL_HIGH_BG_JOBS},
        {"ctrl-interval-sec", required_argument, nullptr, OPT_CTRL_INTERVAL_SEC},
        {"ctrl-cooldown-sec", required_argument, nullptr, OPT_CTRL_COOLDOWN_SEC},
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
            case OPT_RDB_MAX_BACKGROUND_JOBS:
                if (!parse_int_arg("--rocksdb-max-background-jobs", optarg,
                                   &rocksdb_max_background_jobs)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
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
            case OPT_RDB_L0_SLOWDOWN_TRIGGER:
                if (!parse_int_arg("--rocksdb-l0-slowdown-trigger", optarg,
                                   &rocksdb_l0_slowdown_trigger)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_RDB_L0_STOP_TRIGGER:
                if (!parse_int_arg("--rocksdb-l0-stop-trigger", optarg,
                                   &rocksdb_l0_stop_trigger)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_LOW_BG_JOBS:
                if (!parse_int_arg("--ctrl-low-bg-jobs", optarg,
                                   &ctrl_low_bg_jobs)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_HIGH_BG_JOBS:
                if (!parse_int_arg("--ctrl-high-bg-jobs", optarg,
                                   &ctrl_high_bg_jobs)) {
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
    int rocksdb_increase_parallelism = 
        rocksdb_increase_parallelism_override ? rocksdb_increase_parallelism_override :
        (controller_enabled ? ctrl_high_bg_jobs : rocksdb_max_background_jobs);

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
              << ", write_buffer_size_mb=" << rocksdb_write_buffer_size_mb
              << ", l0_compaction_trigger=" << rocksdb_l0_compaction_trigger
              << ", l0_slowdown_trigger=" << rocksdb_l0_slowdown_trigger
              << ", l0_stop_trigger=" << rocksdb_l0_stop_trigger << "\n"
              << "[config][ctrl] low_bg_jobs=" << ctrl_low_bg_jobs
              << ", high_bg_jobs=" << ctrl_high_bg_jobs
              << ", interval_sec=" << ctrl_interval_sec
              << ", cooldown_sec=" << ctrl_cooldown_sec << std::endl;

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
            << "    \"write_buffer_size_mb\": " << rocksdb_write_buffer_size_mb
            << ",\n"
            << "    \"l0_compaction_trigger\": "
            << rocksdb_l0_compaction_trigger << ",\n"
            << "    \"l0_slowdown_trigger\": " << rocksdb_l0_slowdown_trigger
            << ",\n"
            << "    \"l0_stop_trigger\": " << rocksdb_l0_stop_trigger << "\n"
            << "  },\n"
            << "  \"clearctrl\": {\n"
            << "    \"low_bg_jobs\": " << ctrl_low_bg_jobs << ",\n"
            << "    \"high_bg_jobs\": " << ctrl_high_bg_jobs << ",\n"
            << "    \"interval_sec\": " << ctrl_interval_sec << ",\n"
            << "    \"cooldown_sec\": " << ctrl_cooldown_sec << "\n"
            << "  }\n"
            << "}\n";
    }

    Options options;
    options.create_if_missing = true;

    // 先给一个比较容易观察 compaction/backlog 的配置
    options.IncreaseParallelism(rocksdb_increase_parallelism);
    options.max_background_jobs = rocksdb_max_background_jobs;
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

    // 控制器开关：on 为动态调参，off 为 baseline
    CompactionController ctl(
        db,
        options.statistics,
        &g_stop,
        rocksdb_l0_compaction_trigger,
        rocksdb_l0_slowdown_trigger,
        ctrl_low_bg_jobs,
        ctrl_high_bg_jobs,
        rocksdb_max_background_jobs,
        ctrl_interval_sec,
        ctrl_cooldown_sec
    );
    if (controller_enabled) {
        ctl.Start();
    }

    std::ofstream log(log_path, std::ios::trunc);
    if (log.is_open()) {
        log << "run_id,phase_idx,timestamp,l0_files,stall_micros_total,stall_micros_delta,"
               "compaction_pending_bytes,is_write_stopped,actual_delayed_write_rate,"
               "num_running_compactions,batch_payload_bytes,batch_latency_ms,bg_jobs,"
               "process_cpu_user_sec,process_cpu_sys_sec\n";
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
        int bg_jobs = controller_enabled ? ctl.CurrentBgJobs() : rocksdb_max_background_jobs;

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
                << process_cpu_sys_sec << "\n";
            log.flush();
        }

        ++rounds_done;
        ++rounds_in_phase;

        std::this_thread::sleep_for(std::chrono::milliseconds(current_sleep_ms));
    }

    if (!g_stop.load()) {
        g_stop.store(true);
    }

    ctl.Stop();

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