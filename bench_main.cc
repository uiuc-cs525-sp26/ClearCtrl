#include <chrono>
#include <csignal>
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"

#include "compaction_controller.h"

using namespace ROCKSDB_NAMESPACE;

static std::atomic<bool> g_stop{false};

void PrintUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [runner options] [rocksdb options] [ctrl options]\n"
        << "Runner options:\n"
        << "  --controller=on|off                 Enable/disable dynamic controller (default: on)\n"
        << "  --rounds=N                          Number of write rounds; 0 means run until Ctrl+C (default: 0)\n"
        << "  --runner-sleep-ms=N                 Sleep interval between write rounds in ms (default: 100)\n"
        << "  --runner-batch-ops=N                Number of Put ops per round (default: 1000)\n"
        << "  --log-path=FILE                     Output CSV path (default: logs/metrics_<run_id>.csv)\n"
        << "  --run-id=ID                         Explicit run id (default: auto timestamp)\n"
        << "RocksDB options:\n"
        << "  --rocksdb-max-background-jobs=N     Initial max_background_jobs at startup (default: 2)\n"
        << "  --rocksdb-write-buffer-size-mb=N    write_buffer_size in MiB (default: 4)\n"
        << "  --rocksdb-l0-compaction-trigger=N   level0_file_num_compaction_trigger (default: 8)\n"
        << "  --rocksdb-l0-slowdown-trigger=N     level0_slowdown_writes_trigger (default: 12)\n"
        << "  --rocksdb-l0-stop-trigger=N         level0_stop_writes_trigger (default: 16)\n"
        << "ClearCtrl options:\n"
        << "  --ctrl-low-threshold=N              Hysteresis low threshold (default: 4)\n"
        << "  --ctrl-high-threshold=N             Hysteresis high threshold (default: 10)\n"
        << "  --ctrl-low-bg-jobs=N                Low max_background_jobs (default: 2)\n"
        << "  --ctrl-high-bg-jobs=N               High max_background_jobs (default: 6)\n"
        << "  --ctrl-interval-sec=N               Control interval seconds (default: 2)\n"
        << "  --ctrl-cooldown-sec=N               Switch cooldown seconds (default: 6)\n"
        << "  -h, --help            Show help\n";
}

void signal_handler(int) {
    g_stop.store(true);
}

bool TryGetUint64Property(DB* db, const std::string& key, uint64_t* value) {
    std::string prop;
    if (!db->GetProperty(key, &prop)) {
        return false;
    }
    try {
        *value = std::stoull(prop);
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t GetL0Files(DB* db) {
    std::string prop;
    if (!db->GetProperty("rocksdb.num-files-at-level0", &prop)) {
        return 0;
    }
    return std::stoull(prop);
}

uint64_t GetStallMetric(const std::shared_ptr<Statistics>& statistics) {
    if (!statistics) {
        return 0;
    }
    return statistics->getTickerCount(STALL_MICROS);
}

uint64_t GetCompactionPendingBytes(DB* db) {
    uint64_t pending = 0;
    if (TryGetUint64Property(db, "rocksdb.estimate-pending-compaction-bytes", &pending)) {
        return pending;
    }
    return 0;
}

uint64_t GetIsWriteStopped(DB* db) {
    uint64_t stopped = 0;
    if (db->GetIntProperty("rocksdb.is-write-stopped", &stopped)) {
        return stopped;
    }
    return 0;
}

uint64_t GetActualDelayedWriteRate(DB* db) {
    uint64_t rate = 0;
    if (db->GetIntProperty("rocksdb.actual-delayed-write-rate", &rate)) {
        return rate;
    }
    return 0;
}

uint64_t GetNumRunningCompactions(DB* db) {
    uint64_t running = 0;
    if (db->GetIntProperty("rocksdb.num-running-compactions", &running)) {
        return running;
    }
    return 0;
}


int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const std::string kDBPath = "./testdb";
    bool controller_enabled = true;
    uint64_t max_rounds = 0;
    uint64_t runner_sleep_ms = 100;
    uint64_t runner_batch_ops = 1000;
    std::string log_path;
    std::string run_id;

    int rocksdb_max_background_jobs = 2;
    uint64_t rocksdb_write_buffer_size_mb = 4;
    int rocksdb_l0_compaction_trigger = 8;
    int rocksdb_l0_slowdown_trigger = 12;
    int rocksdb_l0_stop_trigger = 16;

    int ctrl_low_threshold = 4;
    int ctrl_high_threshold = 10;
    int ctrl_low_bg_jobs = 2;
    int ctrl_high_bg_jobs = 6;
    int ctrl_interval_sec = 2;
    int ctrl_cooldown_sec = 6;

    enum OptionId {
        OPT_CONTROLLER = 1000,
        OPT_ROUNDS,
        OPT_RUNNER_SLEEP_MS,
        OPT_RUNNER_BATCH_OPS,
        OPT_LOG_PATH,
        OPT_RUN_ID,
        OPT_RDB_MAX_BACKGROUND_JOBS,
        OPT_RDB_WRITE_BUFFER_SIZE_MB,
        OPT_RDB_L0_COMPACTION_TRIGGER,
        OPT_RDB_L0_SLOWDOWN_TRIGGER,
        OPT_RDB_L0_STOP_TRIGGER,
        OPT_CTRL_LOW_THRESHOLD,
        OPT_CTRL_HIGH_THRESHOLD,
        OPT_CTRL_LOW_BG_JOBS,
        OPT_CTRL_HIGH_BG_JOBS,
        OPT_CTRL_INTERVAL_SEC,
        OPT_CTRL_COOLDOWN_SEC,
    };

    static struct option long_options[] = {
        {"controller", required_argument, nullptr, OPT_CONTROLLER},
        {"rounds", required_argument, nullptr, OPT_ROUNDS},
        {"runner-sleep-ms", required_argument, nullptr, OPT_RUNNER_SLEEP_MS},
        {"runner-batch-ops", required_argument, nullptr, OPT_RUNNER_BATCH_OPS},
        {"log-path", required_argument, nullptr, OPT_LOG_PATH},
        {"run-id", required_argument, nullptr, OPT_RUN_ID},
        {"rocksdb-max-background-jobs", required_argument, nullptr,
         OPT_RDB_MAX_BACKGROUND_JOBS},
        {"rocksdb-write-buffer-size-mb", required_argument, nullptr,
         OPT_RDB_WRITE_BUFFER_SIZE_MB},
        {"rocksdb-l0-compaction-trigger", required_argument, nullptr,
         OPT_RDB_L0_COMPACTION_TRIGGER},
        {"rocksdb-l0-slowdown-trigger", required_argument, nullptr,
         OPT_RDB_L0_SLOWDOWN_TRIGGER},
        {"rocksdb-l0-stop-trigger", required_argument, nullptr,
         OPT_RDB_L0_STOP_TRIGGER},
        {"ctrl-low-threshold", required_argument, nullptr, OPT_CTRL_LOW_THRESHOLD},
        {"ctrl-high-threshold", required_argument, nullptr,
         OPT_CTRL_HIGH_THRESHOLD},
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
            case OPT_ROUNDS: {
                if (!parse_u64_arg("--rounds", optarg, &max_rounds)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            }
            case OPT_RUNNER_SLEEP_MS:
                if (!parse_u64_arg("--runner-sleep-ms", optarg,
                                   &runner_sleep_ms)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_RUNNER_BATCH_OPS:
                if (!parse_u64_arg("--runner-batch-ops", optarg,
                                   &runner_batch_ops)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
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
            case OPT_CTRL_LOW_THRESHOLD:
                if (!parse_int_arg("--ctrl-low-threshold", optarg,
                                   &ctrl_low_threshold)) {
                    PrintUsage(argv[0]);
                    return 1;
                }
                break;
            case OPT_CTRL_HIGH_THRESHOLD:
                if (!parse_int_arg("--ctrl-high-threshold", optarg,
                                   &ctrl_high_threshold)) {
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
    const int rocksdb_increase_parallelism =
        controller_enabled ? ctrl_high_bg_jobs : rocksdb_max_background_jobs;

    std::cout << "[config] run_id=" << run_id
              << ", controller=" << (controller_enabled ? "on" : "off")
              << ", rounds=" << max_rounds
              << ", runner_sleep_ms=" << runner_sleep_ms
              << ", runner_batch_ops=" << runner_batch_ops
              << ", log_path=" << log_path << "\n"
              << "[config] run_config_path=" << run_config_path.string() << "\n"
              << "[config][rocksdb] increase_parallelism="
              << rocksdb_increase_parallelism
              << ", max_background_jobs=" << rocksdb_max_background_jobs
              << ", write_buffer_size_mb=" << rocksdb_write_buffer_size_mb
              << ", l0_compaction_trigger=" << rocksdb_l0_compaction_trigger
              << ", l0_slowdown_trigger=" << rocksdb_l0_slowdown_trigger
              << ", l0_stop_trigger=" << rocksdb_l0_stop_trigger << "\n"
              << "[config][ctrl] low_threshold=" << ctrl_low_threshold
              << ", high_threshold=" << ctrl_high_threshold
              << ", low_bg_jobs=" << ctrl_low_bg_jobs
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
            << "    \"rounds\": " << max_rounds << ",\n"
            << "    \"sleep_ms\": " << runner_sleep_ms << ",\n"
            << "    \"batch_ops\": " << runner_batch_ops << ",\n"
            << "    \"log_path\": \"" << log_path << "\"\n"
            << "  },\n"
            << "  \"rocksdb\": {\n"
            << "    \"increase_parallelism\": " << rocksdb_increase_parallelism
            << ",\n"
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
            << "    \"low_threshold\": " << ctrl_low_threshold << ",\n"
            << "    \"high_threshold\": " << ctrl_high_threshold << ",\n"
            << "    \"low_bg_jobs\": " << ctrl_low_bg_jobs << ",\n"
            << "    \"high_bg_jobs\": " << ctrl_high_bg_jobs << ",\n"
            << "    \"interval_sec\": " << ctrl_interval_sec << ",\n"
            << "    \"cooldown_sec\": " << ctrl_cooldown_sec << "\n"
            << "  }\n"
            << "}\n";
    }

    Options options;
    options.create_if_missing = true;
    options.statistics = CreateDBStatistics();

    // 先给一个比较容易观察 compaction/backlog 的配置
    options.IncreaseParallelism(rocksdb_increase_parallelism);
    options.max_background_jobs = rocksdb_max_background_jobs;
    options.OptimizeLevelStyleCompaction();

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
    Status s = DB::Open(options, kDBPath, &db);
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
        &g_stop,
        ctrl_low_threshold,
        ctrl_high_threshold,
        ctrl_low_bg_jobs,
        ctrl_high_bg_jobs,
        ctrl_interval_sec,
        ctrl_cooldown_sec
    );
    if (controller_enabled) {
        ctl.Start();
    }

    std::ofstream log(log_path, std::ios::trunc);
    if (log.is_open()) {
        log << "run_id,timestamp,l0_files,stall_micros_total,stall_micros_delta,"
               "compaction_pending_bytes,is_write_stopped,actual_delayed_write_rate,"
               "num_running_compactions,write_throughput_mb_s,batch_latency_ms,bg_jobs\n";
    }

    // 前台 workload：持续写入，推动 flush/L0/compaction 发生
    uint64_t i = 0;
    uint64_t prev_stall_micros = 0;
    uint64_t rounds_done = 0;

    for (; !g_stop.load() && (max_rounds == 0 || rounds_done < max_rounds);
         ++rounds_done) {
        WriteBatch batch;
        size_t batch_payload_bytes = 0;
        for (uint64_t j = 0; j < runner_batch_ops; ++j) {
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

        uint64_t l0 = GetL0Files(db);
        uint64_t stall_total = GetStallMetric(options.statistics);
        uint64_t stall_delta = stall_total - prev_stall_micros;
        prev_stall_micros = stall_total;
        uint64_t pending_bytes = GetCompactionPendingBytes(db);
        uint64_t is_write_stopped = GetIsWriteStopped(db);
        uint64_t actual_delayed_write_rate = GetActualDelayedWriteRate(db);
        uint64_t num_running_compactions = GetNumRunningCompactions(db);
        double timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        double batch_latency_ms = std::chrono::duration<double, std::milli>(
            write_end - write_begin).count();
        double write_throughput_mb_s = 0.0;
        if (batch_latency_ms > 0.0) {
            write_throughput_mb_s =
                (static_cast<double>(batch_payload_bytes) / (1024.0 * 1024.0)) /
                (batch_latency_ms / 1000.0);
        }
        int bg_jobs = controller_enabled ? ctl.CurrentBgJobs() : rocksdb_max_background_jobs;

        if (log.is_open()) {
            log << run_id << ","
                << timestamp << ","
                << l0 << ","
                << stall_total << ","
                << stall_delta << ","
                << pending_bytes << ","
                << is_write_stopped << ","
                << actual_delayed_write_rate << ","
                << num_running_compactions << ","
                << write_throughput_mb_s << ","
                << batch_latency_ms << ","
                << bg_jobs << "\n";
            log.flush();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(runner_sleep_ms));
    }

    if (!g_stop.load() && max_rounds > 0 && rounds_done >= max_rounds) {
        g_stop.store(true);
    }

    ctl.Stop();

    Status close_status = db->Close();
    if (!close_status.ok()) {
        std::cerr << "Close failed: " << close_status.ToString() << std::endl;
        return 1;
    }

    return 0;
}