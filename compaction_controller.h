#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/status.h"

class CompactionController {
public:
    static constexpr int LOW_PRESSURE_CYCLES_REQUIRED = 10;
    static inline bool TryGetUint64Property(rocksdb::DB* db,
                                            const std::string& key,
                                            uint64_t* value) {
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

    static inline uint64_t GetL0Files(rocksdb::DB* db) {
        uint64_t l0_files = 0;
        if (TryGetUint64Property(db, "rocksdb.num-files-at-level0", &l0_files)) {
            return l0_files;
        }
        return 0;
    }

    static inline uint64_t GetCompactionPendingBytes(rocksdb::DB* db) {
        uint64_t pending = 0;
        if (TryGetUint64Property(
                db, "rocksdb.estimate-pending-compaction-bytes", &pending)) {
            return pending;
        }
        return 0;
    }

    static inline uint64_t GetIsWriteStopped(rocksdb::DB* db) {
        uint64_t stopped = 0;
        if (db->GetIntProperty("rocksdb.is-write-stopped", &stopped)) {
            return stopped;
        }
        return 0;
    }

    static inline uint64_t GetActualDelayedWriteRate(rocksdb::DB* db) {
        uint64_t rate = 0;
        if (db->GetIntProperty("rocksdb.actual-delayed-write-rate", &rate)) {
            return rate;
        }
        return 0;
    }

    static inline uint64_t GetNumRunningCompactions(rocksdb::DB* db) {
        uint64_t running = 0;
        if (db->GetIntProperty("rocksdb.num-running-compactions", &running)) {
            return running;
        }
        return 0;
    }

    static inline uint64_t GetStallMicrosTotal(rocksdb::DB* db) {
        uint64_t stall_micros = 0;
        if (TryGetUint64Property(db, "rocksdb.stall-micros", &stall_micros)) {
            return stall_micros;
        }
        return 0;
    }

    CompactionController(rocksdb::DB* db,
                            std::atomic<bool>* stop_flag,
                            int l0_compaction_trigger,
                            int l0_slowdown_trigger,
                            int low_bg_jobs,
                            int high_bg_jobs,
                            int current_bg_jobs,
                            int interval_sec,
                            int cooldown_sec)
        : db_(db),
            stop_flag_(stop_flag),
            l0_compaction_trigger_(l0_compaction_trigger),
            l0_slowdown_trigger_(l0_slowdown_trigger),
            low_bg_jobs_(low_bg_jobs),
            high_bg_jobs_(high_bg_jobs),
            interval_sec_(interval_sec),
            cooldown_sec_(cooldown_sec),
            current_bg_jobs_(current_bg_jobs),
            last_switch_tp_(std::chrono::steady_clock::now()) {}

    void Start() {
        worker_ = std::thread(&CompactionController::Run, this);
    }

    void Stop() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    int CurrentBgJobs() const {
        return current_bg_jobs_.load();
    }

private:
    bool ApplyMaxBackgroundJobs(int jobs) {
        if (jobs == current_bg_jobs_) {
            return false;  // 避免重复设置
        }

        rocksdb::Status s = db_->SetDBOptions({
            {"max_background_jobs", std::to_string(jobs)}
        });

        if (!s.ok()) {
            std::cerr << "[controller] SetDBOptions failed: "
                        << s.ToString() << std::endl;
            return false;
        }

        current_bg_jobs_.store(jobs);
        std::cout << "[controller] applied max_background_jobs="
                    << jobs << std::endl;
        return true;
    }

    int DecideTargetJobs(uint64_t l0,
                         uint64_t stall_micros_delta,
                         uint64_t pending_bytes,
                         uint64_t delayed_write_rate,
                         uint64_t is_write_stopped,
                         uint64_t num_running_compactions,
                         int current_jobs) {
        const bool should_scale_up =
            (delayed_write_rate > 0 ||
             is_write_stopped > 0 ||
             stall_micros_delta > 0 ||
             l0 > static_cast<uint64_t>(l0_slowdown_trigger_));

        const bool low_pressure_now =
            (stall_micros_delta == 0 &&
             pending_bytes == 0 &&
             num_running_compactions == 0 &&
             delayed_write_rate == 0 &&
             l0 < static_cast<uint64_t>(l0_compaction_trigger_));
        if (low_pressure_now) {
            ++low_pressure_streak_;
        } else {
            low_pressure_streak_ = 0;
        }
        const bool should_scale_down =
            low_pressure_streak_ >= LOW_PRESSURE_CYCLES_REQUIRED;

        if (current_jobs != high_bg_jobs_ && should_scale_up) {
            return high_bg_jobs_;
        }
        if (current_jobs != low_bg_jobs_ && should_scale_down) {
            return low_bg_jobs_;
        }
        return current_jobs;
    }

    void Run() {
        while (!stop_flag_->load()) {
            uint64_t l0 = GetL0Files(db_);
            uint64_t pending_bytes = GetCompactionPendingBytes(db_);
            uint64_t is_write_stopped = GetIsWriteStopped(db_);
            uint64_t delayed_write_rate = GetActualDelayedWriteRate(db_);
            uint64_t num_running_compactions = GetNumRunningCompactions(db_);
            uint64_t stall_micros_total = GetStallMicrosTotal(db_);
            uint64_t stall_micros_delta =
                stall_micros_total >= prev_stall_micros_
                    ? (stall_micros_total - prev_stall_micros_)
                    : 0;
            prev_stall_micros_ = stall_micros_total;
            const int current_jobs = current_bg_jobs_.load();
            int target_jobs = DecideTargetJobs(l0,
                                               stall_micros_delta,
                                               pending_bytes,
                                               delayed_write_rate,
                                               is_write_stopped,
                                               num_running_compactions,
                                               current_jobs);

            std::cout << "[controller] L0 files=" << l0
                        << ", l0_compaction_trigger=" << l0_compaction_trigger_
                        << ", l0_slowdown_trigger=" << l0_slowdown_trigger_
                        << ", stall_micros_delta=" << stall_micros_delta
                        << ", pending_bytes=" << pending_bytes
                        << ", delayed_rate=" << delayed_write_rate
                        << ", is_write_stopped=" << is_write_stopped
                        << ", num_running_compactions=" << num_running_compactions
                        << ", low_pressure_streak=" << low_pressure_streak_
                        << ", current max_background_jobs=" << current_jobs
                        << ", target max_background_jobs=" << target_jobs
                        << std::endl;
            auto now = std::chrono::steady_clock::now();
            bool cooldown_blocked =
                (target_jobs != current_jobs &&
                 std::chrono::duration_cast<std::chrono::seconds>(
                     now - last_switch_tp_).count() < cooldown_sec_);
            if (cooldown_blocked) {
                std::cout << "[controller] cooldown active, skip switching"
                          << std::endl;
            } else if (ApplyMaxBackgroundJobs(target_jobs)) {
                last_switch_tp_ = now;
            }

            for (int i = 0; i < interval_sec_ && !stop_flag_->load(); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        std::cout << "[controller] stopping" << std::endl;
    }

private:
    rocksdb::DB* db_;
    std::atomic<bool>* stop_flag_;
    int l0_compaction_trigger_;
    int l0_slowdown_trigger_;
    int low_bg_jobs_;
    int high_bg_jobs_;
    int interval_sec_;
    int cooldown_sec_;
    int low_pressure_streak_ = 0;
    uint64_t prev_stall_micros_ = 0;
    std::atomic<int> current_bg_jobs_;
    std::chrono::steady_clock::time_point last_switch_tp_;
    std::thread worker_;
};