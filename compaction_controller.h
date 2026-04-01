#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/status.h"

class CompactionController {
public:
    CompactionController(rocksdb::DB* db,
                            std::atomic<bool>* stop_flag,
                            int low_threshold,
                            int high_threshold,
                            int low_bg_jobs,
                            int high_bg_jobs,
                            int current_bg_jobs,
                            int interval_sec,
                            int cooldown_sec)
        : db_(db),
            stop_flag_(stop_flag),
            low_threshold_(low_threshold),
            high_threshold_(high_threshold),
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
    uint64_t GetL0Files() {
        std::string prop;
        bool ok = db_->GetProperty("rocksdb.num-files-at-level0", &prop);
        if (!ok) {
            std::cerr << "[controller] failed to read property "
                        << "rocksdb.num-files-at-level0" << std::endl;
            return 0;
        }
        return std::stoull(prop);
    }

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

    int DecideTargetJobs(uint64_t l0, int current_jobs) const {
        if (current_jobs != high_bg_jobs_ &&
            l0 >= static_cast<uint64_t>(high_threshold_)) {
            return high_bg_jobs_;
        }
        if (current_jobs != low_bg_jobs_ &&
            l0 <= static_cast<uint64_t>(low_threshold_)) {
            return low_bg_jobs_;
        }
        return current_jobs;
    }

    void Run() {
        while (!stop_flag_->load()) {
            uint64_t l0 = GetL0Files();
            const int current_jobs = current_bg_jobs_.load();
            int target_jobs = DecideTargetJobs(l0, current_jobs);

            std::cout << "[controller] L0 files=" << l0
                        << ", low_threshold=" << low_threshold_
                        << ", high_threshold=" << high_threshold_
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
    int low_threshold_;
    int high_threshold_;
    int low_bg_jobs_;
    int high_bg_jobs_;
    int interval_sec_;
    int cooldown_sec_;
    std::atomic<int> current_bg_jobs_;
    std::chrono::steady_clock::time_point last_switch_tp_;
    std::thread worker_;
};