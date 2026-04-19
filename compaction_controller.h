#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"

#include "linucb.h"

// Discrete arm of the fast (high-frequency) bandit.
// Controls global DB-level options:
//   - max_background_jobs   (compaction + flush worker pool size)
//   - max_subcompactions    (intra-compaction parallelism)
struct BanditFastArm {
    int max_background_jobs;
    int max_subcompactions;
};

// Discrete arm of the slow (low-frequency) "profile" bandit.
// Controls column-family-level write-rate-limit triggers:
//   - level0_slowdown_writes_trigger
//   - level0_stop_writes_trigger
struct BanditProfile {
    int l0_slowdown_trigger;
    int l0_stop_trigger;
};

// Two-level LinUCB compaction controller.
//
// Decision loop runs every interval_sec:
//   1. Read current pressure signals (l0, pending, stall delta, throughput).
//   2. Compute reward for the *previously* chosen fast/profile arms.
//   3. Update the corresponding LinUCB models with (context, reward).
//   4. Pick a new fast arm; pick a new profile arm every
//      profile_interval_cycles fast cycles.
//   5. Apply the selected arm via SetDBOptions / SetOptions, respecting
//      cooldown to bound the SetOptions write-amp.
//
// Caller must ensure profile_interval_cycles * interval_sec >=
// 2 * cooldown_sec; otherwise the profile bandit's effective decision
// horizon is shorter than two fast-bandit cooldowns and the per-window
// aggregated reward becomes too noisy. The CLI in bench_main.cc enforces
// this and silently raises profile_interval_cycles when needed.
class CompactionController {
public:
    // ---- Property helpers (also used by the foreground logging loop). ----
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

    static inline uint64_t GetStallMetric(
        const std::shared_ptr<rocksdb::Statistics>& statistics) {
        if (!statistics) {
            return 0;
        }
        return statistics->getTickerCount(rocksdb::STALL_MICROS);
    }

    // 8-dim context vector (must stay in sync with BuildContext()):
    //   [0] bias (1.0)
    //   [1] L0 file count / l0_base_ (fixed compaction trigger,
    //       NOT the profile-controlled slowdown trigger)         (clip [0, 2])
    //   [2] compaction_pending_bytes / 64 MiB                    (clip [0, 2])
    //   [3] stall_micros_delta / interval_micros                 (clip [0, 1])
    //   [4] is_write_stopped indicator                           ({0, 1})
    //   [5] num_running_compactions / current_bg_jobs            (clip [0, 2])
    //   [6] recent throughput MB/s / throughput_norm             (clip [0, 2])
    //   [7] delayed_write_rate > 0 indicator                     ({0, 1})
    static constexpr int FEATURE_DIM = 8;

    CompactionController(rocksdb::DB* db,
                         std::shared_ptr<rocksdb::Statistics> statistics,
                         std::atomic<bool>* stop_flag,
                         std::vector<BanditFastArm> fast_arms,
                         std::vector<BanditProfile> profiles,
                         int initial_arm_id,
                         int initial_profile_id,
                         int interval_sec,
                         int cooldown_sec,
                         int profile_interval_cycles,
                         double bandit_alpha,
                         double reward_lambda_stall,
                         double reward_lambda_l0,
                         double reward_lambda_stop,
                         double reward_lambda_delay,
                         double reward_throughput_norm_mb_s,
                         int l0_compaction_trigger)
        : db_(db),
          statistics_(std::move(statistics)),
          stop_flag_(stop_flag),
          fast_arms_(std::move(fast_arms)),
          profiles_(std::move(profiles)),
          interval_sec_(interval_sec),
          cooldown_sec_(cooldown_sec),
          profile_interval_cycles_(std::max(1, profile_interval_cycles)),
          lambda_stall_(reward_lambda_stall),
          lambda_l0_(reward_lambda_l0),
          lambda_stop_(reward_lambda_stop),
          lambda_delay_(reward_lambda_delay),
          throughput_norm_mb_s_(reward_throughput_norm_mb_s > 0.0
                                    ? reward_throughput_norm_mb_s
                                    : 100.0),
          l0_base_(std::max(1, l0_compaction_trigger)),
          fast_bandit_(static_cast<int>(fast_arms_.size()), FEATURE_DIM,
                       bandit_alpha),
          profile_bandit_(static_cast<int>(profiles_.size()), FEATURE_DIM,
                          bandit_alpha),
          current_bg_jobs_(fast_arms_.at(initial_arm_id).max_background_jobs),
          current_subcompactions_(
              fast_arms_.at(initial_arm_id).max_subcompactions),
          current_l0_slowdown_(
              profiles_.at(initial_profile_id).l0_slowdown_trigger),
          current_l0_stop_(profiles_.at(initial_profile_id).l0_stop_trigger),
          last_arm_id_(initial_arm_id),
          last_profile_id_(initial_profile_id),
          last_fast_reward_(0.0),
          last_profile_reward_(0.0),
          last_switch_tp_(std::chrono::steady_clock::now()) {}

    // Returns false if EnsureInitialState() could not bring RocksDB into
    // the expected initial arm/profile state. In that case the worker
    // thread is NOT started and the caller should treat this as fatal --
    // running the bandit on top of an out-of-sync DB would credit the
    // initial arm with rewards earned under different knobs.
    bool Start() {
        if (!EnsureInitialState()) {
            std::cerr << "[bandit] fatal: failed to align RocksDB with "
                         "initial arm/profile; refusing to start controller"
                      << std::endl;
            return false;
        }
        worker_ = std::thread(&CompactionController::Run, this);
        return true;
    }

    // Verify that RocksDB's actual DBOptions / default-CF options match the
    // controller's expected initial fast arm and profile, and forcibly
    // re-apply via SetDBOptions / SetOptions if they don't. We can't reuse
    // ApplyFastArm / ApplyProfile here because they short-circuit on the
    // controller-tracked current_*_ values, which were initialized to the
    // *expected* values -- they would silently miss a real DB mismatch.
    //
    // Returns true iff every required SetDBOptions / SetOptions call (if
    // any) succeeded; false on any RocksDB API failure. On false, the
    // controller-tracked current_*_ values are left unchanged so that the
    // caller can still inspect them, but the worker should NOT be started.
    bool EnsureInitialState() {
        const BanditFastArm& arm = fast_arms_.at(last_arm_id_.load());
        const BanditProfile& prof = profiles_.at(last_profile_id_.load());
        bool ok = true;

        rocksdb::DBOptions actual_db = db_->GetDBOptions();
        const int db_bg = actual_db.max_background_jobs;
        const int db_sub = static_cast<int>(actual_db.max_subcompactions);
        if (db_bg != arm.max_background_jobs ||
            db_sub != arm.max_subcompactions) {
            std::cerr << "[bandit] initial fast arm[" << last_arm_id_.load()
                      << "] mismatch: db has bg=" << db_bg
                      << ",sub=" << db_sub
                      << " vs expected bg=" << arm.max_background_jobs
                      << ",sub=" << arm.max_subcompactions
                      << "; re-applying" << std::endl;
            rocksdb::Status s = db_->SetDBOptions({
                {"max_background_jobs",
                 std::to_string(arm.max_background_jobs)},
                {"max_subcompactions",
                 std::to_string(arm.max_subcompactions)},
            });
            if (!s.ok()) {
                std::cerr << "[bandit] SetDBOptions failed during initial "
                             "state alignment: "
                          << s.ToString() << std::endl;
                ok = false;
            } else {
                current_bg_jobs_.store(arm.max_background_jobs);
                current_subcompactions_.store(arm.max_subcompactions);
            }
        }

        rocksdb::Options actual_cf =
            db_->GetOptions(db_->DefaultColumnFamily());
        const int db_slow = actual_cf.level0_slowdown_writes_trigger;
        const int db_stop = actual_cf.level0_stop_writes_trigger;
        if (db_slow != prof.l0_slowdown_trigger ||
            db_stop != prof.l0_stop_trigger) {
            std::cerr << "[bandit] initial profile[" << last_profile_id_.load()
                      << "] mismatch: db has slow=" << db_slow
                      << ",stop=" << db_stop
                      << " vs expected slow=" << prof.l0_slowdown_trigger
                      << ",stop=" << prof.l0_stop_trigger
                      << "; re-applying" << std::endl;
            rocksdb::Status s = db_->SetOptions(
                db_->DefaultColumnFamily(),
                {
                    {"level0_slowdown_writes_trigger",
                     std::to_string(prof.l0_slowdown_trigger)},
                    {"level0_stop_writes_trigger",
                     std::to_string(prof.l0_stop_trigger)},
                });
            if (!s.ok()) {
                std::cerr << "[bandit] SetOptions failed during initial "
                             "state alignment: "
                          << s.ToString() << std::endl;
                ok = false;
            } else {
                current_l0_slowdown_.store(prof.l0_slowdown_trigger);
                current_l0_stop_.store(prof.l0_stop_trigger);
            }
        }

        return ok;
    }

    void Stop() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    int CurrentBgJobs() const { return current_bg_jobs_.load(); }
    int CurrentSubcompactions() const {
        return current_subcompactions_.load();
    }
    int CurrentL0Slowdown() const { return current_l0_slowdown_.load(); }
    int CurrentL0Stop() const { return current_l0_stop_.load(); }
    int LastArmId() const { return last_arm_id_.load(); }
    int LastProfileId() const { return last_profile_id_.load(); }
    // Most recent fast-bandit reward (per fast cycle, no stop/delay penalty).
    double LastFastReward() const { return last_fast_reward_.load(); }
    // Most recent profile-bandit reward (computed once per profile window,
    // includes stop / delay penalties). Stays at its previous value between
    // window closes.
    double LastProfileReward() const { return last_profile_reward_.load(); }

private:
    // Returns true if DB state already matches `arm` (no-op) or after a
    // successful SetDBOptions. Only RocksDB Status failure returns false.
    bool ApplyFastArm(const BanditFastArm& arm) {
        if (arm.max_background_jobs == current_bg_jobs_.load() &&
            arm.max_subcompactions == current_subcompactions_.load()) {
            return true;
        }
        rocksdb::Status s = db_->SetDBOptions({
            {"max_background_jobs", std::to_string(arm.max_background_jobs)},
            {"max_subcompactions", std::to_string(arm.max_subcompactions)},
        });
        if (!s.ok()) {
            std::cerr << "[bandit] SetDBOptions failed: " << s.ToString()
                      << std::endl;
            return false;
        }
        current_bg_jobs_.store(arm.max_background_jobs);
        current_subcompactions_.store(arm.max_subcompactions);
        std::cout << "[bandit] applied fast arm: bg_jobs="
                  << arm.max_background_jobs
                  << ", subcompactions=" << arm.max_subcompactions
                  << std::endl;
        return true;
    }

    // Returns true if CF options already match `prof` (no-op) or after a
    // successful SetOptions. Only RocksDB Status failure returns false.
    bool ApplyProfile(const BanditProfile& prof) {
        if (prof.l0_slowdown_trigger == current_l0_slowdown_.load() &&
            prof.l0_stop_trigger == current_l0_stop_.load()) {
            return true;
        }
        rocksdb::Status s = db_->SetOptions(db_->DefaultColumnFamily(), {
            {"level0_slowdown_writes_trigger",
             std::to_string(prof.l0_slowdown_trigger)},
            {"level0_stop_writes_trigger",
             std::to_string(prof.l0_stop_trigger)},
        });
        if (!s.ok()) {
            std::cerr << "[bandit] SetOptions failed: " << s.ToString()
                      << std::endl;
            return false;
        }
        current_l0_slowdown_.store(prof.l0_slowdown_trigger);
        current_l0_stop_.store(prof.l0_stop_trigger);
        std::cout << "[bandit] applied profile: l0_slowdown="
                  << prof.l0_slowdown_trigger
                  << ", l0_stop=" << prof.l0_stop_trigger << std::endl;
        return true;
    }

    std::vector<double> BuildContext(uint64_t l0,
                                     uint64_t pending_bytes,
                                     double stall_fraction,
                                     uint64_t is_write_stopped,
                                     uint64_t delayed_rate,
                                     uint64_t num_running_compactions,
                                     double recent_tp_mb_s,
                                     int current_bg) const {
        std::vector<double> x(FEATURE_DIM, 0.0);
        // Use the fixed compaction trigger (not the profile-controlled
        // slowdown trigger) as the L0 normalization base, so the bandit
        // cannot game its own context/reward by raising slowdown.
        const double l0_norm =
            static_cast<double>(l0) / static_cast<double>(l0_base_);
        x[0] = 1.0;
        x[1] = std::clamp(l0_norm, 0.0, 2.0);
        x[2] = std::clamp(static_cast<double>(pending_bytes) /
                              (64.0 * 1024.0 * 1024.0),
                          0.0, 2.0);
        x[3] = std::clamp(stall_fraction, 0.0, 1.0);
        x[4] = (is_write_stopped > 0) ? 1.0 : 0.0;
        x[5] = std::clamp(static_cast<double>(num_running_compactions) /
                              std::max(1, current_bg),
                          0.0, 2.0);
        x[6] = std::clamp(recent_tp_mb_s / throughput_norm_mb_s_, 0.0, 2.0);
        x[7] = (delayed_rate > 0) ? 1.0 : 0.0;
        return x;
    }

    // Fast-bandit reward:
    //   tp_norm - lambda_stall * stall_fraction - lambda_l0 * l0_risk
    // Encourages high throughput, penalizes stalls and L0 backlog risk.
    // The L0 risk is normalized against the FIXED compaction trigger
    // (l0_base_), not the profile-controlled slowdown trigger -- otherwise
    // the profile bandit could trivially shrink its own L0 penalty by
    // raising slowdown, even when actual backlog has not improved.
    double ComputeReward(double recent_tp_mb_s,
                         double stall_fraction,
                         uint64_t l0_max) const {
        double tp_norm = std::clamp(recent_tp_mb_s / throughput_norm_mb_s_, 0.0, 2.0);
        double l0_risk = std::clamp(
            static_cast<double>(l0_max) / static_cast<double>(l0_base_),
            0.0, 1.5);
        return tp_norm - lambda_stall_ * stall_fraction -
               lambda_l0_ * l0_risk;
    }

    // Profile-bandit reward: extends ComputeReward with two penalties that
    // directly target the knobs the profile actually controls
    // (level0_slowdown_writes_trigger / level0_stop_writes_trigger):
    //
    //   - lambda_stop  * stop_window_indicator      (1 if stop ever observed
    //                                                 in this window, else 0)
    //   - lambda_delay * delayed_active_fraction    (fraction of cycles in
    //                                                 the window with
    //                                                 actual_delayed_write_rate
    //                                                 > 0)
    //
    // Without these, raising slowdown/stop triggers can defer backlog signals
    // long enough to look "fine" on tp/stall/l0 alone while the system is
    // already in delayed/stopped territory.
    double ComputeProfileReward(double recent_tp_mb_s,
                                double stall_fraction,
                                uint64_t l0_max,
                                bool stop_seen_in_window,
                                double delayed_active_fraction) const {
        const double base =
            ComputeReward(recent_tp_mb_s, stall_fraction, l0_max);
        const double stop_term = stop_seen_in_window ? 1.0 : 0.0;
        const double delay_term =
            std::clamp(delayed_active_fraction, 0.0, 1.0);
        return base - lambda_stop_ * stop_term - lambda_delay_ * delay_term;
    }

    // Window-level context for the profile bandit. Mirrors BuildContext in
    // dimensionality, but every component is aggregated over the whole
    // profile window so that the context fed to Update/SelectArm matches the
    // time scale at which profile decisions actually take effect.
    //
    //   x[1] = max L0 in window / l0_base
    //   x[2] = avg pending_bytes  / 64MiB
    //   x[3] = window stall_us    / window elapsed_us
    //   x[4] = 1 iff write_stopped was observed at any cycle in the window
    //   x[5] = avg(num_running_compactions / max(1, current_bg_jobs))
    //   x[6] = window throughput  / tp_norm
    //   x[7] = fraction of cycles in the window with delayed_rate > 0
    std::vector<double> BuildProfileWindowContext(
        uint64_t l0_max,
        double pending_bytes_sum,
        uint64_t bytes_total,
        uint64_t stall_us_total,
        double elapsed_sec_total,
        int cycle_count,
        bool stop_seen,
        int delayed_cycles,
        double compaction_ratio_sum) const {
        std::vector<double> x(FEATURE_DIM, 0.0);
        const int n = std::max(1, cycle_count);
        const double pending_avg = pending_bytes_sum / static_cast<double>(n);
        const double stall_frac =
            elapsed_sec_total > 0
                ? std::clamp(static_cast<double>(stall_us_total) /
                                 (elapsed_sec_total * 1e6),
                             0.0, 1.0)
                : 0.0;
        const double tp_mb_s =
            elapsed_sec_total > 0
                ? (static_cast<double>(bytes_total) / (1024.0 * 1024.0)) /
                      elapsed_sec_total
                : 0.0;
        const double comp_ratio_avg =
            compaction_ratio_sum / static_cast<double>(n);
        const double delayed_frac =
            static_cast<double>(delayed_cycles) / static_cast<double>(n);
        const double l0_norm =
            static_cast<double>(l0_max) / static_cast<double>(l0_base_);
        x[0] = 1.0;
        x[1] = std::clamp(l0_norm, 0.0, 2.0);
        x[2] = std::clamp(pending_avg / (64.0 * 1024.0 * 1024.0), 0.0, 2.0);
        x[3] = stall_frac;
        x[4] = stop_seen ? 1.0 : 0.0;
        x[5] = std::clamp(comp_ratio_avg, 0.0, 2.0);
        x[6] = std::clamp(tp_mb_s / throughput_norm_mb_s_, 0.0, 2.0);
        x[7] = std::clamp(delayed_frac, 0.0, 1.0);
        return x;
    }

    void Run() {
        prev_stall_micros_ = GetStallMetric(statistics_);
        prev_bytes_written_ =
            statistics_ ? statistics_->getTickerCount(rocksdb::BYTES_WRITTEN)
                        : 0;
        last_decision_tp_ = std::chrono::steady_clock::now();

        // Fast bandit: per-cycle (context, arm) snapshot used to update on the
        // following cycle.
        bool have_pending_fast = false;
        std::vector<double> pending_fast_x;
        int pending_fast_arm = last_arm_id_.load();

        // Profile bandit time scale:
        //   * Each "profile window" = profile_interval_cycles_ fast cycles.
        //   * We aggregate raw metrics across the entire window.
        //   * At window close we summarize the window into profile_state_x.
        //
        // Standard LinUCB time-ordering (no post-action context leak):
        //   t_window_start :  observe profile_decision_x  ->  Select arm a_t
        //                     (commit a_t via ApplyProfile)
        //   ... window runs, accumulators fill ...
        //   t_window_end   :  observe window outcome -> profile_reward r_t
        //                     Update(a_t, profile_decision_x_at_t, r_t)
        //                     Build profile_state_x for the just-ended window
        //                     Select a_{t+1} := SelectArm(profile_state_x)
        //                     pending_decision_x for a_{t+1} := profile_state_x
        //
        // The constructor-set initial profile is NOT a bandit decision and
        // therefore receives NO Update at the first window close (we have
        // no decision context for it).
        bool have_pending_profile = false;
        int pending_profile_arm = last_profile_id_.load();
        std::vector<double> pending_profile_decision_x;  // x_t for arm above
        uint64_t profile_window_bytes = 0;
        uint64_t profile_window_stall_us = 0;
        double profile_window_elapsed_sec = 0.0;
        uint64_t profile_window_l0_max = 0;
        bool profile_window_stop_seen = false;
        int profile_window_delayed_cycles = 0;
        int profile_window_cycle_count = 0;
        double profile_window_pending_bytes_sum = 0.0;
        double profile_window_compaction_ratio_sum = 0.0;

        int cycle_idx = 0;
        uint64_t fast_window_l0_max = 0;

        while (!stop_flag_->load()) {
            uint64_t l0_now = GetL0Files(db_);
            fast_window_l0_max = std::max(fast_window_l0_max, l0_now);

            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(
                                     now - last_decision_tp_)
                                     .count();
            if (elapsed_sec < interval_sec_) {
                int slice_ms = std::max(100, interval_sec_ * 250);
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(slice_ms));
                continue;
            }

            uint64_t pending_bytes = GetCompactionPendingBytes(db_);
            uint64_t is_stopped = GetIsWriteStopped(db_);
            uint64_t delayed_rate = GetActualDelayedWriteRate(db_);
            uint64_t num_running = GetNumRunningCompactions(db_);
            uint64_t stall_total = GetStallMetric(statistics_);
            uint64_t stall_delta = stall_total >= prev_stall_micros_
                                       ? (stall_total - prev_stall_micros_)
                                       : 0;
            prev_stall_micros_ = stall_total;
            uint64_t bytes_written =
                statistics_
                    ? statistics_->getTickerCount(rocksdb::BYTES_WRITTEN)
                    : 0;
            uint64_t bytes_delta = bytes_written >= prev_bytes_written_
                                       ? (bytes_written - prev_bytes_written_)
                                       : 0;
            prev_bytes_written_ = bytes_written;

            const double interval_us = elapsed_sec * 1e6;
            const double stall_fraction =
                interval_us > 0
                    ? std::clamp(
                          static_cast<double>(stall_delta) / interval_us,
                          0.0, 1.0)
                    : 0.0;
            const double recent_tp_mb_s =
                elapsed_sec > 0
                    ? (static_cast<double>(bytes_delta) /
                       (1024.0 * 1024.0)) /
                          elapsed_sec
                    : 0.0;
            const int cur_bg = current_bg_jobs_.load();
            std::vector<double> x =
                BuildContext(fast_window_l0_max, pending_bytes, stall_fraction,
                             is_stopped, delayed_rate, num_running,
                             recent_tp_mb_s, cur_bg);

            // 1) reward the previous fast arm based on metrics since it was
            //    applied (per-cycle update is correct for the fast bandit).
            const double reward = ComputeReward(recent_tp_mb_s, stall_fraction,
                                                fast_window_l0_max);
            last_fast_reward_.store(reward);
            if (have_pending_fast) {
                fast_bandit_.Update(pending_fast_arm, pending_fast_x, reward);
            }

            // 2) accumulate this cycle's raw signals into the current profile
            //    window (the active profile is responsible for them).
            profile_window_bytes += bytes_delta;
            profile_window_stall_us += stall_delta;
            profile_window_elapsed_sec += elapsed_sec;
            profile_window_l0_max =
                std::max(profile_window_l0_max, fast_window_l0_max);
            ++profile_window_cycle_count;
            if (is_stopped > 0) {
                profile_window_stop_seen = true;
            }
            if (delayed_rate > 0) {
                ++profile_window_delayed_cycles;
            }
            profile_window_pending_bytes_sum +=
                static_cast<double>(pending_bytes);
            profile_window_compaction_ratio_sum +=
                static_cast<double>(num_running) /
                static_cast<double>(std::max(1, cur_bg));

            // 3) pick next fast arm; respect cooldown for switches.
            //    Only advance "committed" arm id when SetDBOptions actually
            //    succeeds, otherwise the controller's view would drift away
            //    from RocksDB's real state.
            const int prev_fast_arm = last_arm_id_.load();
            int new_fast_arm = fast_bandit_.SelectArm(x);
            const bool need_switch = (new_fast_arm != prev_fast_arm);
            const bool cooldown_blocked =
                need_switch &&
                std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_switch_tp_)
                        .count() < cooldown_sec_;
            int committed_fast_arm = prev_fast_arm;
            if (cooldown_blocked) {
                std::cout << "[bandit] cooldown active, skip fast switch"
                          << std::endl;
            } else if (need_switch) {
                if (ApplyFastArm(fast_arms_[new_fast_arm])) {
                    last_switch_tp_ = now;
                    committed_fast_arm = new_fast_arm;
                } else {
                    std::cerr << "[bandit] keeping fast arm " << prev_fast_arm
                              << " (apply of arm " << new_fast_arm
                              << " failed)" << std::endl;
                }
            }
            last_arm_id_.store(committed_fast_arm);
            // Remember the (context, arm) actually in effect so that the next
            // cycle updates the bandit for the truly-active arm.
            pending_fast_arm = committed_fast_arm;
            pending_fast_x = x;
            have_pending_fast = true;

            ++cycle_idx;

            // 4) End of a profile window. Time-clean LinUCB step:
            //      a) Build profile_state_x = window outcome summary.
            //      b) If we have a pending decision (a_t, x_t) from the
            //         previous window close, credit it with this window's
            //         aggregated reward using x_t (NOT profile_state_x --
            //         that would leak post-action observations into the
            //         decision context).
            //      c) Use profile_state_x to Select the next profile a_{t+1}
            //         and remember (a_{t+1}, profile_state_x) as the next
            //         pending decision pair.
            if (cycle_idx % profile_interval_cycles_ == 0) {
                std::vector<double> profile_state_x =
                    BuildProfileWindowContext(
                        profile_window_l0_max,
                        profile_window_pending_bytes_sum,
                        profile_window_bytes,
                        profile_window_stall_us,
                        profile_window_elapsed_sec,
                        profile_window_cycle_count,
                        profile_window_stop_seen,
                        profile_window_delayed_cycles,
                        profile_window_compaction_ratio_sum);

                if (have_pending_profile) {
                    const double win_tp_mb_s =
                        profile_window_elapsed_sec > 0
                            ? (static_cast<double>(profile_window_bytes) /
                               (1024.0 * 1024.0)) /
                                  profile_window_elapsed_sec
                            : 0.0;
                    const double win_stall_fraction =
                        profile_window_elapsed_sec > 0
                            ? std::clamp(
                                  static_cast<double>(profile_window_stall_us) /
                                      (profile_window_elapsed_sec * 1e6),
                                  0.0, 1.0)
                            : 0.0;
                    const double win_delayed_fraction =
                        profile_window_cycle_count > 0
                            ? static_cast<double>(
                                  profile_window_delayed_cycles) /
                                  static_cast<double>(
                                      profile_window_cycle_count)
                            : 0.0;
                    const double profile_reward = ComputeProfileReward(
                        win_tp_mb_s, win_stall_fraction,
                        profile_window_l0_max, profile_window_stop_seen,
                        win_delayed_fraction);
                    last_profile_reward_.store(profile_reward);
                    profile_bandit_.Update(pending_profile_arm,
                                           pending_profile_decision_x,
                                           profile_reward);
                    std::cout << "[bandit] profile window close: arm["
                              << pending_profile_arm
                              << "] win_tp_mb_s=" << win_tp_mb_s
                              << ", win_stall_frac=" << win_stall_fraction
                              << ", win_l0_max=" << profile_window_l0_max
                              << ", win_stop_seen="
                              << (profile_window_stop_seen ? 1 : 0)
                              << ", win_delayed_frac=" << win_delayed_fraction
                              << ", win_reward=" << profile_reward
                              << std::endl;
                }
                // Only advance "committed" profile id when SetOptions
                // actually succeeds. On failure, the OLD profile is still in
                // effect, so we keep crediting it in the next window.
                const int prev_profile = last_profile_id_.load();
                int new_profile = profile_bandit_.SelectArm(profile_state_x);
                int committed_profile = prev_profile;
                if (new_profile != prev_profile) {
                    if (ApplyProfile(profiles_[new_profile])) {
                        committed_profile = new_profile;
                    } else {
                        std::cerr << "[bandit] keeping profile "
                                  << prev_profile << " (apply of profile "
                                  << new_profile << " failed)" << std::endl;
                    }
                }
                last_profile_id_.store(committed_profile);
                // The arm we just committed will be credited at the NEXT
                // window close. Its decision context is the state we
                // observed *just before* selecting it (profile_state_x).
                pending_profile_arm = committed_profile;
                pending_profile_decision_x = profile_state_x;
                have_pending_profile = true;
                profile_window_bytes = 0;
                profile_window_stall_us = 0;
                profile_window_elapsed_sec = 0.0;
                profile_window_l0_max = 0;
                profile_window_stop_seen = false;
                profile_window_delayed_cycles = 0;
                profile_window_cycle_count = 0;
                profile_window_pending_bytes_sum = 0.0;
                profile_window_compaction_ratio_sum = 0.0;
            }

            std::cout << "[bandit] L0_max=" << fast_window_l0_max
                      << ", pending_bytes=" << pending_bytes
                      << ", stall_frac=" << stall_fraction
                      << ", tp_mb_s=" << recent_tp_mb_s
                      << ", fast_reward=" << reward
                      << ", last_profile_reward="
                      << last_profile_reward_.load()
                      << ", arm[" << last_arm_id_.load()
                      << "]=(bg=" << current_bg_jobs_.load()
                      << ",sub=" << current_subcompactions_.load() << ")"
                      << ", profile[" << last_profile_id_.load()
                      << "]=(slow=" << current_l0_slowdown_.load()
                      << ",stop=" << current_l0_stop_.load() << ")"
                      << std::endl;

            fast_window_l0_max = 0;
            last_decision_tp_ = now;
        }

        std::cout << "[bandit] stopping" << std::endl;
    }

    rocksdb::DB* db_;
    std::shared_ptr<rocksdb::Statistics> statistics_;
    std::atomic<bool>* stop_flag_;
    std::vector<BanditFastArm> fast_arms_;
    std::vector<BanditProfile> profiles_;
    int interval_sec_;
    int cooldown_sec_;
    int profile_interval_cycles_;
    double lambda_stall_;
    double lambda_l0_;
    double lambda_stop_;
    double lambda_delay_;
    double throughput_norm_mb_s_;
    int l0_base_;  // fixed L0 normalization base (compaction trigger)

    LinUCB fast_bandit_;
    LinUCB profile_bandit_;

    std::atomic<int> current_bg_jobs_;
    std::atomic<int> current_subcompactions_;
    std::atomic<int> current_l0_slowdown_;
    std::atomic<int> current_l0_stop_;
    std::atomic<int> last_arm_id_;
    std::atomic<int> last_profile_id_;
    std::atomic<double> last_fast_reward_;
    std::atomic<double> last_profile_reward_;

    std::chrono::steady_clock::time_point last_switch_tp_;
    std::chrono::steady_clock::time_point last_decision_tp_;
    uint64_t prev_stall_micros_ = 0;
    uint64_t prev_bytes_written_ = 0;
    std::thread worker_;
};
