#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

// Header-only LinUCB (disjoint-arm linear UCB).
//
//   For each arm a we maintain:
//     A_a    = lambda * I + sum_t x_t x_t^T            (d x d, SPD)
//     b_a    = sum_t r_t x_t                            (d)
//     theta_a = A_a^{-1} b_a
//
//   At decision time, given context x:
//     ucb_a = theta_a^T x + alpha * sqrt( x^T A_a^{-1} x )
//   pick a* = argmax_a ucb_a.
//
//   Update on observed reward r for chosen arm a:
//     A_a += x x^T   (Sherman-Morrison rank-1 update of A_inv)
//     b_a += r * x
//
//   We store A_inv directly to avoid solving a linear system at every step.
class LinUCB {
public:
    LinUCB(int num_arms,
           int feature_dim,
           double alpha = 1.0,
           double ridge_lambda = 1.0)
        : num_arms_(num_arms),
          d_(feature_dim),
          alpha_(alpha),
          A_inv_(num_arms, std::vector<double>(feature_dim * feature_dim, 0.0)),
          b_(num_arms, std::vector<double>(feature_dim, 0.0)) {
        const double inv_lambda = 1.0 / ridge_lambda;
        for (int a = 0; a < num_arms_; ++a) {
            for (int i = 0; i < d_; ++i) {
                A_inv_[a][i * d_ + i] = inv_lambda;
            }
        }
    }

    // Selects best arm under UCB. Optionally returns per-arm scores.
    int SelectArm(const std::vector<double>& x,
                  std::vector<double>* ucb_scores = nullptr) const {
        int best_arm = 0;
        double best_score = -1e30;
        if (ucb_scores) {
            ucb_scores->assign(num_arms_, 0.0);
        }
        std::vector<double> Ax(d_, 0.0);
        for (int a = 0; a < num_arms_; ++a) {
            MatVec(A_inv_[a], x, &Ax);
            double mean = Dot(b_[a], Ax);
            double var = Dot(x, Ax);
            if (var < 0.0) var = 0.0;
            double score = mean + alpha_ * std::sqrt(var);
            if (ucb_scores) {
                (*ucb_scores)[a] = score;
            }
            if (score > best_score) {
                best_score = score;
                best_arm = a;
            }
        }
        return best_arm;
    }

    // Sherman-Morrison rank-1 update of A_inv when we add x x^T to A.
    void Update(int arm, const std::vector<double>& x, double reward) {
        if (arm < 0 || arm >= num_arms_) return;
        std::vector<double> Ax(d_, 0.0);
        MatVec(A_inv_[arm], x, &Ax);
        double denom = 1.0 + Dot(x, Ax);
        if (denom <= 1e-12) {
            denom = 1e-12;
        }
        const double inv_denom = 1.0 / denom;
        for (int i = 0; i < d_; ++i) {
            const double Axi = Ax[i];
            for (int j = 0; j < d_; ++j) {
                A_inv_[arm][i * d_ + j] -= Axi * Ax[j] * inv_denom;
            }
        }
        for (int i = 0; i < d_; ++i) {
            b_[arm][i] += reward * x[i];
        }
    }

    int num_arms() const { return num_arms_; }
    int feature_dim() const { return d_; }
    double alpha() const { return alpha_; }
    void set_alpha(double a) { alpha_ = a; }

private:
    static double Dot(const std::vector<double>& a,
                      const std::vector<double>& b) {
        double s = 0.0;
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    void MatVec(const std::vector<double>& M,
                const std::vector<double>& v,
                std::vector<double>* out) const {
        for (int i = 0; i < d_; ++i) {
            double s = 0.0;
            const int row = i * d_;
            for (int j = 0; j < d_; ++j) {
                s += M[row + j] * v[j];
            }
            (*out)[i] = s;
        }
    }

    int num_arms_;
    int d_;
    double alpha_;
    std::vector<std::vector<double>> A_inv_;  // each is d*d row-major
    std::vector<std::vector<double>> b_;      // each is d
};
