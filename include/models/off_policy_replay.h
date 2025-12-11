#ifndef RL_OFFPOLICY_REPLAY_H
#define RL_OFFPOLICY_REPLAY_H

#include <models/torch/linear.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "agent.h"

namespace RLlib::Models {

template <typename Net>
class OffPolicyReplayLearner {
 public:
  using Network = Net;
  using State = typename Net::State;
  using ResultsList = typename Net::ResultsList;

  static constexpr int kFeaturesDim = Net::kFeaturesDim;
  static constexpr int kActionsDim = Net::kActionsDim;

  struct Transition {
    State state;
    int action;
    double td_target;
  };

  explicit OffPolicyReplayLearner(const json &config)
      : net_(config),
        alpha_(config.value("learning_rate", 1e-3)),
        replay_capacity_(
            config.value("replay_capacity", static_cast<std::size_t>(100000))),
        batch_size_(config.value("batch_size", static_cast<std::size_t>(32))),
        rng_(std::random_device{}()),
        save_grad_{static_cast<int>(config.value("save_grad", false))} {
    if (batch_size_ == 0) {
      throw std::runtime_error("batch_size must be > 0");
    }
    if (batch_size_ > replay_capacity_) {
      throw std::runtime_error("batch_size must be <= replay_capacity");
    }
    replay_buffer_.reserve(replay_capacity_);

    batch_states_.reserve(batch_size_);
    batch_actions_.reserve(batch_size_);
    batch_targets_.reserve(batch_size_);
    reshuffle_indices_.reserve(replay_capacity_);

    auto optimizer_config =
        config.value("optimizer", json::object({{"type", "adam"}}));
    std::string optimizer_type = optimizer_config["type"];
    if (optimizer_type == "adam") {
      optimizer_ = std::make_unique<torch::optim::Adam>(
          net_.parameters(), torch::optim::AdamOptions(alpha_));
    } else if (optimizer_type == "sgd") {
      optimizer_ = std::make_unique<torch::optim::SGD>(
          net_.parameters(), torch::optim::SGDOptions(alpha_));
    } else {
      throw std::runtime_error("Unknown optimizer type: " + optimizer_type +
                               " (supported: adam, sgd)");
    }
  }

  const ResultsList &GetActionValues(const State &state) {
    return net_.GetActionValues(state);
  }

  void Update(const State &state, int action_idx, double td_target) {
    PushTransition(state, action_idx, td_target);
    if (replay_buffer_.size() >= batch_size_) {
      TrainFromReplay();
    }
  }

  void SetLearningRate(double alpha) {
    alpha_ = alpha;
    for (auto &group : optimizer_->param_groups()) {
      auto &opts = group.options();
      if (auto *adam_opts = dynamic_cast<torch::optim::AdamOptions *>(&opts)) {
        adam_opts->lr(alpha_);
      } else if (auto *sgd_opts =
                     dynamic_cast<torch::optim::SGDOptions *>(&opts)) {
        sgd_opts->lr(alpha_);
      }
    }
  }

  void OutputModel(std::string_view fname, char delimiter = '\n',
                   bool append = false) const {
    net_.OutputModel(fname, delimiter, append);
  }

  void LoadModel(std::string_view fname, char delimiter = '\n') {
    net_.LoadModel(fname, delimiter);
  }

  Network &GetNet() { return net_; }
  const Network &GetNet() const { return net_; }

 private:
  void PushTransition(const State &state, int action, double td_target) {
    if (replay_buffer_.size() < replay_capacity_) {
      replay_buffer_.push_back(Transition{state, action, td_target});
    } else {
      replay_buffer_[replay_pos_] = Transition{state, action, td_target};
    }
    replay_pos_ = (replay_pos_ + 1) % replay_capacity_;
  }

  void TrainFromReplay() {
    const std::size_t buffer_size = replay_buffer_.size();
    if (buffer_size < batch_size_) return;

    reshuffle_indices_.resize(buffer_size);
    std::iota(reshuffle_indices_.begin(), reshuffle_indices_.end(), 0);
    std::shuffle(reshuffle_indices_.begin(), reshuffle_indices_.end(), rng_);

    batch_states_.clear();
    batch_actions_.clear();
    batch_targets_.clear();

    for (std::size_t i = 0; i < batch_size_; ++i) {
      const auto &tr = replay_buffer_[reshuffle_indices_[i]];
      batch_states_.push_back(tr.state);
      batch_actions_.push_back(tr.action);
      batch_targets_.push_back(tr.td_target);
    }

    UpdateMinibatch(batch_states_, batch_actions_, batch_targets_);
  }

  void UpdateMinibatch(const std::vector<State> &states,
                       const std::vector<int> &actions,
                       const std::vector<double> &td_targets) {
    const std::size_t B = states.size();
    if (B == 0) return;
    if (actions.size() != B || td_targets.size() != B) {
      throw std::runtime_error("Minibatch vectors must have the same size");
    }

    const auto optsD =
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    const auto optsL =
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);

    torch::Tensor X = torch::empty({static_cast<long>(B), kFeaturesDim}, optsD);
    {
      auto X_acc = X.accessor<double, 2>();
      for (std::size_t b = 0; b < B; ++b) {
        for (int j = 0; j < kFeaturesDim; ++j) {
          X_acc[b][j] = static_cast<double>(states[b][j]);
        }
      }
    }

    torch::Tensor A = torch::empty({static_cast<int64_t>(B)}, optsL);
    {
      auto A_acc = A.accessor<int64_t, 1>();
      for (std::size_t b = 0; b < B; ++b) {
        A_acc[b] = static_cast<int64_t>(actions[b]);
      }
    }

    torch::Tensor Y = torch::empty({static_cast<long>(B)}, optsD);
    {
      auto Y_acc = Y.accessor<double, 1>();
      for (std::size_t b = 0; b < B; ++b) {
        Y_acc[b] = td_targets[b];
      }
    }

    const auto Q = net_.forward(X);
    const auto A2 = A.unsqueeze(1);
    const auto Q_a = Q.gather(1, A2).squeeze(1);

    const auto diff = Q_a - Y;
    const auto loss = 0.5 * torch::mean(diff * diff);

    optimizer_->zero_grad();
    loss.backward();
    if (save_grad_) {
      std::ofstream grad_file{"grad.txt",
                              save_grad_ == 1 ? std::ios::out : std::ios::app};
      if (save_grad_ == 1) {
        save_grad_ = 2;
      }

      if (!grad_file.is_open())
        throw std::runtime_error("Failed to open grad.txt");

      grad_file << "state = ";
      for (const auto &s : states) {
        for (const auto &is : s) grad_file << is << ",";
        grad_file << "|";
      }
      grad_file << "; action_idx = ";
      for (const auto &a : actions) {
        grad_file << a << ",";
      }
      auto Q_a_cpu = Q_a.detach().to(torch::kCPU);
      auto Q_a_acc = Q_a_cpu.template accessor<double, 1>();

      grad_file << "; last_q = ";
      for (int64_t i = 0; i < Q_a_cpu.size(0); ++i) {
        grad_file << Q_a_acc[i] << ",";
      }
      grad_file << "; new_q = ";
      for (const auto &a : td_targets) {
        grad_file << a << ",";
      }
      grad_file << "\n";

#ifdef DEBUG
      std::cout << "grad:" << std::endl;
#endif
      for (const auto &param : net_.parameters()) {
        if (!param.grad().defined()) continue;

        auto g = param.grad().detach().to(torch::kCPU);
        grad_file << g << "\n";

#ifdef DEBUG
        std::cout << g << std::endl;
#endif
      }
    }
    optimizer_->step();
  }

  Network net_;
  double alpha_;
  std::unique_ptr<torch::optim::Optimizer> optimizer_;

  std::vector<Transition> replay_buffer_;
  std::size_t replay_capacity_;
  std::size_t batch_size_;
  std::size_t replay_pos_{0};
  std::minstd_rand rng_;
  std::vector<State> batch_states_;
  std::vector<int> batch_actions_;
  std::vector<double> batch_targets_;
  std::vector<size_t> reshuffle_indices_;
  int save_grad_{};
};

template <int tFeaturesDim, int tActionsDim, typename TFeature>
using OffPolicyReplayLinearModel = RLlib::Models::OffPolicyReplayLearner<
    RLlib::Models::LinearQNetwork<tFeaturesDim, tActionsDim, TFeature>>;

}  // namespace RLlib::Models

#endif