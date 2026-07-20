#ifndef MODELS_PPO_LEARNER_H
#define MODELS_PPO_LEARNER_H

#include <models/torch/actor_critic.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "agent.h"

namespace RLlib::Models {

// Optimizer-owning wrapper around a discrete actor-critic network, analogous to
// OffPolicyReplayLearner. It provides:
//   * EvaluateAction(state)   -> samples an action from the current policy and
//                                returns {action, log_prob, value}.
//   * LearnFromBatch(...)     -> K epochs of minibatch clipped-surrogate PPO
//                                updates over an on-policy batch.
// The agent is responsible for advantage/return (GAE) computation; this class
// owns the network, the optimizer, and the PPO objective.
template <typename Net> class PPOLearner {
public:
  using Network = Net;
  using State = typename Net::State;
  using ActionParam = int;

  struct Decision {
    ActionParam action_param;
    double log_prob;
    double value;
  };

  explicit PPOLearner(const json &config)
      : net_(config), alpha_(config.value("learning_rate", 3e-4)),
        clip_epsilon_(config.value("clip_epsilon", 0.2)),
        value_coef_(config.value("value_coef", 0.5)),
        entropy_coef_(config.value("entropy_coef", 0.0)),
        epochs_(config.value("epochs", static_cast<std::size_t>(10))),
        minibatch_size_(
            config.value("minibatch_size", static_cast<std::size_t>(64))),
        rng_(std::random_device{}()) {
    if (minibatch_size_ == 0) {
      throw std::runtime_error("minibatch_size must be > 0");
    }

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

  // Sample an action from the current policy for a single state.
  Decision EvaluateAction(const State &state) {
    torch::NoGradGuard no_grad;
    auto X = StateTensor(state);
    auto [logits, value] = net_.forward(X);
    auto log_probs = torch::log_softmax(logits, /*dim=*/1);
    auto probs = torch::exp(log_probs);
    auto sampled = torch::multinomial(probs, /*num_samples=*/1);

    int64_t a = sampled.template item<int64_t>();
    double log_prob = log_probs.index({0, a}).template item<double>();
    double v = value.template item<double>();
    return Decision{static_cast<int>(a), log_prob, v};
  }

  // Evaluate only the critic. Keeping this separate from EvaluateAction lets
  // the agent bootstrap a completed rollout before sampling from the updated
  // policy.
  double EvaluateValue(const State &state) {
    torch::NoGradGuard no_grad;
    auto X = StateTensor(state);
    auto [logits, value] = net_.forward(X);
    (void)logits;
    return value.template item<double>();
  }

  void LearnFromBatch(const std::vector<State> &states,
                      const std::vector<ActionParam> &action_params,
                      const std::vector<double> &old_log_probs,
                      const std::vector<double> &advantages,
                      const std::vector<double> &returns) {
    const int64_t B = static_cast<int64_t>(states.size());
    if (B == 0)
      return;
    if (static_cast<int64_t>(action_params.size()) != B ||
        static_cast<int64_t>(old_log_probs.size()) != B ||
        static_cast<int64_t>(advantages.size()) != B ||
        static_cast<int64_t>(returns.size()) != B) {
      throw std::runtime_error("PPO batch vectors must have the same size");
    }

    const int F = net_.FeaturesDim();
    const auto dtype = torch::CppTypeToScalarType<typename Net::Feature>::value;
    const auto optsF = torch::TensorOptions().dtype(dtype);
    const auto optsL = torch::TensorOptions().dtype(torch::kLong);

    torch::Tensor X = torch::empty({B, F}, optsF);
    torch::Tensor A = torch::empty({B}, optsL);
    torch::Tensor OldLogP = torch::empty({B}, optsF);
    torch::Tensor Adv = torch::empty({B}, optsF);
    torch::Tensor Ret = torch::empty({B}, optsF);
    {
      auto X_acc = X.accessor<typename Net::Feature, 2>();
      auto A_acc = A.accessor<int64_t, 1>();
      auto Old_acc = OldLogP.accessor<typename Net::Feature, 1>();
      auto Adv_acc = Adv.accessor<typename Net::Feature, 1>();
      auto Ret_acc = Ret.accessor<typename Net::Feature, 1>();
      for (int64_t b = 0; b < B; ++b) {
        if (states[b].size() != static_cast<std::size_t>(F)) {
          throw std::invalid_argument(
              "Rollout state size does not match network feature dimension");
        }
        if (action_params[b] < 0 || action_params[b] >= net_.ActionsDim()) {
          throw std::out_of_range(
              "Rollout action index does not match network action dimension");
        }
        for (int j = 0; j < F; ++j) {
          X_acc[b][j] = static_cast<typename Net::Feature>(states[b][j]);
        }
        A_acc[b] = static_cast<int64_t>(action_params[b]);
        Old_acc[b] = static_cast<typename Net::Feature>(old_log_probs[b]);
        Adv_acc[b] = static_cast<typename Net::Feature>(advantages[b]);
        Ret_acc[b] = static_cast<typename Net::Feature>(returns[b]);
      }
    }

    std::vector<int64_t> order(B);
    std::iota(order.begin(), order.end(), 0);

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
      std::shuffle(order.begin(), order.end(), rng_);
      for (int64_t start = 0; start < B;
           start += static_cast<int64_t>(minibatch_size_)) {
        const int64_t end = std::min<int64_t>(start + minibatch_size_, B);
        auto idx = torch::from_blob(order.data() + start, {end - start}, optsL)
                       .clone();

        auto Xb = X.index_select(0, idx);
        auto Ab = A.index_select(0, idx);
        auto OldLogPb = OldLogP.index_select(0, idx);
        auto Advb = Adv.index_select(0, idx);
        auto Retb = Ret.index_select(0, idx);

        auto [logits, value] = net_.forward(Xb);
        auto log_probs = torch::log_softmax(logits, /*dim=*/1);
        auto new_log_p = log_probs.gather(1, Ab.unsqueeze(1)).squeeze(1);

        auto ratio = torch::exp(new_log_p - OldLogPb);
        auto surr1 = ratio * Advb;
        auto surr2 =
            torch::clamp(ratio, 1.0 - clip_epsilon_, 1.0 + clip_epsilon_) *
            Advb;
        // Elementwise min, written with `where` to stay version-portable.
        auto surr = torch::where(surr1 < surr2, surr1, surr2);
        auto policy_loss = -surr.mean();

        auto value_loss = torch::mse_loss(value, Retb);

        auto probs = torch::exp(log_probs);
        auto entropy = -(probs * log_probs).sum(1).mean();

        auto loss =
            policy_loss + value_coef_ * value_loss - entropy_coef_ * entropy;

        optimizer_->zero_grad();
        loss.backward();
        optimizer_->step();
      }
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

  // Copy learned network state without replacing this model's parameter
  // tensors. In-place copies preserve optimizer references and deliberately do
  // not synchronize optimizer state or sampling RNG state.
  void ImportWeights(const PPOLearner &source) {
    torch::NoGradGuard no_grad;
    CopyTensors(net_.parameters(),
                const_cast<Network &>(source.net_).parameters(), "parameters");
    CopyTensors(net_.buffers(), const_cast<Network &>(source.net_).buffers(),
                "buffers");
  }

  Network &GetNet() { return net_; }
  const Network &GetNet() const { return net_; }

private:
  static void CopyTensors(std::vector<torch::Tensor> destination,
                          const std::vector<torch::Tensor> &source,
                          std::string_view description) {
    if (destination.size() != source.size()) {
      throw std::invalid_argument("Cannot import PPO model " +
                                  std::string(description) +
                                  ": tensor counts differ");
    }
    for (std::size_t index = 0; index < destination.size(); ++index) {
      if (destination[index].sizes().vec() != source[index].sizes().vec()) {
        throw std::invalid_argument("Cannot import PPO model " +
                                    std::string(description) +
                                    ": tensor shapes differ");
      }
      destination[index].copy_(source[index]);
    }
  }

  torch::Tensor StateTensor(const State &state) const {
    if (state.size() != static_cast<std::size_t>(net_.FeaturesDim())) {
      throw std::invalid_argument(
          "State size does not match network feature dimension");
    }
    const auto dtype = torch::CppTypeToScalarType<typename Net::Feature>::value;
    return torch::from_blob(const_cast<typename Net::Feature *>(state.data()),
                            {1, static_cast<int64_t>(state.size())},
                            torch::TensorOptions().dtype(dtype))
        .clone();
  }

  Network net_;
  double alpha_;
  double clip_epsilon_;
  double value_coef_;
  double entropy_coef_;
  std::size_t epochs_;
  std::size_t minibatch_size_;
  std::unique_ptr<torch::optim::Optimizer> optimizer_;
  std::mt19937 rng_;
};

} // namespace RLlib::Models

#endif
