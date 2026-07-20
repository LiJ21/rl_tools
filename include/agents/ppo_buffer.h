#ifndef AGENTS_PPO_BUFFER_H
#define AGENTS_PPO_BUFFER_H

#include <cmath>
#include <cstddef>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace RLlib {

// Raw on-policy data collected by PPOAgent. Keeping rewards, critic values,
// and path-end bootstrap markers in the transported representation allows
// advantages to be normalized over the combined distributed batch without
// letting GAE cross episode boundaries.
template <typename TState, typename TActionParam> struct PPORolloutBuffer {
  using State = TState;
  using ActionParam = TActionParam;

  std::vector<State> states;
  std::vector<ActionParam> action_params;
  std::vector<double> old_log_probs;
  std::vector<double> values;
  std::vector<double> rewards;
  std::vector<std::optional<double>> path_end_bootstraps;

  std::size_t Size() const { return states.size(); }
  bool Empty() const { return states.empty(); }

  void Reserve(std::size_t size) {
    states.reserve(size);
    action_params.reserve(size);
    old_log_probs.reserve(size);
    values.reserve(size);
    rewards.reserve(size);
    path_end_bootstraps.reserve(size);
  }

  void Clear() {
    states.clear();
    action_params.clear();
    old_log_probs.clear();
    values.clear();
    rewards.clear();
    path_end_bootstraps.clear();
  }

  void ValidateSizes() const {
    const auto size = Size();
    if (action_params.size() != size || old_log_probs.size() != size ||
        values.size() != size || rewards.size() != size ||
        path_end_bootstraps.size() != size) {
      throw std::invalid_argument(
          "PPO rollout buffer vectors must have the same size");
    }
  }

  // A transported buffer must contain only complete paths. An internal
  // auto-learning buffer may end in an open path and use a trailing bootstrap,
  // so completeness is intentionally separate from ValidateSizes().
  void ValidateComplete() const {
    ValidateSizes();
    if (!Empty() && !path_end_bootstraps.back().has_value()) {
      throw std::logic_error("PPO rollout buffer ends in an unclosed path");
    }
  }

  void Append(const PPORolloutBuffer &source) {
    ValidateComplete();
    source.ValidateComplete();
    Reserve(Size() + source.Size());

    states.insert(states.end(), source.states.begin(), source.states.end());
    action_params.insert(action_params.end(), source.action_params.begin(),
                         source.action_params.end());
    old_log_probs.insert(old_log_probs.end(), source.old_log_probs.begin(),
                         source.old_log_probs.end());
    values.insert(values.end(), source.values.begin(), source.values.end());
    rewards.insert(rewards.end(), source.rewards.begin(), source.rewards.end());
    path_end_bootstraps.insert(path_end_bootstraps.end(),
                               source.path_end_bootstraps.begin(),
                               source.path_end_bootstraps.end());
  }

  void Append(PPORolloutBuffer &&source) {
    ValidateComplete();
    source.ValidateComplete();
    Reserve(Size() + source.Size());

    states.insert(states.end(), std::make_move_iterator(source.states.begin()),
                  std::make_move_iterator(source.states.end()));
    action_params.insert(action_params.end(),
                         std::make_move_iterator(source.action_params.begin()),
                         std::make_move_iterator(source.action_params.end()));
    old_log_probs.insert(old_log_probs.end(),
                         std::make_move_iterator(source.old_log_probs.begin()),
                         std::make_move_iterator(source.old_log_probs.end()));
    values.insert(values.end(), std::make_move_iterator(source.values.begin()),
                  std::make_move_iterator(source.values.end()));
    rewards.insert(rewards.end(),
                   std::make_move_iterator(source.rewards.begin()),
                   std::make_move_iterator(source.rewards.end()));
    path_end_bootstraps.insert(
        path_end_bootstraps.end(),
        std::make_move_iterator(source.path_end_bootstraps.begin()),
        std::make_move_iterator(source.path_end_bootstraps.end()));
    source.Clear();
  }
};

struct PPOTargets {
  std::vector<double> advantages;
  std::vector<double> returns;
};

template <typename TState, typename TActionParam>
PPOTargets
ComputePPOTargets(const PPORolloutBuffer<TState, TActionParam> &buffer,
                  double gamma, double lambda, bool normalize_advantage,
                  double trailing_bootstrap = 0.0) {
  buffer.ValidateSizes();
  const auto size = buffer.Size();
  PPOTargets targets{std::vector<double>(size), std::vector<double>(size)};
  if (size == 0) {
    return targets;
  }

  double gae = 0.0;
  double next_value = trailing_bootstrap;
  for (std::size_t end = size; end > 0; --end) {
    const auto index = end - 1;
    if (buffer.path_end_bootstraps[index].has_value()) {
      gae = 0.0;
      next_value = *buffer.path_end_bootstraps[index];
    }
    const double delta =
        buffer.rewards[index] + gamma * next_value - buffer.values[index];
    gae = delta + gamma * lambda * gae;
    targets.advantages[index] = gae;
    targets.returns[index] = gae + buffer.values[index];
    next_value = buffer.values[index];
  }

  if (normalize_advantage && size > 1) {
    double mean = 0.0;
    for (double advantage : targets.advantages) {
      mean += advantage;
    }
    mean /= static_cast<double>(size);

    double variance = 0.0;
    for (double advantage : targets.advantages) {
      variance += (advantage - mean) * (advantage - mean);
    }
    variance /= static_cast<double>(size);

    const double standard_deviation = std::sqrt(variance) + 1e-8;
    for (double &advantage : targets.advantages) {
      advantage = (advantage - mean) / standard_deviation;
    }
  }

  return targets;
}

} // namespace RLlib

#endif
