#ifndef DISTRIBUTED_LEARNER_H
#define DISTRIBUTED_LEARNER_H

#include <cstddef>
#include <mutex>
#include <utility>

#include "agents/ppo.h"

namespace RLlib {

// Transport-agnostic PPO aggregation. Collection and synchronization barriers
// belong to the application; this helper only accepts complete raw rollout
// buffers, computes targets over their union, and updates a caller-owned model.
template <CPolicyModel TModel> class DistributedLearner {
public:
  using Model = TModel;
  using State = typename Model::State;
  using ActionParam = typename Model::ActionParam;
  using Buffer = PPORolloutBuffer<State, ActionParam>;

  DistributedLearner(Model &model, double gamma, double lambda,
                     bool normalize_advantage)
      : model_(model), gamma_(gamma), lambda_(lambda),
        normalize_advantage_(normalize_advantage) {}

  DistributedLearner(Model &model, const json &config)
      : DistributedLearner(model, config.value("gamma", 0.99),
                           config.value("gae_lambda", 0.95),
                           config.value("normalize_advantage", true)) {}

  // Taking the buffer by value supports both copying an lvalue (useful for
  // serialization paths) and cheaply moving the result of ReleaseBuffer().
  void Submit(Buffer buffer) {
    buffer.ValidateComplete();
    std::scoped_lock lock(mutex_);
    buffer_.Append(std::move(buffer));
  }

  std::size_t SubmittedSteps() const {
    std::scoped_lock lock(mutex_);
    return buffer_.Size();
  }

  bool Empty() const {
    std::scoped_lock lock(mutex_);
    return buffer_.Empty();
  }

  // The application must stop collection before Learn and must not submit the
  // next policy generation until updated weights have reached every actor.
  void Learn() {
    std::scoped_lock lock(mutex_);
    if (buffer_.Empty()) {
      return;
    }

    const auto targets =
        ComputePPOTargets(buffer_, gamma_, lambda_, normalize_advantage_);
    model_.LearnFromBatch(buffer_.states, buffer_.action_params,
                          buffer_.old_log_probs, targets.advantages,
                          targets.returns);
    buffer_.Clear();
  }

  void Clear() {
    std::scoped_lock lock(mutex_);
    buffer_.Clear();
  }

  Model &GetModel() { return model_; }
  const Model &GetModel() const { return model_; }

private:
  Model &model_;
  double gamma_;
  double lambda_;
  bool normalize_advantage_;

  mutable std::mutex mutex_;
  Buffer buffer_;
};

} // namespace RLlib

#endif
