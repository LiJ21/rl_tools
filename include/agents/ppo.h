#ifndef AGENTS_PPO_H
#define AGENTS_PPO_H
#include <agent.h>

#include <concepts>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "agents/ppo_buffer.h"
#include "agents/sarsa.h" // for RLlib::load_json

namespace RLlib {

template <typename TActionParam, typename TAction = TActionParam>
class IdentityActionMapper {
public:
  static_assert(std::convertible_to<const TActionParam &, TAction>,
                "IdentityActionMapper requires ActionParam to be convertible "
                "to the environment Action");

  TAction operator()(const TActionParam &action_param) const {
    return action_param;
  }
};

template <std::integral TActionParam, typename TAction>
class IndexedActionMapper {
public:
  using ActionsList = std::vector<TAction>;

  explicit IndexedActionMapper(const ActionsList &actions) : actions_(actions) {
    ValidateActions();
  }

  explicit IndexedActionMapper(ActionsList &&actions)
      : actions_(std::move(actions)) {
    ValidateActions();
  }

  const TAction &operator()(const TActionParam &action_param) const {
    if constexpr (std::is_signed_v<TActionParam>) {
      if (action_param < 0) {
        throw std::out_of_range("ActionParam cannot be negative");
      }
    }
    const auto index = static_cast<std::size_t>(action_param);
    if (index >= actions_.size()) {
      throw std::out_of_range(
          "ActionParam indexes outside the environment action list");
    }
    return actions_[index];
  }

private:
  void ValidateActions() const {
    if (actions_.empty()) {
      throw std::invalid_argument(
          "IndexedActionMapper requires at least one environment action");
    }
  }

  ActionsList actions_;
};

// Policy-model contract, the policy-gradient counterpart to CModel. Instead of
// exposing action-values and a scalar TD update, a policy model samples an
// action (returning its log-prob and the state value) and learns from an
// on-policy batch.
template <typename TModel>
concept CPolicyModel =
    requires(TModel model, typename TModel::State state,
             const std::vector<typename TModel::State> &states,
             const std::vector<typename TModel::ActionParam> &action_params,
             const std::vector<double> &dvec) {
      {
        model.EvaluateAction(state)
      } -> std::same_as<typename TModel::Decision>;
      { model.EvaluateValue(state) } -> std::convertible_to<double>;
      {
        model.LearnFromBatch(states, action_params, dvec, dvec, dvec)
      } -> std::same_as<void>;
      { model.SetLearningRate(0.1) } -> std::same_as<void>;
      { model.OutputModel(std::string_view{}) } -> std::same_as<void>;
      { model.LoadModel(std::string_view{}) } -> std::same_as<void>;
    } &&
    requires(typename TModel::Decision decision) {
      {
        decision.action_param
      } -> std::convertible_to<typename TModel::ActionParam>;
      { decision.log_prob } -> std::convertible_to<double>;
      { decision.value } -> std::convertible_to<double>;
    };

// Importing weights is deliberately a separate contract: ordinary policy
// models need not support distributed actor synchronization.
template <typename TModel>
concept CWeightImportableModel =
    requires(TModel destination, const TModel source) {
      { destination.ImportWeights(source) } -> std::same_as<void>;
    };

// Proximal Policy Optimization agent. The model defines the probability-bearing
// ActionParam stored in the batch, while ActionMapper converts it into the
// environment-facing Action returned by UpdateState.
template <typename TModel, typename TAction, typename TReward,
          typename TActionMapper =
              IdentityActionMapper<typename TModel::ActionParam, TAction>,
          bool tAutoLearn = true>
class PPOAgent
    : public AgentBase<
          PPOAgent<TModel, TAction, TReward, TActionMapper, tAutoLearn>,
          TAction, TReward, typename TModel::State> {
public:
  using Model = TModel;
  using Self = PPOAgent<TModel, TAction, TReward, TActionMapper, tAutoLearn>;
  using Base = AgentBase<Self, TAction, TReward, typename TModel::State>;
  using State = typename Model::State;
  using Action = TAction;
  using ActionParam = typename Model::ActionParam;
  using ActionMapper = TActionMapper;
  using Reward = TReward;
  using Buffer = PPORolloutBuffer<State, ActionParam>;

  static constexpr bool AutoLearn = tAutoLearn;

  static_assert(std::invocable<ActionMapper &, const ActionParam &>,
                "PPO ActionMapper must accept const ActionParam&");
  static_assert(
      std::convertible_to<
          std::invoke_result_t<ActionMapper &, const ActionParam &>, Action>,
      "PPO ActionMapper result must be convertible to the environment Action");

  PPOAgent(const char *config_file)
    requires std::default_initializable<ActionMapper> &&
             std::constructible_from<Model, const json &>
      : PPOAgent(ActionMapper{}, load_json(config_file)) {}

  PPOAgent(const json &config)
    requires std::default_initializable<ActionMapper> &&
             std::constructible_from<Model, const json &>
      : PPOAgent(ActionMapper{}, config) {}

  PPOAgent(ActionMapper action_mapper, const char *config_file)
    requires std::constructible_from<Model, const json &>
      : PPOAgent(std::move(action_mapper), load_json(config_file)) {}

  PPOAgent(ActionMapper action_mapper, const json &config)
    requires std::constructible_from<Model, const json &>
      : PPOAgent(std::move(action_mapper), config, config["model"]) {}

  template <typename TModelParams>
  PPOAgent(const char *config_file, TModelParams &&model_params)
    requires std::default_initializable<ActionMapper> &&
             std::constructible_from<Model, TModelParams &&>
      : PPOAgent(ActionMapper{}, load_json(config_file),
                 std::forward<TModelParams>(model_params)) {}

  template <typename TModelParams>
  PPOAgent(const json &config, TModelParams &&model_params)
    requires std::default_initializable<ActionMapper> &&
             std::constructible_from<Model, TModelParams &&>
      : PPOAgent(ActionMapper{}, config,
                 std::forward<TModelParams>(model_params)) {}

  template <typename TModelParams>
  PPOAgent(ActionMapper action_mapper, const char *config_file,
           TModelParams &&model_params)
    requires std::constructible_from<Model, TModelParams &&>
      : PPOAgent(std::move(action_mapper), load_json(config_file),
                 std::forward<TModelParams>(model_params)) {}

  template <typename TModelParams>
  PPOAgent(ActionMapper action_mapper, const json &config,
           TModelParams &&model_params)
    requires std::constructible_from<Model, TModelParams &&>
      : Base(config), action_mapper_(std::move(action_mapper)),
        gamma_(config.value("gamma", 0.99)),
        lambda_(config.value("gae_lambda", 0.95)),
        batch_steps_(config.value(
            "batch_steps",
            config.value("horizon", static_cast<std::size_t>(2048)))),
        normalize_adv_(config.value("normalize_advantage", true)),
        model_(std::forward<TModelParams>(model_params)) {
    static_assert(CPolicyModel<TModel>,
                  "TModel must satisfy the CPolicyModel concept");
    static_assert(std::copy_constructible<ActionParam>,
                  "PPO ActionParam must be copy constructible");
    if (batch_steps_ == 0) {
      throw std::invalid_argument("PPO batch_steps must be greater than zero");
    }
    buffer_.Reserve(batch_steps_);
  }

  void UpdateStateImpl() {
    // The current state is the next state of the pending transition. Finalize
    // that transition first. If this fills the batch, bootstrap and learn
    // before sampling from the policy that will generate the next batch.
    if (pending_transition_) {
      AppendPendingTransition();
      if constexpr (tAutoLearn) {
        if (BatchIsReady()) {
          Learn(model_.EvaluateValue(Base::state_));
        }
      }
    }

    auto decision = model_.EvaluateAction(Base::state_);
    Base::action_ = action_mapper_(decision.action_param);
    pending_transition_.emplace(
        PendingTransition{Base::state_, std::move(decision.action_param),
                          decision.log_prob, decision.value});
  }

  // A genuine environment terminal has no bootstrap value.
  void TerminatePathImpl() { FinishPath(0.0); }

  // A time-limit or externally truncated path bootstraps from its final
  // observation, but GAE still stops at this path boundary.
  void TerminatePathImpl(const State &final_state) {
    if (!pending_transition_)
      return;
    FinishPath(model_.EvaluateValue(final_state));
  }

  // Explicitly learn from a final partial batch. Every collected path must
  // already have been closed with TerminatePath so its bootstrap is known.
  void FlushBatch()
    requires(tAutoLearn)
  {
    if (pending_transition_) {
      throw std::logic_error(
          "TerminatePath must close the current path before FlushBatch");
    }
    Learn(0.0);
  }

  void SetLearningRate(double alpha) { model_.SetLearningRate(alpha); }

  void SetGamma(double gamma) { gamma_ = gamma; }

  void SetLambda(double lambda) { lambda_ = lambda; }

  auto &GetModel() { return model_; }
  const auto &GetModel() const { return model_; }

  const Buffer &GetBuffer() const { return buffer_; }

  std::size_t BufferSize() const { return buffer_.Size(); }

  // Transfer is only safe at a path boundary. This is the deployment-facing
  // operation used by threaded or distributed collectors.
  Buffer ReleaseBuffer() {
    if (pending_transition_) {
      throw std::logic_error(
          "TerminatePath must close the current path before ReleaseBuffer");
    }
    buffer_.ValidateComplete();
    Buffer released = std::move(buffer_);
    buffer_ = Buffer{};
    buffer_.Reserve(batch_steps_);
    return released;
  }

private:
  struct PendingTransition {
    State state;
    ActionParam action_param;
    double log_prob;
    double value;
  };

  void AppendPendingTransition(
      std::optional<double> path_end_bootstrap = std::nullopt) {
    buffer_.states.push_back(std::move(pending_transition_->state));
    buffer_.action_params.push_back(
        std::move(pending_transition_->action_param));
    buffer_.old_log_probs.push_back(pending_transition_->log_prob);
    buffer_.values.push_back(pending_transition_->value);
    buffer_.rewards.push_back(static_cast<double>(Base::reward_));
    buffer_.path_end_bootstraps.push_back(path_end_bootstrap);
    pending_transition_.reset();
  }

  void FinishPath(double bootstrap_value) {
    if (!pending_transition_)
      return;
    AppendPendingTransition(bootstrap_value);
    if constexpr (tAutoLearn) {
      if (BatchIsReady()) {
        // The final transition's path marker supplies the bootstrap value.
        Learn(0.0);
      }
    }
  }

  bool BatchIsReady() const { return buffer_.Size() >= batch_steps_; }

  void Learn(double trailing_bootstrap) {
    if (buffer_.Empty())
      return;
    const auto targets = ComputePPOTargets(buffer_, gamma_, lambda_,
                                           normalize_adv_, trailing_bootstrap);
    model_.LearnFromBatch(buffer_.states, buffer_.action_params,
                          buffer_.old_log_probs, targets.advantages,
                          targets.returns);
    ClearBatch();
  }

  void ClearBatch() { buffer_.Clear(); }

  ActionMapper action_mapper_;
  double gamma_;
  double lambda_;
  std::size_t batch_steps_;
  bool normalize_adv_;
  Model model_;

  std::optional<PendingTransition> pending_transition_;
  Buffer buffer_;
};

// Convenience wrapper for categorical policies whose integral ActionParam
// indexes an environment action list.
template <typename TModel, typename TAction, typename TReward,
          bool tAutoLearn = true>
class DiscretePPOAgent
    : public PPOAgent<
          TModel, TAction, TReward,
          IndexedActionMapper<typename TModel::ActionParam, TAction>,
          tAutoLearn> {
public:
  using ActionParam = typename TModel::ActionParam;
  static_assert(std::integral<ActionParam>,
                "DiscretePPOAgent requires an integral ActionParam");

  using ActionMapper = IndexedActionMapper<ActionParam, TAction>;
  using Base = PPOAgent<TModel, TAction, TReward, ActionMapper, tAutoLearn>;
  using ActionsList = typename ActionMapper::ActionsList;

  DiscretePPOAgent(const ActionsList &actions, const char *config_file)
      : Base(ActionMapper(actions), config_file) {}

  DiscretePPOAgent(const ActionsList &actions, const json &config)
      : Base(ActionMapper(actions), config) {}

  template <typename TModelParams>
  DiscretePPOAgent(const ActionsList &actions, const char *config_file,
                   TModelParams &&model_params)
      : Base(ActionMapper(actions), config_file,
             std::forward<TModelParams>(model_params)) {}

  template <typename TModelParams>
  DiscretePPOAgent(const ActionsList &actions, const json &config,
                   TModelParams &&model_params)
      : Base(ActionMapper(actions), config,
             std::forward<TModelParams>(model_params)) {}

  DiscretePPOAgent(ActionsList &&actions, const char *config_file)
      : Base(ActionMapper(std::move(actions)), config_file) {}

  DiscretePPOAgent(ActionsList &&actions, const json &config)
      : Base(ActionMapper(std::move(actions)), config) {}

  template <typename TModelParams>
  DiscretePPOAgent(ActionsList &&actions, const char *config_file,
                   TModelParams &&model_params)
      : Base(ActionMapper(std::move(actions)), config_file,
             std::forward<TModelParams>(model_params)) {}

  template <typename TModelParams>
  DiscretePPOAgent(ActionsList &&actions, const json &config,
                   TModelParams &&model_params)
      : Base(ActionMapper(std::move(actions)), config,
             std::forward<TModelParams>(model_params)) {}
};

} // namespace RLlib
#endif
