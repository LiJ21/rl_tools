#ifndef AGENTS_PPO_H
#define AGENTS_PPO_H
#include <agent.h>

#include <cmath>
#include <concepts>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

// Proximal Policy Optimization agent. The model defines the probability-bearing
// ActionParam stored in the batch, while ActionMapper converts it into the
// environment-facing Action returned by UpdateState.
template <typename TModel, typename TAction, typename TReward,
          typename TActionMapper =
              IdentityActionMapper<typename TModel::ActionParam, TAction>>
class PPOAgent
    : public AgentBase<PPOAgent<TModel, TAction, TReward, TActionMapper>,
                       TAction, TReward, typename TModel::State> {
public:
  using Model = TModel;
  using Self = PPOAgent<TModel, TAction, TReward, TActionMapper>;
  using Base = AgentBase<Self, TAction, TReward, typename TModel::State>;
  using State = typename Model::State;
  using Action = TAction;
  using ActionParam = typename Model::ActionParam;
  using ActionMapper = TActionMapper;
  using Reward = TReward;

  static_assert(std::invocable<ActionMapper &, const ActionParam &>,
                "PPO ActionMapper must accept const ActionParam&");
  static_assert(
      std::convertible_to<
          std::invoke_result_t<ActionMapper &, const ActionParam &>, Action>,
      "PPO ActionMapper result must be convertible to the environment Action");

  PPOAgent(const char *config_file)
    requires std::default_initializable<ActionMapper>
      : PPOAgent(ActionMapper{}, load_json(config_file)) {}

  PPOAgent(const json &config)
    requires std::default_initializable<ActionMapper>
      : PPOAgent(ActionMapper{}, config) {}

  PPOAgent(ActionMapper action_mapper, const char *config_file)
      : PPOAgent(std::move(action_mapper), load_json(config_file)) {}

  PPOAgent(ActionMapper action_mapper, const json &config)
      : Base(config), action_mapper_(std::move(action_mapper)),
        gamma_(config.value("gamma", 0.99)),
        lambda_(config.value("gae_lambda", 0.95)),
        batch_steps_(config.value(
            "batch_steps",
            config.value("horizon", static_cast<std::size_t>(2048)))),
        normalize_adv_(config.value("normalize_advantage", true)),
        model_(config["model"]) {
    static_assert(CPolicyModel<TModel>,
                  "TModel must satisfy the CPolicyModel concept");
    static_assert(std::copy_constructible<ActionParam>,
                  "PPO ActionParam must be copy constructible");
    if (batch_steps_ == 0) {
      throw std::invalid_argument("PPO batch_steps must be greater than zero");
    }
    states_.reserve(batch_steps_);
    action_params_.reserve(batch_steps_);
    log_probs_.reserve(batch_steps_);
    values_.reserve(batch_steps_);
    rewards_.reserve(batch_steps_);
    path_end_bootstraps_.reserve(batch_steps_);
  }

  void UpdateStateImpl() {
    // The current state is the next state of the pending transition. Finalize
    // that transition first. If this fills the batch, bootstrap and learn
    // before sampling from the policy that will generate the next batch.
    if (pending_transition_) {
      AppendPendingTransition();
      if (BatchIsReady()) {
        Learn(model_.EvaluateValue(Base::state_));
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
  void FlushBatch() {
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

private:
  struct PendingTransition {
    State state;
    ActionParam action_param;
    double log_prob;
    double value;
  };

  void AppendPendingTransition(
      std::optional<double> path_end_bootstrap = std::nullopt) {
    states_.push_back(std::move(pending_transition_->state));
    action_params_.push_back(std::move(pending_transition_->action_param));
    log_probs_.push_back(pending_transition_->log_prob);
    values_.push_back(pending_transition_->value);
    rewards_.push_back(static_cast<double>(Base::reward_));
    path_end_bootstraps_.push_back(path_end_bootstrap);
    pending_transition_.reset();
  }

  void FinishPath(double bootstrap_value) {
    if (!pending_transition_)
      return;
    AppendPendingTransition(bootstrap_value);
    if (BatchIsReady()) {
      // The final transition's path marker supplies the bootstrap value.
      Learn(0.0);
    }
  }

  bool BatchIsReady() const { return states_.size() >= batch_steps_; }

  void Learn(double trailing_bootstrap) {
    const std::size_t N = states_.size();
    if (N == 0)
      return;
    std::vector<double> advantages(N), returns(N);

    double gae = 0.0;
    double next_value = trailing_bootstrap;
    for (int t = static_cast<int>(N) - 1; t >= 0; --t) {
      if (path_end_bootstraps_[t]) {
        // Do not let GAE cross into a different episode. A truncation still
        // bootstraps its delta, while a true terminal stores zero.
        gae = 0.0;
        next_value = *path_end_bootstraps_[t];
      }
      double delta = rewards_[t] + gamma_ * next_value - values_[t];
      gae = delta + gamma_ * lambda_ * gae;
      advantages[t] = gae;
      returns[t] = gae + values_[t];
      next_value = values_[t];
    }

    if (normalize_adv_ && N > 1) {
      double mean = 0.0;
      for (double a : advantages)
        mean += a;
      mean /= static_cast<double>(N);
      double var = 0.0;
      for (double a : advantages)
        var += (a - mean) * (a - mean);
      var /= static_cast<double>(N);
      double stddev = std::sqrt(var) + 1e-8;
      for (double &a : advantages)
        a = (a - mean) / stddev;
    }

    model_.LearnFromBatch(states_, action_params_, log_probs_, advantages,
                          returns);
    ClearBatch();
  }

  void ClearBatch() {
    states_.clear();
    action_params_.clear();
    log_probs_.clear();
    values_.clear();
    rewards_.clear();
    path_end_bootstraps_.clear();
  }

  ActionMapper action_mapper_;
  double gamma_;
  double lambda_;
  std::size_t batch_steps_;
  bool normalize_adv_;
  Model model_;

  std::optional<PendingTransition> pending_transition_;

  std::vector<State> states_;
  std::vector<ActionParam> action_params_;
  std::vector<double> log_probs_;
  std::vector<double> values_;
  std::vector<double> rewards_;
  std::vector<std::optional<double>> path_end_bootstraps_;
};

// Convenience wrapper for categorical policies whose integral ActionParam
// indexes an environment action list.
template <typename TModel, typename TAction, typename TReward>
class DiscretePPOAgent
    : public PPOAgent<
          TModel, TAction, TReward,
          IndexedActionMapper<typename TModel::ActionParam, TAction>> {
public:
  using ActionParam = typename TModel::ActionParam;
  static_assert(std::integral<ActionParam>,
                "DiscretePPOAgent requires an integral ActionParam");

  using ActionMapper = IndexedActionMapper<ActionParam, TAction>;
  using Base = PPOAgent<TModel, TAction, TReward, ActionMapper>;
  using ActionsList = typename ActionMapper::ActionsList;

  DiscretePPOAgent(const ActionsList &actions, const char *config_file)
      : Base(ActionMapper(actions), config_file) {}

  DiscretePPOAgent(const ActionsList &actions, const json &config)
      : Base(ActionMapper(actions), config) {}

  DiscretePPOAgent(ActionsList &&actions, const char *config_file)
      : Base(ActionMapper(std::move(actions)), config_file) {}

  DiscretePPOAgent(ActionsList &&actions, const json &config)
      : Base(ActionMapper(std::move(actions)), config) {}
};

} // namespace RLlib
#endif
