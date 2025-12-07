#ifndef AGENTS_SARSA_H
#define AGENTS_SARSA_H
#include <agent.h>
#include <models/linear.h>
#include <models/tabular.h>

namespace RLlib {

enum class SarsaTrainingMode { kOnPolicy = 0, kQLearning = 1 };
template <typename TModel, typename TAction, typename TReward>
class SarsaAgent : public AgentBase<SarsaAgent<TModel, TAction, TReward>,
                                    TAction, TReward, typename TModel::State> {
 public:
  static constexpr int kActionsDim = TModel::kActionsDim;
  using Model = TModel;
  using Base = AgentBase<SarsaAgent<TModel, TAction, TReward>, TAction, TReward,
                         typename TModel::State>;
  using State = typename Model::State;
  using Action = TAction;
  using ActionsList = std::array<Action, kActionsDim>;
  using Reward = TReward;

  template <typename... TArgs>
  SarsaAgent(const ActionsList &actions, double epsilon, double gamma,
             TArgs &&...args)
      : actions_(actions),
        epsilon_(epsilon),
        gamma_(gamma),
        current_gamma_(1.0),
        model_(std::forward<TArgs>(args)...) {
    static_assert(CModel<TModel>, "TModel must satisfy the CModel concept");
  }

  void UpdateStateImpl() {
    int idx_best_ = 0;
    auto action_values = model_.GetActionValues(Base::state_);
    double max_value_ = action_values[0];

    for (int i = 1; i < kActionsDim; ++i) {
      if (action_values[i] > max_value_) {
        idx_best_ = i;
        max_value_ = action_values[i];
      }
    }

    // epsilon greedy action selection
    int idx_result_{};

    if (rng_util::uniform01() < epsilon_) {
      int idx_random_ =
          static_cast<int>(rng_util::uniform01() * (kActionsDim - 1));
      idx_result_ = idx_random_ < idx_best_ ? idx_random_ : idx_random_ + 1;
    } else {
      idx_result_ = idx_best_;
    }
    Base::action_ = actions_[idx_result_];
    double new_action_value{};
    if (training_mode_ == SarsaTrainingMode::kQLearning) {
      new_action_value = max_value_;
    } else if (training_mode_ == SarsaTrainingMode::kOnPolicy) {
      new_action_value = action_values[idx_result_];
    }

    if (Base::round_ % steps_ == 0) {
      if (!is_first_round_) {
        target_ += current_gamma_ * (Base::reward_ + gamma_ * new_action_value);
        model_.Update(last_state_, last_action_idx_, last_action_value_,
                      target_);
        target_ = 0.0;
      } else {
        is_first_round_ = false;
      }

      last_state_ = Base::state_;
      last_action_idx_ = idx_result_;
      last_action_value_ = action_values[idx_result_];
    } else {
      target_ += current_gamma_ * Base::reward_;
      current_gamma_ *= gamma_;
    }
  }

  void SetEpsilon(double epsilon) { epsilon_ = epsilon; }

  void SetGamma(double gamma) { gamma_ = gamma; }

  void SetLearningRate(double alpha) { model_.SetLearningRate(alpha); }

  auto &GetModel() { return model_; }

  void SetSteps(size_t steps) { steps_ = steps; }

  void SetTrainingMode(SarsaTrainingMode mode) { training_mode_ = mode; }

 private:
  Model model_;
  const ActionsList actions_;
  double epsilon_;
  State last_state_{};
  int last_action_idx_ = -1;
  double last_action_value_ = 0.0;
  double gamma_;
  bool is_first_round_ = true;
  double target_{};
  size_t steps_{1};
  double current_gamma_{};
  SarsaTrainingMode training_mode_{SarsaTrainingMode::kOnPolicy};
};

template <int tFeaturesDim, int tActionsDim, typename TAction = double,
          typename TFeature = double, typename TReward = double>
using LinearSarsaAgent =
    SarsaAgent<Models::SimpleLinearModel<tFeaturesDim, tActionsDim, TFeature>,
               TAction, TReward>;

template <int tStatesDim, int tActionsDim, typename TAction = int,
          typename TReward = double>
using TabularSarsaAgent =
    SarsaAgent<Models::Tabular<tStatesDim, tActionsDim>, TAction, TReward>;
}  // namespace RLlib
#endif