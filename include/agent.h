#ifndef TRAINER_H
#define TRAINER_H

#include <utility>
#include <vector>

#include "random_generator.h"
namespace RLlib {

template <typename TModel>
concept CModel =
    requires(typename TModel::State state, TModel model, int action_idx,
             double action_value, double td_target) {
      {
        model.GetActionValues(state)
      } -> std::same_as<std::array<double, TModel::kActionsDim>>;
      {
        model.Update(state, action_idx, action_value, td_target)
      } -> std::same_as<void>;
    };

template <typename TDerived>
class AgentBase {
 public:
  using Action = typename TDerived::Action;
  using Reward = typename TDerived::Reward;
  using State = typename TDerived::State;

  AgentBase() = default;

  const Action &UpdateState(const State &state) {
    state_ = state;
    ++round_;
    if (learning_rates_.size() != 0) {
      Derived().SetLearningRate(
          learning_rates_[std::max(round_, learning_rates_.size() - 1)]);
    }
    Derived().UpdateStateImpl();

    return action_;
  }

  void ResetRound() { round_ = 0; }

  bool CollectReward(const Reward &reward, int round = -1) {
    if (round != round_ && round != -1) return false;
    reward_ = reward;
    return true;
  }

  void SetLearningRates(const std::vector<double> &learning_rates) {
    learning_rates_ = learning_rates;
  }

  void SetLearningRates(std::vector<double> &&learning_rates) {
    learning_rates_ = learning_rates;
  }

  void ResetLearningRates() { learning_rates_.clear(); }

 protected:
  int round_ = 0;
  Reward reward_;
  Action action_;
  State state_;

 private:
  auto &Derived() { return static_cast<TDerived &>(*this); }
  std::vector<double> learning_rates_;
};

template <CModel TModel>
class SarsaAgent : public AgentBase<SarsaAgent<TModel>> {
 public:
  static constexpr int kActionsDim = TModel::kActionsDim;
  using Model = TModel;
  using Base = AgentBase<SarsaAgent<TModel>>;
  using State = typename Model::State;
  using ActionsList = std::array<double, kActionsDim>;
  using Action = double;
  using Reward = double;

  template <typename TActions, typename... TArgs>
  SarsaAgent(TActions &&actions, double epsilon, double gamma, TArgs &&...args)
      : actions_(std::forward<decltype(actions)>(actions)),
        epsilon_(epsilon),
        gamma_(gamma),
        model_(std::forward<TArgs>(args)...) {}

  void UpdateStateImpl() {
    int idx_best_ = -1;
    double max_value_ = -1e9;
    auto action_values = model_.GetActionValues(Base::state_);

    for (int i = 0; i < kActionsDim; ++i) {
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

    if (!is_first_round_) {
      double td_target = Base::reward_ + gamma_ * max_value_;
      model_.Update(last_state_, last_action_idx_, last_action_value_,
                    td_target);
    } else {
      is_first_round_ = false;
    }

    last_state_ = Base::state_;
    last_action_idx_ = idx_result_;
    last_action_value_ = action_values[idx_result_];
  }

  void SetEpsilon(double epsilon) { epsilon_ = epsilon; }

  void SetGamma(double gamma) { gamma_ = gamma; }

  void SetLearningRate(double alpha) { model_.SetLearningRate(alpha); }

 private:
  Model model_;
  const ActionsList actions_;
  double epsilon_;
  State last_state_{};
  int last_action_idx_ = -1;
  double last_action_value_ = 0.0;
  double gamma_;
  bool is_first_round_ = true;
};
}  // namespace RLlib
#endif  // TRAINER_H