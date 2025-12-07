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
      } -> std::convertible_to<const typename TModel::ResultsList &>;
      {
        model.Update(state, action_idx, action_value, td_target)
      } -> std::same_as<void>;
      { model.SetLearningRate(0.1) } -> std::same_as<void>;
      { model.OutputModel(std::string_view{}) } -> std::same_as<void>;
      { model.LoadModel(std::string_view{}) } -> std::same_as<void>;
    };

template <typename TAgent>
concept CAgent = requires(TAgent agent, typename TAgent::State state,
                          typename TAgent::Reward reward) {
  { agent.UpdateStateImpl() } -> std::same_as<void>;
  { agent.CollectReward(reward) } -> std::same_as<bool>;
  { agent.ResetRound() } -> std::same_as<void>;
};

template <typename TDerived, typename TAction, typename TReward,
          typename TState>
class AgentBase {
 public:
  using Action = TAction;
  using Reward = TReward;
  using State = TState;

  AgentBase() {
    static_assert(CAgent<TDerived>, "TDerived must satisfy the CAgent concept");
  }

  const Action &UpdateState(const State &state) {
    state_ = state;
    ++round_;
    if (learning_rates_.size() != 0) {
      Derived().SetLearningRate(learning_rates_[std::min(
          static_cast<size_t>(round_), learning_rates_.size() - 1)]);
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
  std::vector<double> learning_rates_{};
};

}  // namespace RLlib
#endif  // AGENT_H