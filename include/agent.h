#ifndef TRAINER_H
#define TRAINER_H

#include <extern/miniexpr.h>

#include <extern/json.hpp>
#include <algorithm>
#include <concepts>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;

namespace RLlib {

template <typename TModel>
concept CModel = requires(typename TModel::State state, TModel model,
                          const typename TModel::ActionParam &action_param,
                          double td_target) {
  {
    model.GetActionValues(state)
  } -> std::convertible_to<const typename TModel::ResultsList &>;
  { model.Update(state, action_param, td_target) } -> std::same_as<void>;
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

  AgentBase(const json &config = {}) {
    static_assert(CAgent<TDerived>, "TDerived must satisfy the CAgent concept");
    if (config.contains("learning_rates")) {
      if (config["learning_rates"].is_array()) {
        learning_rates_ = config["learning_rates"].get<std::vector<double>>();
      } else if (config["learning_rates"].is_string()) {
        std::cout << "Using learning_rates formula: "
                  << config["learning_rates"].get<std::string>() << std::endl;
        learning_rates_formula_ = config["learning_rates"].get<std::string>();
      } else {
        throw std::runtime_error(
            "Invalid learning_rates format in config JSON");
      }
    }
    round_ = config.value("init_round", 0);
  }

  const Action &UpdateState(const State &state) {
    state_ = state;
    ++round_;
    if (learning_rates_.size() != 0) {
      Derived().SetLearningRate(learning_rates_[std::min(
          static_cast<size_t>(round_), learning_rates_.size() - 1)]);
    } else if (!learning_rates_formula_.empty()) {
      double round_double = static_cast<double>(round_);
      mini_expr::Parser::Vars vars{
          {"round", round_double},
      };
      double lr = mini_expr::eval(learning_rates_formula_, vars);
      Derived().SetLearningRate(lr);
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

  // Finish a genuine terminal path. Agents that do not provide a
  // TerminatePathImpl hook retain their existing behavior.
  void TerminatePath() {
    if constexpr (requires(TDerived &agent) { agent.TerminatePathImpl(); }) {
      Derived().TerminatePathImpl();
    }
  }

  // Finish a truncated path and allow agents to bootstrap from its final
  // observation. This is also a no-op unless the derived agent opts in.
  void TerminatePath(const State &final_state) {
    if constexpr (requires(TDerived &agent, const State &state) {
                    agent.TerminatePathImpl(state);
                  }) {
      Derived().TerminatePathImpl(final_state);
    }
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
  std::string learning_rates_formula_{};
};

}  // namespace RLlib
#endif  // AGENT_H
