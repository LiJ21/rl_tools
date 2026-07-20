#include <agents/ppo.h>
#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

class MockPolicyModel {
public:
  using State = std::vector<double>;
  using ActionParam = int;

  struct Decision {
    ActionParam action_param;
    double log_prob;
    double value;
  };

  struct Rollout {
    std::vector<State> states;
    std::vector<ActionParam> action_params;
    std::vector<double> old_log_probs;
    std::vector<double> advantages;
    std::vector<double> returns;
  };

  explicit MockPolicyModel(const json &) {}

  Decision EvaluateAction(const State &state) {
    action_evaluation_versions.push_back(policy_version);
    action_evaluation_states.push_back(state);
    return Decision{policy_version % 2,
                    static_cast<double>(policy_version) + state.at(0) / 100.0,
                    state.at(0)};
  }

  double EvaluateValue(const State &state) {
    value_evaluation_versions.push_back(policy_version);
    value_evaluation_states.push_back(state);
    return state.at(0);
  }

  void LearnFromBatch(const std::vector<State> &states,
                      const std::vector<ActionParam> &action_params,
                      const std::vector<double> &old_log_probs,
                      const std::vector<double> &advantages,
                      const std::vector<double> &returns) {
    rollouts.push_back(
        Rollout{states, action_params, old_log_probs, advantages, returns});
    ++policy_version;
  }

  void SetLearningRate(double learning_rate) {
    last_learning_rate = learning_rate;
  }

  void OutputModel(std::string_view) {}
  void LoadModel(std::string_view) {}

  int policy_version = 0;
  double last_learning_rate = 0.0;
  std::vector<int> action_evaluation_versions;
  std::vector<State> action_evaluation_states;
  std::vector<int> value_evaluation_versions;
  std::vector<State> value_evaluation_states;
  std::vector<Rollout> rollouts;
};

struct CustomActionParam {
  double amplitude;
  bool stop;

  bool operator==(const CustomActionParam &) const = default;
};

class CustomActionPolicyModel {
public:
  using State = std::vector<double>;
  using ActionParam = CustomActionParam;

  struct Decision {
    ActionParam action_param;
    double log_prob;
    double value;
  };

  explicit CustomActionPolicyModel(const json &) {}

  Decision EvaluateAction(const State &state) {
    return Decision{ActionParam{state.at(0), false}, -0.5, state.at(0)};
  }

  double EvaluateValue(const State &state) { return state.at(0); }

  void LearnFromBatch(const std::vector<State> &,
                      const std::vector<ActionParam> &action_params,
                      const std::vector<double> &, const std::vector<double> &,
                      const std::vector<double> &) {
    learned_action_params.push_back(action_params);
  }

  void SetLearningRate(double) {}
  void OutputModel(std::string_view) {}
  void LoadModel(std::string_view) {}

  std::vector<std::vector<ActionParam>> learned_action_params;
};

struct AmplitudeMapper {
  double operator()(const CustomActionParam &action_param) const {
    return 2.0 * action_param.amplitude;
  }
};

using Agent = RLlib::DiscretePPOAgent<MockPolicyModel, int, double>;
using GenericAgent = RLlib::PPOAgent<MockPolicyModel, int, double>;
using CustomActionAgent =
    RLlib::PPOAgent<CustomActionPolicyModel, double, double, AmplitudeMapper>;

json AgentConfig(std::size_t batch_steps = 16) {
  return {
      {"gamma", 1.0},
      {"gae_lambda", 1.0},
      {"batch_steps", batch_steps},
      {"normalize_advantage", false},
      {"model", json::object()},
  };
}

class NoTerminationAgent
    : public RLlib::AgentBase<NoTerminationAgent, int, double, int> {
public:
  using Base = RLlib::AgentBase<NoTerminationAgent, int, double, int>;
  using State = int;
  using Reward = double;

  NoTerminationAgent() : Base(json::object()) {}

  void UpdateStateImpl() { action_ = state_; }
  void SetLearningRate(double) {}
};

TEST(AgentBaseTermination, IsNoOpWithoutDerivedHook) {
  NoTerminationAgent agent;

  EXPECT_EQ(agent.UpdateState(3), 3);
  agent.TerminatePath();
  agent.TerminatePath(4);
  EXPECT_EQ(agent.UpdateState(5), 5);
}

TEST(PPOActionMapper, MapsAndValidatesIntegralParameters) {
  RLlib::IndexedActionMapper<int, std::string> mapper({"left", "right"});

  EXPECT_EQ(mapper(0), "left");
  EXPECT_EQ(mapper(1), "right");
  EXPECT_THROW(mapper(-1), std::out_of_range);
  EXPECT_THROW(mapper(2), std::out_of_range);
}

TEST(PPOAgent, IdentityMapperNeedsNoActionList) {
  GenericAgent agent(AgentConfig());

  EXPECT_EQ(agent.UpdateState({1.0}), 0);
  agent.CollectReward(2.0);
  agent.TerminatePath();
  EXPECT_TRUE(agent.GetModel().rollouts.empty());
  agent.FlushBatch();

  ASSERT_EQ(agent.GetModel().rollouts.size(), 1);
  EXPECT_EQ(agent.GetModel().rollouts[0].action_params, std::vector<int>({0}));
}

TEST(PPOAgent, PreservesCustomActionParamForLearning) {
  CustomActionAgent agent(AmplitudeMapper{}, AgentConfig());

  EXPECT_EQ(agent.UpdateState({1.5}), 3.0);
  agent.CollectReward(2.0);
  agent.TerminatePath();
  EXPECT_TRUE(agent.GetModel().learned_action_params.empty());
  agent.FlushBatch();

  const auto &batches = agent.GetModel().learned_action_params;
  ASSERT_EQ(batches.size(), 1);
  EXPECT_EQ(batches[0], std::vector<CustomActionParam>({{1.5, false}}));
}

TEST(DiscretePPOAgent, TerminalAppendsWithoutLearningPartialBatch) {
  Agent agent({10, 20}, AgentConfig());

  EXPECT_EQ(agent.UpdateState({1.0}), 10);
  agent.CollectReward(5.0);
  EXPECT_EQ(agent.UpdateState({2.0}), 10);
  agent.CollectReward(7.0);
  agent.TerminatePath();

  EXPECT_TRUE(agent.GetModel().rollouts.empty());
  agent.FlushBatch();

  const auto &model = agent.GetModel();
  ASSERT_EQ(model.rollouts.size(), 1);
  const auto &rollout = model.rollouts.front();
  ASSERT_EQ(rollout.states.size(), 2);
  EXPECT_EQ(rollout.states[0], MockPolicyModel::State({1.0}));
  EXPECT_EQ(rollout.states[1], MockPolicyModel::State({2.0}));
  EXPECT_EQ(rollout.action_params, std::vector<int>({0, 0}));
  EXPECT_EQ(rollout.returns, std::vector<double>({12.0, 7.0}));
  EXPECT_EQ(rollout.advantages, std::vector<double>({11.0, 5.0}));
}

TEST(DiscretePPOAgent, BatchStepsLearnsBeforeSamplingNextAction) {
  Agent agent({10, 20}, AgentConfig(2));

  EXPECT_EQ(agent.UpdateState({1.0}), 10);
  agent.CollectReward(1.0);
  EXPECT_EQ(agent.UpdateState({2.0}), 10);
  agent.CollectReward(2.0);

  // This state bootstraps the first rollout. Learning increments the mock
  // policy version, so its action must be sampled from version 1.
  EXPECT_EQ(agent.UpdateState({3.0}), 20);

  const auto &model = agent.GetModel();
  ASSERT_EQ(model.rollouts.size(), 1);
  EXPECT_EQ(model.action_evaluation_versions, std::vector<int>({0, 0, 1}));
  EXPECT_EQ(model.value_evaluation_versions, std::vector<int>({0}));
  EXPECT_EQ(model.value_evaluation_states,
            std::vector<MockPolicyModel::State>({{3.0}}));
  EXPECT_EQ(model.rollouts[0].returns, std::vector<double>({6.0, 5.0}));
}

TEST(DiscretePPOAgent, TruncationBootstrapsFromFinalObservation) {
  Agent agent({10, 20}, AgentConfig());

  agent.UpdateState({1.0});
  agent.CollectReward(2.0);
  agent.TerminatePath(MockPolicyModel::State{10.0});

  EXPECT_TRUE(agent.GetModel().rollouts.empty());
  agent.FlushBatch();

  const auto &model = agent.GetModel();
  ASSERT_EQ(model.rollouts.size(), 1);
  EXPECT_EQ(model.rollouts[0].returns, std::vector<double>({12.0}));
  EXPECT_EQ(model.rollouts[0].advantages, std::vector<double>({11.0}));
  EXPECT_EQ(model.value_evaluation_states,
            std::vector<MockPolicyModel::State>({{10.0}}));
}

TEST(DiscretePPOAgent, PathsDoNotShareAdvantages) {
  Agent agent({10, 20}, AgentConfig(2));

  agent.UpdateState({1.0});
  agent.CollectReward(2.0);
  agent.TerminatePath();
  EXPECT_TRUE(agent.GetModel().rollouts.empty());

  agent.UpdateState({3.0});
  agent.CollectReward(4.0);
  agent.TerminatePath();

  const auto &rollouts = agent.GetModel().rollouts;
  ASSERT_EQ(rollouts.size(), 1);
  EXPECT_EQ(agent.GetModel().action_evaluation_versions,
            std::vector<int>({0, 0}));
  EXPECT_EQ(rollouts[0].states.size(), 2);
  EXPECT_EQ(rollouts[0].returns, std::vector<double>({2.0, 4.0}));
  EXPECT_EQ(rollouts[0].advantages, std::vector<double>({1.0, 1.0}));
}

TEST(DiscretePPOAgent, CompletedPathIsIsolatedFromTrailingBootstrap) {
  Agent agent({10, 20}, AgentConfig(2));

  agent.UpdateState({1.0});
  agent.CollectReward(2.0);
  agent.TerminatePath();

  agent.UpdateState({3.0});
  agent.CollectReward(4.0);

  // This fills the batch in the second path. V({5}) bootstraps only that
  // unfinished path; it must not flow backward through the terminal marker.
  EXPECT_EQ(agent.UpdateState({5.0}), 20);

  const auto &model = agent.GetModel();
  ASSERT_EQ(model.rollouts.size(), 1);
  EXPECT_EQ(model.rollouts[0].returns, std::vector<double>({2.0, 9.0}));
  EXPECT_EQ(model.rollouts[0].advantages, std::vector<double>({1.0, 6.0}));
  EXPECT_EQ(model.value_evaluation_states,
            std::vector<MockPolicyModel::State>({{5.0}}));
  EXPECT_EQ(model.action_evaluation_versions, std::vector<int>({0, 0, 1}));
}

TEST(DiscretePPOAgent, EmptyAndRepeatedTerminationAreNoOps) {
  Agent agent({10, 20}, AgentConfig());

  agent.TerminatePath();
  agent.UpdateState({1.0});
  agent.CollectReward(2.0);
  agent.TerminatePath();
  agent.TerminatePath();
  agent.TerminatePath(MockPolicyModel::State{5.0});

  EXPECT_TRUE(agent.GetModel().rollouts.empty());
  agent.FlushBatch();
  EXPECT_EQ(agent.GetModel().rollouts.size(), 1);
  EXPECT_TRUE(agent.GetModel().value_evaluation_states.empty());
}

TEST(DiscretePPOAgent, FlushBatchRequiresAClosedPath) {
  Agent agent({10, 20}, AgentConfig());

  agent.UpdateState({1.0});
  agent.CollectReward(2.0);

  EXPECT_THROW(agent.FlushBatch(), std::logic_error);

  agent.TerminatePath();
  agent.FlushBatch();
  EXPECT_EQ(agent.GetModel().rollouts.size(), 1);
}

TEST(DiscretePPOAgent, ResetRoundDoesNotDiscardPath) {
  Agent agent({10, 20}, AgentConfig());

  agent.UpdateState({1.0});
  agent.CollectReward(2.0);
  agent.ResetRound();
  agent.TerminatePath();
  agent.FlushBatch();

  ASSERT_EQ(agent.GetModel().rollouts.size(), 1);
  EXPECT_EQ(agent.GetModel().rollouts[0].returns, std::vector<double>({2.0}));
}

} // namespace
