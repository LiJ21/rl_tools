#include <distributed_learner.h>
#include <gtest/gtest.h>
#include <models/ppo_learner.h>
#include <models/torch/actor_critic.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <string_view>
#include <thread>
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

  struct LearnedBatch {
    std::vector<State> states;
    std::vector<ActionParam> action_params;
    std::vector<double> old_log_probs;
    std::vector<double> advantages;
    std::vector<double> returns;
  };

  explicit MockPolicyModel(const json &) {}

  Decision EvaluateAction(const State &state) {
    return Decision{policy_version % 2, -0.5, state.at(0)};
  }

  double EvaluateValue(const State &state) { return state.at(0); }

  void LearnFromBatch(const std::vector<State> &states,
                      const std::vector<ActionParam> &action_params,
                      const std::vector<double> &old_log_probs,
                      const std::vector<double> &advantages,
                      const std::vector<double> &returns) {
    learned_batches.push_back(
        {states, action_params, old_log_probs, advantages, returns});
    ++policy_version;
  }

  void ImportWeights(const MockPolicyModel &source) {
    policy_version = source.policy_version;
  }

  void SetLearningRate(double) {}
  void OutputModel(std::string_view) {}
  void LoadModel(std::string_view) {}

  int policy_version = 0;
  std::vector<LearnedBatch> learned_batches;
};

static_assert(RLlib::CPolicyModel<MockPolicyModel>);
static_assert(RLlib::CWeightImportableModel<MockPolicyModel>);

using Buffer = RLlib::PPORolloutBuffer<MockPolicyModel::State,
                                       MockPolicyModel::ActionParam>;

TEST(DistributedLearner, ConcatenatesCompleteBuffersAndLearnsOnce) {
  MockPolicyModel model(json::object());
  RLlib::DistributedLearner learner(model, 1.0, 1.0, false);

  Buffer first{
      .states = {{1.0}, {2.0}},
      .action_params = {0, 1},
      .old_log_probs = {-0.1, -0.2},
      .values = {1.0, 2.0},
      .rewards = {2.0, 3.0},
      .path_end_bootstraps = {std::nullopt, 0.0},
  };
  Buffer second{
      .states = {{10.0}},
      .action_params = {1},
      .old_log_probs = {-0.3},
      .values = {10.0},
      .rewards = {4.0},
      .path_end_bootstraps = {0.0},
  };

  learner.Submit(first);
  learner.Submit(std::move(second));
  EXPECT_EQ(learner.SubmittedSteps(), 3);

  learner.Learn();

  EXPECT_TRUE(learner.Empty());
  ASSERT_EQ(model.learned_batches.size(), 1);
  const auto &batch = model.learned_batches.front();
  EXPECT_EQ(batch.states,
            std::vector<MockPolicyModel::State>({{1.0}, {2.0}, {10.0}}));
  EXPECT_EQ(batch.action_params, std::vector<int>({0, 1, 1}));
  EXPECT_EQ(batch.old_log_probs, std::vector<double>({-0.1, -0.2, -0.3}));
  EXPECT_EQ(batch.advantages, std::vector<double>({4.0, 1.0, -6.0}));
  EXPECT_EQ(batch.returns, std::vector<double>({5.0, 3.0, 4.0}));
}

TEST(DistributedLearner, NormalizesAdvantagesAcrossSubmittedBuffers) {
  MockPolicyModel model(json::object());
  RLlib::DistributedLearner learner(model, 1.0, 1.0, true);

  learner.Submit(Buffer{
      .states = {{0.0}},
      .action_params = {0},
      .old_log_probs = {-0.1},
      .values = {0.0},
      .rewards = {1.0},
      .path_end_bootstraps = {0.0},
  });
  learner.Submit(Buffer{
      .states = {{0.0}},
      .action_params = {1},
      .old_log_probs = {-0.2},
      .values = {0.0},
      .rewards = {3.0},
      .path_end_bootstraps = {0.0},
  });

  learner.Learn();

  ASSERT_EQ(model.learned_batches.size(), 1);
  const auto &advantages = model.learned_batches.front().advantages;
  ASSERT_EQ(advantages.size(), 2);
  EXPECT_NEAR(advantages[0], -1.0, 1e-7);
  EXPECT_NEAR(advantages[1], 1.0, 1e-7);
}

TEST(DistributedLearner, RejectsAnUnclosedSubmittedPath) {
  MockPolicyModel model(json::object());
  RLlib::DistributedLearner learner(model, 0.99, 0.95, true);

  Buffer incomplete{
      .states = {{1.0}},
      .action_params = {0},
      .old_log_probs = {-0.1},
      .values = {1.0},
      .rewards = {2.0},
      .path_end_bootstraps = {std::nullopt},
  };

  EXPECT_THROW(learner.Submit(std::move(incomplete)), std::logic_error);
  EXPECT_TRUE(learner.Empty());
  learner.Learn();
  EXPECT_TRUE(model.learned_batches.empty());
}

TEST(DistributedLearner, AcceptsConcurrentBufferSubmissions) {
  MockPolicyModel model(json::object());
  RLlib::DistributedLearner learner(model, 1.0, 1.0, false);

  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&learner, worker] {
      learner.Submit(Buffer{
          .states = {{static_cast<double>(worker)}},
          .action_params = {worker % 2},
          .old_log_probs = {-0.1},
          .values = {0.0},
          .rewards = {static_cast<double>(worker + 1)},
          .path_end_bootstraps = {0.0},
      });
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(learner.SubmittedSteps(), 8);
  learner.Learn();

  ASSERT_EQ(model.learned_batches.size(), 1);
  auto returns = model.learned_batches.front().returns;
  std::sort(returns.begin(), returns.end());
  EXPECT_EQ(returns,
            std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
}

using Network = RLlib::Models::ActorCriticNetwork<double, double>;
using TorchModel = RLlib::Models::PPOLearner<Network>;

static_assert(RLlib::CWeightImportableModel<TorchModel>);

json TorchModelConfig() {
  return {
      {"features_dim", 2},
      {"actions_dim", 2},
      {"hidden_dim", 3},
      {"learning_rate", 1e-3},
      {"epochs", 1},
      {"minibatch_size", 2},
      {"optimizer", {{"type", "adam"}}},
  };
}

TEST(PPOLearner, ImportsWeightsWithoutSharingTensorStorage) {
  TorchModel source(TorchModelConfig());
  TorchModel destination(TorchModelConfig());

  {
    torch::NoGradGuard no_grad;
    auto source_parameters = source.GetNet().parameters();
    auto destination_parameters = destination.GetNet().parameters();
    ASSERT_EQ(source_parameters.size(), destination_parameters.size());
    for (std::size_t index = 0; index < source_parameters.size(); ++index) {
      source_parameters[index].fill_(static_cast<double>(index + 1));
      destination_parameters[index].zero_();
    }
  }

  destination.ImportWeights(source);

  auto source_parameters = source.GetNet().parameters();
  auto destination_parameters = destination.GetNet().parameters();
  ASSERT_EQ(source_parameters.size(), destination_parameters.size());
  for (std::size_t index = 0; index < source_parameters.size(); ++index) {
    EXPECT_TRUE(torch::allclose(source_parameters[index],
                                destination_parameters[index]));
  }

  {
    torch::NoGradGuard no_grad;
    source_parameters.front().fill_(99.0);
  }
  EXPECT_FALSE(torch::allclose(source_parameters.front(),
                               destination_parameters.front()));
}

} // namespace
