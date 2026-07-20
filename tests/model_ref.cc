#include <agent.h>
#include <agents/ppo.h>
#include <gtest/gtest.h>

#include <concepts>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class ValueModel {
public:
  using State = int;
  using ActionParam = int;
  using ResultsList = std::vector<double>;

  const ResultsList &GetActionValues(const State &state) {
    values = {static_cast<double>(state)};
    return values;
  }

  void Update(const State &state, const ActionParam &action_param,
              double target) {
    last_update = {state, action_param, target};
  }

  void SetLearningRate(double value) { learning_rate = value; }
  void OutputModel(std::string_view path) { output_path = path; }
  void LoadModel(std::string_view path) { load_path = path; }

  struct UpdateRecord {
    int state = 0;
    int action_param = 0;
    double target = 0.0;
  };

  ResultsList values;
  UpdateRecord last_update;
  double learning_rate = 0.0;
  std::string output_path;
  std::string load_path;
};

class PolicyModel {
public:
  using State = int;
  using ActionParam = int;

  struct Decision {
    ActionParam action_param;
    double log_prob;
    double value;
  };

  Decision EvaluateAction(const State &state) {
    return {state + weight, -0.25, static_cast<double>(state)};
  }

  double EvaluateValue(const State &state) { return state + weight; }

  void LearnFromBatch(const std::vector<State> &states,
                      const std::vector<ActionParam> &,
                      const std::vector<double> &, const std::vector<double> &,
                      const std::vector<double> &) {
    learned_states = states;
  }

  void SetLearningRate(double value) { learning_rate = value; }
  void OutputModel(std::string_view path) { output_path = path; }
  void LoadModel(std::string_view path) { load_path = path; }

  void ImportWeights(const PolicyModel &source) { weight = source.weight; }

  int weight = 0;
  double learning_rate = 0.0;
  std::vector<State> learned_states;
  std::string output_path;
  std::string load_path;
};

using ValueModelRef = RLlib::ModelRef<ValueModel>;
using PolicyModelRef = RLlib::ModelRef<PolicyModel>;

static_assert(RLlib::CModel<ValueModelRef>);
static_assert(RLlib::CPolicyModel<PolicyModelRef>);
static_assert(RLlib::CWeightImportableModel<PolicyModelRef>);
static_assert(std::copy_constructible<ValueModelRef>);
static_assert(!std::constructible_from<ValueModelRef, ValueModel &&>);

TEST(ModelRef, ForwardsValueModelOperations) {
  ValueModel model;
  ValueModelRef ref(model);

  EXPECT_EQ(&ref.Get(), &model);
  EXPECT_EQ(ref.GetActionValues(3), std::vector<double>({3.0}));

  ref.Update(4, 2, 7.5);
  ref.SetLearningRate(0.01);
  ref.OutputModel("output.json");
  ref.LoadModel("input.json");

  EXPECT_EQ(model.last_update.state, 4);
  EXPECT_EQ(model.last_update.action_param, 2);
  EXPECT_EQ(model.last_update.target, 7.5);
  EXPECT_EQ(model.learning_rate, 0.01);
  EXPECT_EQ(model.output_path, "output.json");
  EXPECT_EQ(model.load_path, "input.json");
}

TEST(ModelRef, ForwardsPolicyOperationsAndWeightImport) {
  PolicyModel destination;
  PolicyModel source;
  source.weight = 9;

  PolicyModelRef destination_ref(destination);
  PolicyModelRef source_ref(source);

  EXPECT_EQ(destination_ref.EvaluateAction(2).action_param, 2);
  const std::vector<int> states{1, 2};
  const std::vector<int> actions{0, 1};
  const std::vector<double> values{0.0, 0.0};
  const std::vector<double> targets{1.0, 1.0};
  destination_ref.LearnFromBatch(states, actions, values, targets, targets);
  EXPECT_EQ(destination.learned_states, std::vector<int>({1, 2}));

  destination_ref.ImportWeights(source);
  EXPECT_EQ(destination.weight, 9);

  source.weight = 12;
  destination_ref.ImportWeights(source_ref);
  EXPECT_EQ(destination.weight, 12);
}

} // namespace
