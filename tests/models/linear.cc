#include <gtest/gtest.h>
#include <models/linear.h>

using LinearModel = RLlib::Models::SimpleLinearModel<4, 2>;
using WeightsList = LinearModel::WeightsList;
using State = LinearModel::State;

TEST(LinearModel, Forward) {
  json simple_config = {{"weights", 1.0}};
  LinearModel model(simple_config);
  EXPECT_DOUBLE_EQ(model.GetActionValues({1, 2, 3, 4})[0], 10.0);

  json config = {
      {"weights", json{
          json{1.0, 2.0, 3.0, 4.0},
          json{1.0, -2.0, -3.0, -4.0}
      }}
  };
  LinearModel model2(config);
  auto results = model2.GetActionValues({1, 2, 3, 1});
  EXPECT_DOUBLE_EQ(results[0], 18.0);
  EXPECT_DOUBLE_EQ(results[1], -16.0);
}

TEST(LinearModel, Backward) {
  json config = {
      {"weights", json{
          json{1.0, 2.0, 3.0, 4.0},
          json{1.0, -2.0, -3.0, -4.0}
      }}
  };
  LinearModel model(config);
  model.SetLearningRate(1.0);
  model.Update({1, 1, 2, 1}, 0, 0.0, 1.0);
  // After update, weights for action 0 should be: (2, 3, 5, 5)
  auto results = model.GetActionValues({1, 1, 2, 1});
  EXPECT_DOUBLE_EQ(results[0], 2 * 1 + 3 * 1 + 5 * 2 + 5 * 1);
  EXPECT_DOUBLE_EQ(results[1], 1 * 1 - 2 * 1 - 3 * 2 - 4 * 1);
}