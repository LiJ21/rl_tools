#ifndef MODELS_LINEAR_H
#define MODELS_LINEAR_H
#include <array>

namespace RLLib {
template <int tFeaturesDim, int tActionsDim>
class SimpleLinearModel {
 public:
  static constexpr int kFeaturesDim = tFeaturesDim;
  static constexpr int kActionsDim = tActionsDim;
  using State = std::array<double, kFeaturesDim>;
  using WeightsList = std::array<State, kActionsDim>;
  using ResultsList = std::array<double, kActionsDim>;

  SimpleLinearModel(double init_weight = 0.0) {
    for (auto &weights : weights_) {
      weights.fill(init_weight);
    }
  }

  SimpleLinearModel(const WeightsList &init_weights) : weights_(init_weights) {}

  template <typename... TWeights>
  SimpleLinearModel(TWeights &&...init_weights) {
    FillWeights(std::forward<TWeights>(init_weights)...);
  }

  const ResultsList &GetActionValues(const State &state) {
    for (int i = 0; i < kActionsDim; ++i) {
      double value_ = 0.0;
      for (int j = 0; j < kFeaturesDim; ++j) {
        value_ += weights_[i][j] * state[j];
      }
      results_[i] = value_;
    }
    return results_;
  }

  void Update(const State &state, int action_idx, double last_action_value,
              double td_target) {
    double error_ = td_target - last_action_value;
    for (int j = 0; j < kFeaturesDim; ++j) {
      weights_[action_idx][j] += alpha_ * error_ * state[j];
    }
  }

  void SetLearningRate(double alpha) { alpha_ = alpha; }

 private:
  WeightsList weights_{};
  ResultsList results_{};
  double alpha_{1.0};

  template <typename TFirst, typename... TRest>
  void FillWeights(TFirst &&first, TRest &&...rest) {
    static_assert(sizeof...(rest) + 1 <= kActionsDim,
                  "Too many weight sets provided");
    weights_[kActionsDim - sizeof...(rest) - 1] = std::forward<TFirst>(first);
    if constexpr (sizeof...(rest) > 0)
      FillWeights(std::forward<TRest>(rest)...);
  }
  
};
}  // namespace RLLib
#endif