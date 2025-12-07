#ifndef MODELS_LINEAR_H
#define MODELS_LINEAR_H
#include <array>
#include <fstream>
#include <iostream>
#include <string_view>

namespace RLlib::Models {
template <int tFeaturesDim, int tActionsDim, typename TFeature = double,
          typename TWeight = double, typename TResult = double>
class SimpleLinearModel {
 public:
  static constexpr int kFeaturesDim = tFeaturesDim;
  static constexpr int kActionsDim = tActionsDim;
  using Feature = TFeature;
  using Result = TResult;
  using State = std::array<Feature, kFeaturesDim>;
  using Weight = TWeight;
  using Weights = std::array<Weight, kFeaturesDim>;
  using WeightsList = std::array<Weights, kActionsDim>;
  using ResultsList = std::array<Result, kActionsDim>;

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

  void OutputModel(std::string_view fname, char delimiter = '\n',
                   bool append = false) const {
    std::ofstream ofs(fname.data(), append ? std::ios::app : std::ios::out);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open output file");
    }
    for (int i = 0; i < kActionsDim; ++i) {
      for (int j = 0; j < kFeaturesDim; ++j) {
        ofs << weights_[i][j];
        if (j != kFeaturesDim - 1) {
          ofs << ",";
        }
      }
      if (i != kActionsDim - 1) {
        ofs << delimiter;
      }
    }
    ofs << '\n';
  }

  void LoadModel(std::string_view fname, char delimiter = '\n') {
    std::ifstream ifs(fname.data());
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open input file");
    }
    for (int i = 0; i < kActionsDim; ++i) {
      for (int j = 0; j < kFeaturesDim; ++j) {
        ifs >> weights_[i][j];
        if (ifs.peek() == ',') {
          ifs.ignore();
        }
      }
      if (ifs.peek() == delimiter) {
        ifs.ignore();
      }
    }
  }

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
}  // namespace RLlib::Models
#endif