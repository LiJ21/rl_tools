#ifndef MODELS_LINEAR_H
#define MODELS_LINEAR_H
#include <agent.h>
#include <random_generator.h>

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

  SimpleLinearModel() = default;
  SimpleLinearModel(const json &config) {
    if (config["weights"].is_array() &&
        config["weights"].size() == kActionsDim) {
      for (int i = 0; i < kActionsDim; ++i) {
        if (config["weights"][i].is_array() &&
            config["weights"][i].size() == kFeaturesDim) {
          for (int j = 0; j < kFeaturesDim; ++j) {
            weights_[i][j] = config["weights"][i][j].get<Weight>();
          }
        } else {
          throw std::runtime_error("Invalid weights format in config JSON");
        }
      }
    } else if (config["weights"].is_number()) {
      Weight init_weight = config["weights"].get<Weight>();
      for (auto &weights : weights_) {
        weights.fill(init_weight);
      }
    } else if (config["weights"].is_object()) {
      if (config["weights"].contains("mean") &&
          config["weights"].contains("stddev")) {
        Weight mean = config["weights"]["mean"].get<Weight>();
        Weight stddev = config["weights"]["stddev"].get<Weight>();
        for (auto &weights : weights_) {
          for (auto &w : weights) {
            w = static_cast<Weight>(rng_util::normal(mean, stddev));
          }
        }
      } else {
        throw std::runtime_error("Invalid weights format in config JSON");
      }
    } else {
      throw std::runtime_error("Invalid weights format in config JSON");
    }

    if (config.contains("learning_rate")) {
      if (!config["learning_rate"].is_number()) {
        alpha_ = config["learning_rate"].get<double>();
      } else {
        throw std::runtime_error("Invalid learning_rate format in config JSON");
      }
    }
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
};
}  // namespace RLlib::Models
#endif