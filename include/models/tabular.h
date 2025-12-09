#ifndef MODELS_TABULAR_H
#define MODELS_TABULAR_H
#include <agent.h>

#include <array>
#include <fstream>
#include <string_view>

#include "random_generator.h"

namespace RLlib::Models {
template <int tStatesDim, int tActionsDim>
class Tabular {
 public:
  static constexpr int kActionsDim = tActionsDim;
  static constexpr int kStatesDim = tStatesDim;
  using State = int;
  using Action = int;
  using Reward = double;
  using ResultsList = std::array<double, kActionsDim>;
  using QType = std::array<ResultsList, kStatesDim>;

  Tabular() = default;
  Tabular(const json &config) {
    if (config["action_values"].is_array() &&
        config["action_values"].size() == kStatesDim) {
      for (int s = 0; s < kStatesDim; ++s) {
        if (config["action_values"][s].is_array() &&
            config["action_values"][s].size() == kActionsDim) {
          for (int a = 0; a < kActionsDim; ++a) {
            action_values_[s][a] = config["action_values"][s][a].get<double>();
          }
        } else {
          throw std::runtime_error(
              "Invalid action_values format in config JSON");
        }
      }
    } else if (config["action_values"].is_number()) {
      double init_value = config["action_values"].get<double>();
      for (auto &values : action_values_) {
        values.fill(init_value);
      }
    } else if (config["action_values"].is_object()) {
      if (config["action_values"].contains("mean") &&
          config["action_values"].contains("stddev")) {
        double mean = config["weights"]["mean"].get<double>();
        double stddev = config["weights"]["stddev"].get<double>();
        for (auto &values : action_values_) {
          for (auto &val : values) {
            val = static_cast<double>(rng_util::normal(mean, stddev));
          }
        }
      } else {
        throw std::runtime_error("Invalid weights format in config JSON");
      }
    } else {
      throw std::runtime_error("Invalid action_values format in config JSON");
    }

    if (config.contains("learning_rate")) {
      if (!config["learning_rate"].is_number()) {
        alpha_ = config["learning_rate"].get<double>();
      } else {
        throw std::runtime_error("Invalid learning_rate format in config JSON");
      }
    }
  }

  const ResultsList &GetActionValues(State state) {
    return action_values_[state];
  }

  void Update(State state, int action_idx, double last_action_value,
              double td_target) {
    double error_ = td_target - last_action_value;
    action_values_[state][action_idx] += alpha_ * error_;
  }

  void SetLearningRate(double alpha) { alpha_ = alpha; }

  const QType &GetActionValues() const { return action_values_; }

  void OutputModel(std::string_view fname, char delimiter = '\n',
                   bool append = false) const {
    std::ofstream ofs(fname.data(), append ? std::ios::app : std::ios::out);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open output file");
    }
    for (int s = 0; s < kStatesDim; ++s) {
      for (int a = 0; a < kActionsDim; ++a) {
        ofs << action_values_[s][a];
        if (a != kActionsDim - 1) {
          ofs << ",";
        }
      }
      if (s != kStatesDim - 1) {
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
    for (int s = 0; s < kStatesDim; ++s) {
      for (int a = 0; a < kActionsDim; ++a) {
        ifs >> action_values_[s][a];
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
  double alpha_{1.0};
  QType action_values_{};
};
}  // namespace RLlib::Models
#endif