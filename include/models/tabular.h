#ifndef MODELS_TABULAR_H
#define MODELS_TABULAR_H
#include <array>
#include <fstream>
#include <string_view>

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

  Tabular(double init_value = 0.0) {
    for (auto &values : action_values_) {
      values.fill(init_value);
    }
  }

  Tabular(const ResultsList &init_values) : action_values_(init_values) {}

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

  void OutputModel(std::string_view fname) const {
    std::ofstream ofs(fname.data());
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open output file");
      ;
    }
    for (int s = 0; s < kStatesDim; ++s) {
      for (int a = 0; a < kActionsDim; ++a) {
        ofs << action_values_[s][a] << ",";
      }
      ofs << std::endl;
    }
  }

 private:
  double alpha_{1.0};
  QType action_values_{};
};
}  // namespace RLlib::Models
#endif