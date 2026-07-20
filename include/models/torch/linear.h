#ifndef MODELS_TORCH_LINEAR_H
#define MODELS_TORCH_LINEAR_H

#include <agent.h>
#include <torch/torch.h>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace RLlib::Models {

template <typename TFeature = double, typename TResult = double>
class LinearQNetwork : public torch::nn::Module {
 public:

  using Feature = TFeature;
  using Result = TResult;
  using State = std::vector<Feature>;
  using ActionParam = int;
  using ResultsList = std::vector<Result>;

  explicit LinearQNetwork(const json &config) {
    auto features_dim = config["features_dim"];
    auto actions_dim = config["actions_dim"];

    linear_ = register_module(
        "linear",
        torch::nn::Linear(
            torch::nn::LinearOptions(features_dim, actions_dim).bias(false)));
    results_.resize(actions_dim, {});
    linear_->to(torch::CppTypeToScalarType<Feature>::value);
    if (!config.contains("weights")) {
      throw std::runtime_error("Missing 'weights' in config JSON");
    }

    const auto &w_cfg = config["weights"];
    InitializeWeights(w_cfg);
  }

  torch::Tensor forward(const torch::Tensor &X) { return linear_->forward(X); }

  const ResultsList &GetActionValues(const State &state, bool semigrad = true) {
    assert(state.size() == FeaturesDim());
    auto dtype = torch::CppTypeToScalarType<TFeature>::value;
    const auto opts = torch::TensorOptions().dtype(
        torch::CppTypeToScalarType<Feature>::value);

#ifdef DEBUG
    const auto W = linear_->weight.detach().to(torch::kCPU);
    const auto acc = W.accessor<double, 2>();
    for (int i = 0; i < ActionsDim(); ++i) {
      for (int j = 0; j < FeaturesDim(); ++j) {
        std::cout << i << "," << j << "," << acc[i][j] << std::endl;
      }
    }
#endif

    auto eval_forward = [&]() {
      auto input = torch::from_blob(
          const_cast<Feature *>(state.data()),
          std::array<int64_t, 2>{1, static_cast<long long>(state.size())},
          opts);

      auto output = linear_->forward(input);
      auto q = output.squeeze(0);

      auto q_cpu = q.to(torch::kCPU);
      auto q_acc = q_cpu.template accessor<Result, 1>();

      for (int i = 0; i < results_.size(); ++i) {
        results_[i] = static_cast<Result>(q_acc[i]);
      }
    };

    if (semigrad) {
      torch::NoGradGuard no_grad;
      eval_forward();
    } else {
      eval_forward();
    }

    return results_;
  }

  void OutputModel(std::string_view fname, char delimiter = '\n',
                   bool append = false) const {
    std::ofstream ofs(std::string(fname),
                      append ? std::ios::app : std::ios::out);
    if (!ofs) {
      throw std::runtime_error("Failed to open output file: " +
                               std::string(fname));
    }

    const auto W = linear_->weight.detach().to(torch::kCPU);
    const auto acc = W.accessor<double, 2>();

    for (int i = 0; i < ActionsDim(); ++i) {
      for (int j = 0; j < FeaturesDim(); ++j) {
        ofs << acc[i][j];
        if (j != FeaturesDim() - 1) {
          ofs << ",";
        }
      }
      if (i != ActionsDim() - 1) {
        ofs << delimiter;
      }
    }
    ofs << '\n';

    if (!ofs) {
      throw std::runtime_error("Error writing to file: " + std::string(fname));
    }
  }

  void LoadModel(std::string_view fname, char delimiter = '\n') {
    std::ifstream ifs{std::string(fname)};
    if (!ifs) {
      throw std::runtime_error("Failed to open input file: " +
                               std::string(fname));
    }

    auto &W = linear_->weight;

    for (int i = 0; i < ActionsDim(); ++i) {
      for (int j = 0; j < FeaturesDim(); ++j) {
        double v;
        if (!(ifs >> v)) {
          throw std::runtime_error("Failed to read weight at position [" +
                                   std::to_string(i) + "," + std::to_string(j) +
                                   "]");
        }
        W.index_put_({i, j}, v);

        if (j != FeaturesDim() - 1) {
          char sep;
          if (ifs >> sep && sep != ',') {
            throw std::runtime_error("Expected ',' separator at position [" +
                                     std::to_string(i) + "," +
                                     std::to_string(j) + "]");
          }
        }
      }

      if (i != ActionsDim() - 1 && delimiter != '\n') {
        char sep;
        if (ifs >> sep && sep != delimiter) {
          throw std::runtime_error("Expected delimiter after row " +
                                   std::to_string(i));
        }
      }
    }
  }

  int ActionsDim() const { return linear_->weight.size(0); }
  int FeaturesDim() const { return linear_->weight.size(1); }

  using ModuleType = LinearQNetwork;

 private:
  void InitializeWeights(const json &w_cfg) {
    const auto opts = torch::TensorOptions().dtype(
        torch::CppTypeToScalarType<Feature>::value);
    auto &W = linear_->weight;

    if (w_cfg.is_array()) {
      if (w_cfg.size() != ActionsDim()) {
        throw std::runtime_error(
            "weights array size " + std::to_string(w_cfg.size()) +
            " does not match ActionsDim " + std::to_string(ActionsDim()));
      }

      for (int i = 0; i < ActionsDim(); ++i) {
        if (!w_cfg[i].is_array() || w_cfg[i].size() != FeaturesDim()) {
          throw std::runtime_error("weights[" + std::to_string(i) + "] size " +
                                   std::to_string(w_cfg[i].size()) +
                                   " does not match FeaturesDim " +
                                   std::to_string(FeaturesDim()));
        }
        for (int j = 0; j < FeaturesDim(); ++j) {
          if (!w_cfg[i][j].is_number()) {
            throw std::runtime_error("weights[" + std::to_string(i) + "][" +
                                     std::to_string(j) + "] is not a number");
          }
          double v = w_cfg[i][j].get<double>();
          W.index_put_({i, j}, v);
        }
      }
    } else if (w_cfg.is_number()) {
      double init_weight = w_cfg.get<double>();
      W.data().fill_(init_weight);
    } else if (w_cfg.is_object()) {
      if (w_cfg.contains("mean") && w_cfg.contains("stddev")) {
        if (!w_cfg["mean"].is_number() || !w_cfg["stddev"].is_number()) {
          throw std::runtime_error("weights mean and stddev must be numbers");
        }
        double mean = w_cfg["mean"].get<double>();
        double stddev = w_cfg["stddev"].get<double>();
        if (stddev < 0.0) {
          throw std::runtime_error("weights stddev must be non-negative");
        }
        auto randW = torch::randn({ActionsDim(), FeaturesDim()}, opts);
        W.data().copy_(randW * stddev + mean);
      } else {
        throw std::runtime_error(
            "weights object must contain 'mean' and 'stddev' fields");
      }
    } else {
      throw std::runtime_error(
          "weights must be an array, number, or object with mean/stddev");
    }
  }

  torch::nn::Linear linear_{nullptr};
  ResultsList results_{};
  bool debug_output_{};
};

}  // namespace RLlib::Models

#endif
