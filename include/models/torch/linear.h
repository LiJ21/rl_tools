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

template <int tFeaturesDim, int tActionsDim, typename TFeature = double,
          typename TResult = double>
class LinearQNetwork : public torch::nn::Module {
 public:
  static constexpr int kFeaturesDim = tFeaturesDim;
  static constexpr int kActionsDim = tActionsDim;

  using Feature = TFeature;
  using Result = TResult;
  using State = std::array<Feature, kFeaturesDim>;
  using ResultsList = std::array<Result, kActionsDim>;

  LinearQNetwork() : linear_(nullptr) {
    static_assert(kFeaturesDim > 0, "kFeaturesDim must be > 0");
    static_assert(kActionsDim > 0, "kActionsDim must be > 0");

    linear_ = register_module(
        "linear",
        torch::nn::Linear(
            torch::nn::LinearOptions(kFeaturesDim, kActionsDim).bias(false)));

    linear_->to(torch::CppTypeToScalarType<Feature>::value);
    results_.fill(Result{0});
  }

  explicit LinearQNetwork(const json &config) : LinearQNetwork() {
    if (!config.contains("weights")) {
      throw std::runtime_error("Missing 'weights' in config JSON");
    }

    const auto &w_cfg = config["weights"];
    InitializeWeights(w_cfg);
  }

  torch::Tensor forward(const torch::Tensor &X) { return linear_->forward(X); }

  const ResultsList &GetActionValues(const State &state, bool semigrad = true) {
    auto dtype = torch::CppTypeToScalarType<TFeature>::value;
    const auto opts = torch::TensorOptions().dtype(
        torch::CppTypeToScalarType<Feature>::value);

#ifdef DEBUG
    const auto W = linear_->weight.detach().to(torch::kCPU);
    const auto acc = W.accessor<double, 2>();
    for (int i = 0; i < kActionsDim; ++i) {
      for (int j = 0; j < kFeaturesDim; ++j) {
        std::cout << i << "," << j << "," << acc[i][j] << std::endl;
      }
    }
#endif

    auto eval_forward = [&]() {
      auto input =
          torch::from_blob(const_cast<Feature *>(state.data()),
                           std::array<int64_t, 2>{1, kFeaturesDim}, opts);

      auto output = linear_->forward(input);
      auto q = output.squeeze(0);

      auto q_cpu = q.to(torch::kCPU);
      auto q_acc = q_cpu.template accessor<Result, 1>();

      for (int i = 0; i < kActionsDim; ++i) {
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

    for (int i = 0; i < kActionsDim; ++i) {
      for (int j = 0; j < kFeaturesDim; ++j) {
        ofs << acc[i][j];
        if (j != kFeaturesDim - 1) {
          ofs << ",";
        }
      }
      if (i != kActionsDim - 1) {
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

    for (int i = 0; i < kActionsDim; ++i) {
      for (int j = 0; j < kFeaturesDim; ++j) {
        double v;
        if (!(ifs >> v)) {
          throw std::runtime_error("Failed to read weight at position [" +
                                   std::to_string(i) + "," + std::to_string(j) +
                                   "]");
        }
        W.index_put_({i, j}, v);

        if (j != kFeaturesDim - 1) {
          char sep;
          if (ifs >> sep && sep != ',') {
            throw std::runtime_error("Expected ',' separator at position [" +
                                     std::to_string(i) + "," +
                                     std::to_string(j) + "]");
          }
        }
      }

      if (i != kActionsDim - 1 && delimiter != '\n') {
        char sep;
        if (ifs >> sep && sep != delimiter) {
          throw std::runtime_error("Expected delimiter after row " +
                                   std::to_string(i));
        }
      }
    }
  }

  static constexpr int ActionsDim() { return kActionsDim; }
  static constexpr int FeaturesDim() { return kFeaturesDim; }

  using ModuleType = LinearQNetwork;

 private:
  void InitializeWeights(const json &w_cfg) {
    const auto opts = torch::TensorOptions().dtype(
        torch::CppTypeToScalarType<Feature>::value);
    auto &W = linear_->weight;

    if (w_cfg.is_array()) {
      if (w_cfg.size() != kActionsDim) {
        throw std::runtime_error(
            "weights array size " + std::to_string(w_cfg.size()) +
            " does not match kActionsDim " + std::to_string(kActionsDim));
      }

      for (int i = 0; i < kActionsDim; ++i) {
        if (!w_cfg[i].is_array() || w_cfg[i].size() != kFeaturesDim) {
          throw std::runtime_error("weights[" + std::to_string(i) + "] size " +
                                   std::to_string(w_cfg[i].size()) +
                                   " does not match kFeaturesDim " +
                                   std::to_string(kFeaturesDim));
        }
        for (int j = 0; j < kFeaturesDim; ++j) {
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
        auto randW = torch::randn({kActionsDim, kFeaturesDim}, opts);
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