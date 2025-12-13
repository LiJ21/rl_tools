#ifndef MODELS_TORCH_SCRIPT_H
#define MODELS_TORCH_SCRIPT_H

#include <agent.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace RLlib::Models {

template <typename TFeature = double, typename TResult = double>
class JITNetwork : public torch::nn::Module {
 public:
  using Feature = TFeature;
  using Result = TResult;
  using State = std::vector<Feature>;
  using ResultsList = std::vector<Result>;

  explicit JITNetwork(const json &config) {
    if (!config.contains("model_path") || !config["model_path"].is_string()) {
      throw std::runtime_error("model_path is mandatory in config!");
    }

    const std::string path = config["model_path"].get<std::string>();
    try {
      model_ = torch::jit::load(path);
      model_.to(torch::CppTypeToScalarType<Feature>::value);
      std::cout << "Loaded TorchScript Q-network from: " << path << std::endl;
    } catch (const c10::Error &e) {
      throw std::runtime_error("Failed to load TorchScript model \"" + path +
                               "\": " + e.what());
    }

    if (model_.hasattr("features_dim")) {
      features_dim = model_.attr("features_dim").toInt();
    } else {
      features_dim = config["features_dim"];
    }

    if (model_.hasattr("actions_dim")) {
      actions_dim = model_.attr("actions_dim").toInt();
    } else {
      actions_dim = config["actions_dim"];
    }
    results_.resize(ActionsDim(), {});
  }

  torch::Tensor forward(const torch::Tensor &X) {
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(X);
    auto out = model_.forward(inputs);
    return out.toTensor();
  }

  const ResultsList &GetActionValues(const State &state, bool semigrad = true) {
    const auto opts = torch::TensorOptions()
                          .dtype(torch::CppTypeToScalarType<Feature>::value)
                          .device(torch::kCPU);

    auto perform_forward = [&]() {
      auto input =
          torch::from_blob(const_cast<Feature *>(state.data()),
                           std::array<int64_t, 2>{1, FeaturesDim()}, opts);

      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(input);
      auto out = model_.forward(inputs).toTensor();

      auto q = out.squeeze(0);
      auto q_cpu = q.to(torch::kCPU);
      auto acc = q_cpu.template accessor<TResult, 1>();

      for (int i = 0; i < ActionsDim(); ++i) {
        results_[i] = acc[i];
      }
    };

    if (semigrad) {
      torch::NoGradGuard no_grad;
      perform_forward();
    } else {
      perform_forward();
    }

    return results_;
  }

  std::vector<torch::Tensor> parameters() {
    std::vector<torch::Tensor> out;
    for (const auto &p : model_.parameters()) {
      out.push_back(p);
    }
    return out;
  }

  void OutputModel(std::string_view fname, char /**/,
                   bool append = false) const {
    if (append) {
      std::cerr << "Warning: append mode for OutputModel() not supported for "
                   "TorchScript QNetwork.\n";
    }
    model_.save(std::string(fname));
  }

  void LoadModel(std::string_view fname, char /**/) {
    model_ = torch::jit::load(std::string(fname));
  }

  int ActionsDim() { return actions_dim; }
  int FeaturesDim() { return features_dim; }

 private:
  torch::jit::script::Module model_;
  ResultsList results_{};
  size_t actions_dim{};
  size_t features_dim{};
};

}  // namespace RLlib::Models

#endif
