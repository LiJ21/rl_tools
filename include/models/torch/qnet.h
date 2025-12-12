#ifndef MODELS_TORCH_QNET_H
#define MODELS_TORCH_QNET_H

#include <agent.h>
#include <torch/torch.h>
#include <torch/script.h>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace RLlib::Models {

template <int tFeaturesDim, int tActionsDim,
          typename TFeature = double, typename TResult = double>
class QNetwork : public torch::nn::Module {
 public:
  static constexpr int kFeaturesDim = tFeaturesDim;
  static constexpr int kActionsDim  = tActionsDim;

  using Feature     = TFeature;
  using Result      = TResult;
  using State       = std::array<Feature, kFeaturesDim>;
  using ResultsList = std::array<Result, kActionsDim>;

  QNetwork() { 
    throw std::runtime_error(
      "QNetwork() default constructor is not allowed. "
      "Use QNetwork(json) with model_path.");
  }

  explicit QNetwork(const json &config) {
    if (!config.contains("model_path") || !config["model_path"].is_string()) {
      throw std::runtime_error(
        "QNetwork requires config[\"model_path\"] to load a TorchScript model.");
    }

    const std::string path = config["model_path"].get<std::string>();
    try {
      model_ = torch::jit::load(path);
      model_.to(torch::kFloat64);
      std::cout << "Loaded TorchScript Q-network from: " << path << std::endl;
    } catch (const c10::Error &e) {
      throw std::runtime_error(
        "Failed to load TorchScript model \"" + path + "\": " + e.what());
    }

    results_.fill(Result{0});
  }


  torch::Tensor forward(const torch::Tensor &X) override {
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(X);
    auto out = model_.forward(inputs);
    return out.toTensor();
  }

  const ResultsList &GetActionValues(const State &state, bool semigrad = true) {
    const auto opts =
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto perform_forward = [&]() {
      auto input = torch::from_blob(
          const_cast<Feature *>(state.data()),
          std::array<int64_t, 2>{1, kFeaturesDim},
          opts);

      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(input);
      auto out = model_.forward(inputs).toTensor();    // [1, kActionsDim]

      auto q = out.squeeze(0);                         // [kActionsDim]
      auto q_cpu = q.to(torch::kCPU);
      auto acc = q_cpu.template accessor<TResult, 1>();

      for (int i = 0; i < kActionsDim; ++i) {
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

  void OutputModel(std::string_view fname,
                   char delimiter = '\n', bool append = false) const {
    std::cerr << "Warning: OutputModel() not supported for TorchScript QNetwork.\n";
    (void)fname; (void)delimiter; (void)append;
  }

  void LoadModel(std::string_view fname, char delimiter = '\n') {
    std::cerr << "Warning: LoadModel(CSV) not supported for TorchScript QNetwork.\n";
    (void)fname; (void)delimiter;
  }

  static constexpr int ActionsDim()  { return kActionsDim; }
  static constexpr int FeaturesDim() { return kFeaturesDim; }

  using ModuleType = QNetwork;

 private:
  torch::jit::script::Module model_;   // Always used
  ResultsList results_{};
};

}  // namespace RLlib::Models

#endif
