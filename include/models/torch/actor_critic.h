#ifndef MODELS_TORCH_ACTOR_CRITIC_H
#define MODELS_TORCH_ACTOR_CRITIC_H

#include <agent.h>
#include <torch/torch.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace RLlib::Models {

// Discrete-action actor-critic network for policy-gradient agents (e.g. PPO).
//
// Mirrors the config/style of LinearQNetwork: it reads `features_dim` and
// `actions_dim`, is templated on the feature/result scalar type, and exposes
// OutputModel/LoadModel. `forward` returns {policy_logits [B, A], value [B]}.
// An optional shared trunk is added when `hidden_dim` > 0 (tanh activation);
// otherwise the policy/value heads read the raw features directly.
template <typename TFeature = double, typename TResult = double>
class ActorCriticNetwork : public torch::nn::Module {
public:
  using Feature = TFeature;
  using Result = TResult;
  using State = std::vector<Feature>;

  explicit ActorCriticNetwork(const json &config) {
    features_dim_ = config.at("features_dim").get<int>();
    actions_dim_ = config.at("actions_dim").get<int>();
    hidden_dim_ = config.value("hidden_dim", 0);

    int head_in = features_dim_;
    if (hidden_dim_ > 0) {
      trunk_ =
          register_module("trunk", torch::nn::Linear(torch::nn::LinearOptions(
                                       features_dim_, hidden_dim_)));
      head_in = hidden_dim_;
    }
    policy_head_ = register_module(
        "policy",
        torch::nn::Linear(torch::nn::LinearOptions(head_in, actions_dim_)));
    value_head_ = register_module(
        "value", torch::nn::Linear(torch::nn::LinearOptions(head_in, 1)));

    this->to(torch::CppTypeToScalarType<Feature>::value);

    // Small-gain policy head keeps the initial policy close to uniform, which
    // stabilises the first PPO updates.
    torch::NoGradGuard no_grad;
    policy_head_->weight.mul_(0.01);
  }

  // Returns {logits [B, A], value [B]}.
  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor &X) {
    torch::Tensor h = X;
    if (trunk_) {
      h = torch::tanh(trunk_->forward(X));
    }
    auto logits = policy_head_->forward(h);
    auto value = value_head_->forward(h).squeeze(-1);
    return {logits, value};
  }

  std::vector<torch::Tensor> parameters() {
    return torch::nn::Module::parameters();
  }

  void OutputModel(std::string_view fname, char /*delimiter*/ = '\n',
                   bool append = false) const {
    if (append) {
      std::cerr << "Warning: append mode for OutputModel() not supported for "
                   "ActorCriticNetwork.\n";
    }
    torch::serialize::OutputArchive archive;
    const_cast<ActorCriticNetwork *>(this)->save(archive);
    archive.save_to(std::string(fname));
  }

  void LoadModel(std::string_view fname, char /*delimiter*/ = '\n') {
    torch::serialize::InputArchive archive;
    archive.load_from(std::string(fname));
    this->load(archive);
  }

  int ActionsDim() const { return actions_dim_; }
  int FeaturesDim() const { return features_dim_; }

private:
  torch::nn::Linear trunk_{nullptr};
  torch::nn::Linear policy_head_{nullptr};
  torch::nn::Linear value_head_{nullptr};
  int features_dim_{};
  int actions_dim_{};
  int hidden_dim_{};
};

} // namespace RLlib::Models

#endif
