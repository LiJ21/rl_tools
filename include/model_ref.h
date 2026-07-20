#ifndef MODEL_REF_H
#define MODEL_REF_H

#include <functional>
#include <type_traits>
#include <utility>

namespace RLlib {

namespace detail {

template <typename TModel, typename = void> struct ModelRefResultsList {};

template <typename TModel>
struct ModelRefResultsList<TModel, std::void_t<typename TModel::ResultsList>> {
  using ResultsList = typename TModel::ResultsList;
};

template <typename TModel, typename = void> struct ModelRefDecision {};

template <typename TModel>
struct ModelRefDecision<TModel, std::void_t<typename TModel::Decision>> {
  using Decision = typename TModel::Decision;
};

} // namespace detail

// A non-owning model adapter. It lets an agent keep its usual value-like Model
// member while the actual model lifetime and ownership remain external.
template <typename TModel>
class ModelRef : public detail::ModelRefResultsList<TModel>,
                 public detail::ModelRefDecision<TModel> {
public:
  using Model = TModel;
  using State = typename Model::State;
  using ActionParam = typename Model::ActionParam;

  explicit ModelRef(Model &model) noexcept : model_(model) {}
  ModelRef(Model &&) = delete;

  Model &Get() noexcept { return model_.get(); }
  const Model &Get() const noexcept { return model_.get(); }

  template <typename... TArgs>
  decltype(auto) GetActionValues(TArgs &&...args)
    requires requires(Model &model) {
      model.GetActionValues(std::forward<TArgs>(args)...);
    }
  {
    return Get().GetActionValues(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) Update(TArgs &&...args)
    requires requires(Model &model) {
      model.Update(std::forward<TArgs>(args)...);
    }
  {
    return Get().Update(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) EvaluateAction(TArgs &&...args)
    requires requires(Model &model) {
      model.EvaluateAction(std::forward<TArgs>(args)...);
    }
  {
    return Get().EvaluateAction(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) EvaluateValue(TArgs &&...args)
    requires requires(Model &model) {
      model.EvaluateValue(std::forward<TArgs>(args)...);
    }
  {
    return Get().EvaluateValue(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) LearnFromBatch(TArgs &&...args)
    requires requires(Model &model) {
      model.LearnFromBatch(std::forward<TArgs>(args)...);
    }
  {
    return Get().LearnFromBatch(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) SetLearningRate(TArgs &&...args)
    requires requires(Model &model) {
      model.SetLearningRate(std::forward<TArgs>(args)...);
    }
  {
    return Get().SetLearningRate(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) OutputModel(TArgs &&...args)
    requires requires(Model &model) {
      model.OutputModel(std::forward<TArgs>(args)...);
    }
  {
    return Get().OutputModel(std::forward<TArgs>(args)...);
  }

  template <typename... TArgs>
  decltype(auto) LoadModel(TArgs &&...args)
    requires requires(Model &model) {
      model.LoadModel(std::forward<TArgs>(args)...);
    }
  {
    return Get().LoadModel(std::forward<TArgs>(args)...);
  }

  void ImportWeights(const Model &source)
    requires requires(Model &destination, const Model &source_model) {
      destination.ImportWeights(source_model);
    }
  {
    Get().ImportWeights(source);
  }

  void ImportWeights(const ModelRef &source)
    requires requires(Model &destination, const Model &source_model) {
      destination.ImportWeights(source_model);
    }
  {
    Get().ImportWeights(source.Get());
  }

private:
  std::reference_wrapper<Model> model_;
};

} // namespace RLlib

#endif // MODEL_REF_H
