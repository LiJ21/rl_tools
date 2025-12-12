#ifndef TORCH_AGENT_H
#define TORCH_AGENT_H
#include <models/off_policy_replay.h>
#include <models/torch/linear.h>
#include <models/torch/jit.h>

#include "agents/sarsa.h"

namespace RLlib {
template <int tFeaturesDim, int tActionsDim, typename TAction = double,
          typename TFeature = double, typename TReward = double>
using OffPolicyLinearSarsaAgent =
    SarsaAgent<Models::OffPolicyReplayLearner<
                   Models::LinearQNetwork<tFeaturesDim, tActionsDim, TFeature>>,
               TAction, TReward>;

template <int tFeaturesDim, int tActionsDim, typename TAction = double,
          typename TFeature = double, typename TReward = double>
using OffPolicyQNetSarsaAgent =
    SarsaAgent<Models::OffPolicyReplayLearner<
                   Models::JITNetwork<tFeaturesDim, tActionsDim, TFeature>>,
               TAction, TReward>;
}  // namespace RLlib
#endif