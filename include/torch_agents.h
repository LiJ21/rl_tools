#ifndef TORCH_AGENT_H
#define TORCH_AGENT_H
#include <models/off_policy_replay.h>
#include <models/torch/jit.h>
#include <models/torch/linear.h>

#include "agents/sarsa.h"

namespace RLlib {
template <typename TAction = double, typename TFeature = double,
          typename TReward = double>
using OffPolicyLinearSarsaAgent =
    SarsaAgent<Models::OffPolicyReplayLearner<Models::LinearQNetwork<TFeature>>,
               TAction, TReward>;

template <typename TAction = double, typename TFeature = double,
          typename TReward = double>
using OffPolicyJITNetworkSarsaAgent =
    SarsaAgent<Models::OffPolicyReplayLearner<Models::JITNetwork<TFeature>>,
               TAction, TReward>;
}  // namespace RLlib
#endif