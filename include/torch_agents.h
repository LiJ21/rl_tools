#ifndef TORCH_AGENT_H
#define TORCH_AGENT_H
#include <models/off_policy_replay.h>
#include <models/ppo_learner.h>
#include <models/torch/actor_critic.h>
#include <models/torch/jit.h>
#include <models/torch/linear.h>

#include "agents/ppo.h"
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

template <typename TAction = double, typename TFeature = double,
          typename TReward = double>
using DiscretePPOActorCriticAgent =
    DiscretePPOAgent<
        Models::PPOLearner<Models::ActorCriticNetwork<TFeature>>, TAction,
        TReward>;

template <typename TAction = double, typename TFeature = double,
          typename TReward = double>
using PPOActorCriticAgent =
    DiscretePPOActorCriticAgent<TAction, TFeature, TReward>;
}  // namespace RLlib
#endif
