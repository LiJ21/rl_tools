#ifndef LINEAR_AGENT_H
#define LINEAR_AGENT_H
#include <models/linear.h>

#include "agents/sarsa.h"

namespace RLlib {
template <int tFeaturesDim, int tActionsDim, typename TAction = double,
          typename TFeature = double, typename TReward = double>
using LinearSarsaAgent =
    SarsaAgent<Models::SimpleLinearModel<tFeaturesDim, tActionsDim, TFeature>,
               TAction, TReward>;

} // RLlib
#endif