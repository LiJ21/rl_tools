#ifndef TABULAR_AGENT_H
#define TABULAR_AGENT_H
#include <models/tabular.h>

#include "agents/sarsa.h"

namespace RLlib {
template <int tStatesDim, int tActionsDim, typename TAction = int,
          typename TReward = double>
using TabularSarsaAgent =
    SarsaAgent<Models::Tabular<tStatesDim, tActionsDim>, TAction, TReward>;
}  // namespace RLlib
#endif