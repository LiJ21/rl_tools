# RL Tools — A Lightweight Reinforcement Learning Framework in C++

## 🚀 Build Instructions

To compile the project, run the following commands in the **root project directory**:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate all executables, libraries, and test binaries inside `build/`.

---

## 📦 Overview

RL Tools is a modular reinforcement learning framework designed around **compile-time efficiency**, **template-based agents**, and **pluggable models**.

### Core design:

- Create an **Agent** (e.g., `SarsaAgent`, `QLearningAgent`, etc.)
- The agent is parameterized by a **Model`** that provides Q-value storage or approximation.
- During environment interaction:
  - Call `agent.UpdateState(state)` when a new state is observed  
    → returns an action selected with **epsilon-greedy**.
  - Call `agent.CollectReward(reward)` after receiving a reward  
    → triggers model updates and Q-learning logic.

The user does **not** manually compute TD targets or update Q-values—the agent and its model handle everything internally.

---

## 🧩 Example: 5×6 Grid World

Below is a simple example using a 5×6 grid world with 4 possible actions.

### Define grid dimensions:

```cpp
constexpr int nrows   = 5;
constexpr int ncols   = 6;
constexpr int nstates = nrows * ncols;
constexpr int nactions = 4;
```

### Create the agent type:

```cpp
using Direction = std::pair<int, int>;
using Agent = RLlib::TabularSarsaAgent<nstates, nactions, Direction>;
using ActionsList = Agent::ActionsList;
```

### Initialize the agent:

```cpp
Agent agent(
    ActionsList{
        Direction{1, 0},   // down
        Direction{0, 1},   // right
        Direction{-1, 0},  // up
        Direction{0, -1}   // left
    },
    config   // JSON configuration with parameters
);
```

`config` is a JSON object that may contain:

- learning rate  
- epsilon (for epsilon-greedy policy selection)  
- discount factor (gamma)  
- initialization options  
- model-specific parameters  

---

## 🎮 Interaction Loop

A typical RL loop with this agent looks like:

```cpp
int action = agent.UpdateState(state);   // observe state → get next action

// Take action in your environment...
// Observe reward and next state...

agent.CollectReward(reward);             // update agent based on (s, a, r, s')
```

Internally, the agent:

- computes TD targets  
- applies SARSA or Q-learning update rules  
- updates the model (tabular or neural network)  
- selects the next action using epsilon-greedy  

For episodic policy-gradient agents, close each path after collecting its final
reward:

```cpp
agent.TerminatePath();                  // genuine terminal: bootstrap value is zero
agent.TerminatePath(final_state);       // truncation: bootstrap from V(final_state)
agent.FlushBatch();                     // learn from a final partial PPO batch
```

Agents that do not implement path termination retain their existing behavior;
the base implementation is a no-op. PPO path termination records a GAE
boundary but does not force an update; completed paths accumulate until
`batch_steps` transitions have been collected. `FlushBatch` is only needed
when training ends with a partial batch.

PPO models define the probability-bearing action representation through
`Model::ActionParam`. A generic `PPOAgent` maps that parameter to the
environment action with an action mapper; `DiscretePPOAgent` is the convenience
wrapper where an integral parameter indexes a fixed action list. PPO batches
retain `ActionParam` values so the model can recompute their policy
log-probabilities during `LearnFromBatch`.

### Parallel PPO collection

`PPOAgent` and `DiscretePPOAgent` take a final `tAutoLearn` template
parameter, which defaults to `true`. An agent instantiated with `false` keeps
the normal `UpdateState`, `CollectReward`, and `TerminatePath` interaction but
only records raw rollout data:

```cpp
using Actor = RLlib::DiscretePPOAgent<Model, Action, Reward, false>;

Model learner_model(config["model"]);
RLlib::DistributedLearner learner(learner_model, config);

// Initially, and after every learner update:
actor.GetModel().ImportWeights(learner_model);

// Once collection has stopped at a path boundary:
learner.Submit(actor.ReleaseBuffer());
learner.Learn();
```

`Submit` is safe for concurrent thread submissions. `Learn` computes GAE
without crossing the submitted episode boundaries and normalizes advantages
over the combined batch. The application owns collection barriers and model
synchronization, so the same buffer API can be transported with threads,
OpenMP, or MPI.

---

## 📝 Notes

- The framework is fully **header only** and template-based, allowing compile-time optimization.
- Models can be:
  - **Tabular** (perfect for small state spaces)
  - **Torch-based function approximators** using the C++ LibTorch API
- The separation of **Agent** and **Model** makes it easy to extend with new architectures.

---

## 🧱 Using RL Tools as a Header-Only Library

Most of the core framework lives in the `include/` directory and is fully header-only.

- If you only need **tabular agents** or **non-Torch models**,  
  you may simply include the headers in your project and **ignore the `bin/` folder** entirely.

- When using **Torch-based models** (e.g., neural function approximators),  
  you must link your project with **LibTorch**.  
  The rest of the code remains header-only and requires no additional build steps.

Example CMake snippet for linking LibTorch:

```cmake
find_package(Torch REQUIRED)
target_link_libraries(your_target PRIVATE Torch::Torch)
```

This makes RL Tools easy to embed into any project:
- include only what you need,
- link LibTorch only if you use neural models,
- no need to build the entire framework unless you want the example binaries.

---
