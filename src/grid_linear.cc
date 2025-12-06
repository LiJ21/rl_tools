#include <agents/sarsa.h>

#include <cassert>
#include <fstream>
#include <iostream>

constexpr int nrows = 5;
constexpr int ncols = 6;
constexpr int nstates = nrows * ncols;
constexpr int nstate_dim = 2;
constexpr int nactions = 4;

using Direction = std::array<int, 2>;
using Agent = RLlib::LinearSarsaAgent<nstate_dim, nactions, Direction, int>;
using State = typename Agent::State;
using ActionsList = Agent::ActionsList;

int main(int argc, char **argv) {
  assert(argc > 3);
  std::string fname = argv[1];
  int Nstep = std::stoi(argv[2]);
  int train_step = std::stoi(argv[3]);
  double epsilon = std::stof(argv[4]);
  Agent agent(ActionsList{Direction{1, 0}, Direction{0, 1}, Direction{-1, 0},
                          Direction{0, -1}},
              epsilon, 1.0, 0.0);
  agent.SetSteps(train_step);

  std::array<double, nstates> state_values{};
  {
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open input file: " << fname << std::endl;
      return 2;
    }
    for (int i = 0; i < nstates; ++i) {
      ifs >> state_values[i];
      if (ifs.fail()) {
        std::cerr << "Failed to read state value " << i << " from " << fname
                  << std::endl;
        return 3;
      }
    }
  }

  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      std::cout << state_values[i * ncols + j] << "\t";
    }
    std::cout << std::endl;
  }

  auto loc = [](int x, int y) { return x * ncols + y; };

  // auto coords = [](int state) {
  //   return std::make_pair(state / ncols, state % ncols);
  // };

  auto state = State{0, 0};

  std::vector<double> rewards(Nstep, 0.0);
  std::vector<State> states(Nstep, {{}});
  agent.SetLearningRate(0.1);
  for (int step = 0; step < Nstep; ++step) {
    auto action = agent.UpdateState(state);

    for (int i = 0; i < nstate_dim; ++i) {
      state[i] = (state[i] + action[i] + (i == 0 ? nrows : ncols)) %
                 (i == 0 ? nrows : ncols);
    }

    // agent.SetLearningRate(0.1 / (step + 1));
    auto reward = state_values[loc(state[0], state[1])];
    rewards[step] = reward;
    states[step] = state;
    agent.CollectReward(reward);
  }

  std::cout << "Finished " << Nstep << " steps." << std::endl;

  {
    std::cout << "Writing rewards to files..." << std::endl;
    std::ofstream ofs("./rewards.txt");
    if (!ofs.is_open()) {
      std::cerr << "Failed to open ./rewards.txt for writing" << std::endl;
      return 4;
    }
    for (const auto &r : rewards) {
      ofs << r << std::endl;
    }
    ofs.close();
  }
  {
    std::cout << "Writing states to files..." << std::endl;
    std::ofstream ofs("./states.txt");
    if (!ofs.is_open()) {
      std::cerr << "Failed to open ./states.txt for writing" << std::endl;
      return 5;
    }
    for (const auto &r : states) {
      ofs << r[0] << "," << r[1] << std::endl;
    }
    ofs.close();
  }
  agent.GetModel().OutputModel("./trained_model.txt");

  return 0;
}