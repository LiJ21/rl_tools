#include <agents/sarsa.h>

#include <cassert>
#include <fstream>
#include <iostream>

constexpr int nrows = 5;
constexpr int ncols = 6;
constexpr int nstates = nrows * ncols;
constexpr int nactions = 4;

using Direction = std::pair<int, int>;
using Agent = RLlib::TabularSarsaAgent<nstates, nactions, Direction>;
using ActionsList = Agent::ActionsList;

int main(int argc, char **argv) {
  assert(argc > 3);
  std::string fname = argv[1];
  int Nstep = std::stoi(argv[2]);
  int train_step = std::stoi(argv[3]);
  double epsilon = std::stof(argv[4]);

  Agent agent(ActionsList{Direction{1, 0}, Direction{0, 1}, Direction{-1, 0},
                          Direction{0, -1}},
              epsilon, 0.5, 0.0);
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

  auto coords = [](int state) {
    return std::make_pair(state / ncols, state % ncols);
  };

  int state = 0;

  std::vector<double> rewards(Nstep, 0.0);
  std::vector<int> states(Nstep, 0);
  agent.SetLearningRate(0.1);
  for (int step = 0; step < Nstep; ++step) {
    auto action = agent.UpdateState(state);

    state = loc((coords(state).first + action.first + nrows) % nrows,
                (coords(state).second + action.second + ncols) % ncols);
    // agent.SetLearningRate(0.1 / (step + 1));
    auto reward = state_values[state];
    rewards[step] = reward;
    states[step] = state;
    agent.CollectReward(reward);
    agent.GetModel().OutputModel("./intermediate_model.txt", ',',
                                 step == 0 ? false : true);
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
      ofs << coords(r).first << "," << coords(r).second << std::endl;
    }
    ofs.close();
  }
  agent.GetModel().OutputModel("./trained_model.txt");
  return 0;
}