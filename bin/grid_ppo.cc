#include <torch_agents.h>

#include <cassert>
#include <fstream>
#include <iostream>

using Direction = std::array<int, 2>;
using Agent = RLlib::DiscretePPOActorCriticAgent<Direction, double>;
using State = typename Agent::State;
using Position = std::array<int, 2>;
using ActionsList = Agent::ActionsList;

int main(int, char **argv) {
  json config = RLlib::load_json(argv[1]);

  Agent agent(ActionsList{Direction{1, 0}, Direction{0, 1}, Direction{-1, 0},
                          Direction{0, -1}},
              config);
  size_t nrows = config["nrows"];
  size_t ncols = config["ncols"];
  size_t nstates = nrows * ncols;
  auto Nstep = config["Nstep"].get<int>();
  std::vector<double> pos_values(nstates, {});
  {
    auto fname = config["position_values_file"].get<std::string>();
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open input file: " << fname << std::endl;
      return 2;
    }
    for (int i = 0; i < nstates; ++i) {
      ifs >> pos_values[i];
      if (ifs.fail()) {
        std::cerr << "Failed to read state value " << i << " from " << fname
                  << std::endl;
        return 3;
      }
    }
  }

  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      std::cout << pos_values[i * ncols + j] << "\t";
    }
    std::cout << std::endl;
  }

  auto loc = [&](const Position &pos) { return pos[0] * ncols + pos[1]; };

  auto pos = Position{0, 0};

  std::vector<double> rewards(Nstep, 0.0);
  std::vector<Position> positions(Nstep, {{}});
  auto features = [](const Position &s) {
    return State{static_cast<double>(s[0]), static_cast<double>(s[1]),
                 static_cast<double>(s[0] * s[0]),
                 static_cast<double>(s[1] * s[1]),
                 static_cast<double>(s[1] * s[0])};
  };
  std::cout << "Initialization complete, run for " << Nstep << " steps."
            << std::endl;
  for (int step = 0; step < Nstep; ++step) {
    auto action = agent.UpdateState(features(pos));

    for (int i = 0; i < 2; ++i) {
      pos[i] = (pos[i] + action[i] + (i == 0 ? nrows : ncols)) %
               (i == 0 ? nrows : ncols);
    }

    auto reward = pos_values[loc(pos)];

    rewards[step] = reward;
    positions[step] = pos;
    agent.CollectReward(reward);
  }

  // The fixed step budget truncates a continuing task rather than reaching a
  // true terminal state, so bootstrap from the final observation.
  agent.TerminatePath(features(pos));
  agent.FlushBatch();

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
    for (const auto &r : positions) {
      ofs << r[0] << "," << r[1] << std::endl;
    }
    ofs.close();
  }
  agent.GetModel().OutputModel("./trained_model.pt");

  return 0;
}
