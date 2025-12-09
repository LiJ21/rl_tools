#pragma once

#include <random>
#include <chrono>
#include <array>

namespace rng_util {

inline std::mt19937_64 &engine() {
    thread_local std::mt19937_64 eng([]() {
        std::random_device rd;
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::array<uint32_t, 4> seed_data{
            static_cast<uint32_t>(rd()),
            static_cast<uint32_t>(rd()),
            static_cast<uint32_t>(now & 0xffffffffu),
            static_cast<uint32_t>((now >> 32) & 0xffffffffu)
        };
        std::seed_seq seq(seed_data.begin(), seed_data.end());
        std::mt19937_64 e;
        e.seed(seq);
        return e;
    }());
    return eng;
}

inline double uniform01() {
    thread_local std::uniform_real_distribution<double> dist(0.0, std::nextafter(1.0, 2.0));
    return dist(engine());
}


inline double normal(double mean = 0.0, double stddev = 1.0) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(engine());
}

} // namespace rng_util
