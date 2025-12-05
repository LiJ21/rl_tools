#include <gtest/gtest.h>
#include "random_generator.h"

TEST(RandomGenerator, Uniform01Range) {
  for (int i = 0; i < 1000; ++i) {
    double v = rng_util::uniform01();
    EXPECT_GE(v, 0.0);
    EXPECT_LT(v, 1.0);
  }
}
