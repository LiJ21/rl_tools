cmake_minimum_required(VERSION 3.0)
project(rl_tools)

# Common flags
set(FLAGS ${CMAKE_CXX_FLAGS} -g -O3)

# Test sources
file(GLOB_RECURSE SOURCES_TEST tests/*.cc)

# Executable sources under bin/
file(GLOB_RECURSE BIN_SOURCES bin/*.cc)

# -------------------------------------------------------------------
# Build static library rllib.a from src/*.cc
# -------------------------------------------------------------------
file(GLOB_RECURSE RL_SOURCES
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/extern/*.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/extern/*.c
)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "RL_SOURCES = ${RL_SOURCES}")

if(NOT RL_SOURCES)
    message(FATAL_ERROR "No source files found for rllib under src/ or src/extern")
endif()

add_library(rllib STATIC ${RL_SOURCES})
target_include_directories(rllib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -------------------------------------------------------------------
# Build executables from bin/*.cc and link with rllib
# -------------------------------------------------------------------
foreach(_src IN LISTS BIN_SOURCES)
  get_filename_component(_file ${_src} NAME_WE)
  add_executable(${_file} ${_src})
  target_include_directories(${_file} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${_file} PUBLIC ${FLAGS})
  target_compile_features(${_file} PUBLIC cxx_std_20)
  target_link_libraries(${_file} PRIVATE rllib)
endforeach()

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
find_package(GTest REQUIRED)

foreach(testsourcefile ${SOURCES_TEST})
  get_filename_component(testname ${testsourcefile} NAME_WE)
  add_executable(test_${testname} ${testsourcefile})

  target_compile_options(test_${testname} PUBLIC ${FLAGS})
  target_include_directories(test_${testname} PUBLIC
      ${GTEST_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_compile_features(test_${testname} PUBLIC cxx_std_20)

  target_link_libraries(test_${testname} PRIVATE GTest::Main rllib)
endforeach()
