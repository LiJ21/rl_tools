cmake_minimum_required(VERSION 3.0)
project(rl_tools)

file(GLOB_RECURSE sources src/*.cc)
file(GLOB_RECURSE sources_test tests/*.cc)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O3")

foreach(_src IN LISTS sources)
  get_filename_component(_file ${_src} NAME_WE)
  add_executable(${_file} ${_src})
  target_include_directories(${_file} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(${_file} PUBLIC cxx_std_20)
endforeach()

find_package(GTest REQUIRED)

foreach(testsourcefile ${sources_test})
  get_filename_component(testname ${testsourcefile} NAME_WE)
  add_executable(test_${testname} ${testsourcefile})
  target_compile_options(test_${testname} PUBLIC ${RIGI_COMPILE_FLAGS})
  target_link_libraries(test_${testname} GTest::Main)
  target_include_directories(test_${testname} PUBLIC ${GTEST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach(testsourcefile ${APP_SOURCES})


