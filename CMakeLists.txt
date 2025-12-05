# cmake_minimum_required(VERSION 3.15)
# project(rl_tools VERSION 0.1.0 LANGUAGES CXX)

# # Minimal CMake: build each src/ file into an executable and build tests.

# # Use modern C++ and enable debug symbols by default for local development.
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O3")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O3")

# # Helpful for language servers
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # Make include/ visible
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# # Build all implementation files under src/ into individual executables
# file(GLOB_RECURSE SRC_SOURCES
#      "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
#      "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# if(SRC_SOURCES)
#   # For each source file found under src/, create a standalone executable.
#   foreach(_src IN LISTS SRC_SOURCES)
#     # Use the source filename and replace its extension with .ex for the target name
#     get_filename_component(_file ${_src} NAME)
#     string(REGEX REPLACE "\\.(cc|cpp)$" ".ex" _target "${_file}")
#     add_executable(${_target} ${_src})
#     target_include_directories(${_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#     target_compile_features(${_target} PUBLIC cxx_std_20)
#     # Optional: place executables in bin/ under build dir
#     set_target_properties(${_target} PROPERTIES
#       RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
#   endforeach()
# endif()

# # Tests (GoogleTest)
# option(BUILD_TESTS "Build unit tests" ON)
# if(BUILD_TESTS)
# include(FetchContent)
# # Put all FetchContent downloads and extracted sources under a dedicated
# # folder so they're easy to find and clean. This will create
# # <project-root>/resources/_deps/... when FetchContent downloads projects.
# if(NOT DEFINED FETCHCONTENT_BASE_DIR)
#   set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources" CACHE PATH "Base dir for FetchContent downloads" FORCE)
# endif()
#   FetchContent_Declare(
#     googletest
#     URL https://github.com/google/googletest/archive/refs/heads/main.zip
#   )
#   FetchContent_MakeAvailable(googletest)

#   enable_testing()

#   file(GLOB_RECURSE TEST_SOURCES
#        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
#        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
#   if(TEST_SOURCES)
#     foreach(_src IN LISTS TEST_SOURCES)
#       # Use the source filename and replace its extension with .ex for the test name
#       get_filename_component(_file ${_src} NAME)
#       string(REGEX REPLACE "\\.(cc|cpp)$" ".ex" _target "${_file}")
#       add_executable(${_target} ${_src})
#       target_include_directories(${_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#       target_link_libraries(${_target} PRIVATE GTest::gtest_main)
#       target_compile_features(${_target} PUBLIC cxx_std_20)
#       set_target_properties(${_target} PROPERTIES
#         RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/tests)

#       include(GoogleTest)
#       gtest_discover_tests(${_target})
#     endforeach()
#   endif()
# endif()
cmake_minimum_required(VERSION 3.0)
project(rl_tools)
# set(CMAKE_OSX_DEPLOYMENT_TARGET 13.3)

file(GLOB_RECURSE sources src/*.cc)
file(GLOB_RECURSE sources_test tests/*.cc)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O3")

foreach(_src IN LISTS sources)
  get_filename_component(_file ${_src} NAME_WE)
  add_executable(${_file} ${_src})
  target_include_directories(${_file} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_features(${_file} PUBLIC cxx_std_20)
endforeach()

find_package(GTest REQUIRED)
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/refs/heads/main.zip
# )
# FetchContent_MakeAvailable(googletest)

foreach(testsourcefile ${sources_test})
  get_filename_component(testname ${testsourcefile} NAME_WE)
  add_executable(test_${testname} ${testsourcefile})
  target_compile_options(test_${testname} PUBLIC ${RIGI_COMPILE_FLAGS})
  target_link_libraries(test_${testname} GTest::Main)
  target_include_directories(test_${testname} PUBLIC ${GTEST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach(testsourcefile ${APP_SOURCES})


