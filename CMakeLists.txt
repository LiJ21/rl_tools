cmake_minimum_required(VERSION 3.0)
project(rl_tools)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_PREFIX_PATH /Users/supermeng/my_py/lib/python3.13/site-packages/)
find_package(Torch REQUIRED)

set(FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g -O3)

file(GLOB_RECURSE SOURCES_TEST tests/*.cc)

file(GLOB_RECURSE BIN_SOURCES bin/*.cc)

foreach(_src IN LISTS BIN_SOURCES)
    get_filename_component(_file ${_src} NAME_WE)
    add_executable(${_file} ${_src})
    target_include_directories(${_file} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})
    target_compile_options(${_file} PUBLIC ${FLAGS})
    target_compile_features(${_file} PUBLIC cxx_std_20)
    target_link_libraries(${_file} PRIVATE ${TORCH_LIBRARIES})
endforeach()

find_package(GTest REQUIRED)

foreach(testsourcefile ${SOURCES_TEST})
    get_filename_component(testname ${testsourcefile} NAME_WE)
    add_executable(test_${testname} ${testsourcefile})

    target_compile_options(test_${testname} PUBLIC ${FLAGS})
    target_include_directories(test_${testname} PUBLIC
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_features(test_${testname} PUBLIC cxx_std_20)

    target_link_libraries(test_${testname} PRIVATE
        GTest::Main
        ${TORCH_LIBRARIES}
    )
endforeach()
