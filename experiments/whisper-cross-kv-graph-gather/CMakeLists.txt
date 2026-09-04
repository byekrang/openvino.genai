cmake_minimum_required(VERSION 3.10)
project(cand4_cxx CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# The transformation under test lives in the GenAI production tree; compile it
# from there (do not duplicate it into this experiment directory).
set(XFORM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpp/src/whisper/transformations")

add_executable(cand4_test test_main.cpp "${XFORM_DIR}/cross_kv_beam_gather.cpp")
target_include_directories(cand4_test PRIVATE "${XFORM_DIR}")
target_link_libraries(cand4_test PRIVATE openvino::runtime)
