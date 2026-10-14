cmake_minimum_required(VERSION 3.16)
project(wann VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# nlohmann/json: system package if available, otherwise FetchContent.
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found in system, fetching via FetchContent...")
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3 GIT_SHALLOW TRUE)
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  set(JSON_Install    OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(OpenMP REQUIRED)

# rl-tools (header-only). Point this at any rl-tools checkout's include/ dir.
set(RL_TOOLS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../snn-simulator/external/rl-tools/include
    CACHE PATH "rl-tools include directory")
if(NOT EXISTS ${RL_TOOLS_INCLUDE_DIR}/rl_tools/rl_tools.h)
  message(FATAL_ERROR "rl-tools not found at ${RL_TOOLS_INCLUDE_DIR}; pass -DRL_TOOLS_INCLUDE_DIR=...")
endif()
add_library(rl_tools_iface INTERFACE)
target_include_directories(rl_tools_iface SYSTEM INTERFACE ${RL_TOOLS_INCLUDE_DIR})

# Core WANN algorithm (no environment, no SNN)
add_library(wann_lib STATIC
  src/Ind.cpp
  src/NsgaSort.cpp
  src/Wann.cpp
  src/DataGatherer.cpp)
target_include_directories(wann_lib PUBLIC include)
target_link_libraries(wann_lib PUBLIC nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
target_compile_options(wann_lib PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

# Car task (classical ANN on rl-tools CarTrack). rl-tools has non-inline
# symbols, so each environment must live in its own binary / translation unit.
add_library(ann_car_task STATIC src/AnnCarTask.cpp)
target_link_libraries(ann_car_task PUBLIC wann_lib rl_tools_iface)
target_compile_options(ann_car_task PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)

add_executable(wann_car src/main_car.cpp)
target_link_libraries(wann_car PRIVATE ann_car_task OpenMP::OpenMP_CXX)

add_executable(wann_car_eval src/main_eval_car.cpp)
target_link_libraries(wann_car_eval PRIVATE ann_car_task)

# Tests (no training)
include(CTest)
add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE ann_car_task)
add_test(NAME forward_equivalence COMMAND test_forward)

add_executable(test_python_equiv tests/test_python_equiv.cpp)
target_link_libraries(test_python_equiv PRIVATE ann_car_task)
add_test(NAME python_equivalence
         COMMAND test_python_equiv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/python_equiv.json)

add_executable(test_operators tests/test_operators.cpp)
target_link_libraries(test_operators PRIVATE ann_car_task)
add_test(NAME operators COMMAND test_operators)
