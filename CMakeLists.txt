cmake_minimum_required(VERSION 3.16)
project(wann VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native lets the compiler contract a*b+c into FMA, which changes floating-point results in the
# last bits; chaotic physics (BipedalWalker) amplifies that, so the same seed would not give the same
# episode as wann-cpp (built with plain -O3). OFF by default for comparability with the SNN runs.
option(WANN_MARCH_NATIVE "Compile with -march=native (faster, but numerics differ from wann-cpp)" OFF)
if(WANN_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

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

# Core WANN algorithm + classical-ANN task base (no environment, no SNN)
add_library(wann_lib STATIC
  src/Ind.cpp
  src/NsgaSort.cpp
  src/Wann.cpp
  src/DataGatherer.cpp
  src/CompiledNet.cpp
  src/AnnTask.cpp)
target_include_directories(wann_lib PUBLIC include)
target_link_libraries(wann_lib PUBLIC nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
target_compile_options(wann_lib PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

# BipedalWalker is not part of rl-tools: it is the Box2D-free port living in snn-simulator
# (include/rl/environments/bipedal_walker/, header-only).
set(BIPEDAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../snn-simulator/include
    CACHE PATH "directory containing rl/environments/bipedal_walker/")
if(NOT EXISTS ${BIPEDAL_INCLUDE_DIR}/rl/environments/bipedal_walker/operations_generic.h)
  message(WARNING "bipedal_walker headers not found in ${BIPEDAL_INCLUDE_DIR}; wann_bipedal will not be built "
                  "(pass -DBIPEDAL_INCLUDE_DIR=...)")
  set(HAVE_BIPEDAL OFF)
else()
  set(HAVE_BIPEDAL ON)
endif()

# One environment per executable: rl-tools has non-inline symbols, so two environments cannot
# share a binary. For each task X:  ann_X_task (static lib), wann_X (train), wann_X_eval.
function(add_ann_task NAME)
  add_library(ann_${NAME}_task STATIC src/Ann${ARGV1}Task.cpp)
  target_link_libraries(ann_${NAME}_task PUBLIC wann_lib rl_tools_iface)
  target_compile_options(ann_${NAME}_task PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
  add_executable(wann_${NAME} src/main_${NAME}.cpp)
  target_link_libraries(wann_${NAME} PRIVATE ann_${NAME}_task OpenMP::OpenMP_CXX)
  add_executable(wann_${NAME}_eval src/main_eval_${NAME}.cpp)
  target_link_libraries(wann_${NAME}_eval PRIVATE ann_${NAME}_task)
endfunction()

add_ann_task(car        Car)
add_ann_task(acrobot    Acrobot)
add_ann_task(disc_mc    DiscMC)
add_ann_task(bottleneck Bottleneck)
if(HAVE_BIPEDAL)
  add_ann_task(bipedal  Bipedal)
  target_include_directories(ann_bipedal_task SYSTEM PUBLIC ${BIPEDAL_INCLUDE_DIR})
endif()

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

# Environment-equivalence tools (not tests: they need trajectories recorded by wann-cpp; see tests/env/README.md)
foreach(T car acrobot disc_mc bottleneck bipedal)
  if(TARGET ann_${T}_task)
    add_executable(env_replay_${T} tests/env/replay_${T}.cpp)
    target_link_libraries(env_replay_${T} PRIVATE ann_${T}_task)
  endif()
endforeach()
