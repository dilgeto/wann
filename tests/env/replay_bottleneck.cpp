#include "../../include/wann/AnnBottleneckTask.h"
#include "replay.h"
int main(int argc, char** argv) {
    using namespace wann::replay;
    return run<wann::AnnBottleneckTask>(argc, argv, [](const Csv& c) {
        // row: step, then per agent 12 observations + 2 actions, then reward
        constexpr int PER = 14, N = wann::AnnBottleneckTask::N_AGENTS;
        std::vector<std::vector<double>> tape;
        for (const auto& r : c.rows)
            for (int a = 0; a < N; ++a) tape.push_back({r[1 + a * PER + 12], r[1 + a * PER + 13]});
        return tape;
    });
}
