#include "../../include/wann/AnnBipedalTask.h"
#include "replay.h"
int main(int argc, char** argv) {
    using namespace wann::replay;
    return run<wann::AnnBipedalTask>(argc, argv, [](const Csv& c) {
        const int a0 = column(c, "a0");
        std::vector<std::vector<double>> tape;
        for (const auto& r : c.rows) tape.push_back({r[a0], r[a0 + 1], r[a0 + 2], r[a0 + 3]});
        return tape;
    });
}
