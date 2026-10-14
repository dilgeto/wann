#include "../../include/wann/AnnDiscMCTask.h"
#include "replay.h"
int main(int argc, char** argv) {
    using namespace wann::replay;
    return run<wann::AnnDiscMCTask>(argc, argv, [](const Csv& c) {
        const int a = column(c, "action");
        std::vector<std::vector<double>> tape;
        for (const auto& r : c.rows) tape.push_back(oneHot(r[a] - 1.0, 3));   // CSV holds the winning index, not the force
        return tape;
    });
}
