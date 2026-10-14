#include "../../include/wann/AnnAcrobotTask.h"
#include "replay.h"
int main(int argc, char** argv) {
    using namespace wann::replay;
    return run<wann::AnnAcrobotTask>(argc, argv, [](const Csv& c) {
        const int a = column(c, "action");
        std::vector<std::vector<double>> tape;
        for (const auto& r : c.rows) tape.push_back(oneHot(r[a], 3));
        return tape;
    });
}
