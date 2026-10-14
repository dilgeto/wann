#include "../../include/wann/AnnCarTask.h"
#include "replay.h"
int main(int argc, char** argv) {
    using namespace wann::replay;
    return run<wann::AnnCarTask>(argc, argv, [](const Csv& c) {
        const int t = column(c, "throttle"), s = column(c, "steering");
        std::vector<std::vector<double>> tape;
        for (const auto& r : c.rows) tape.push_back({r[t], r[s]});
        return tape;
    });
}
