// wann_bottleneck – WANN evolution of classical ANNs + rl-tools multi-agent Bottleneck.
#include "../include/wann/AnnBottleneckTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::trainMain<wann::AnnBottleneckTask>(argc, argv,
        {"wann_bottleneck", "ANN Bottleneck (5 agents, shared network)", "p/bottleneck_ann.json", "ann_bottleneck"});
}
