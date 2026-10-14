// wann_acrobot – WANN evolution of classical ANNs + rl-tools Acrobot.
#include "../include/wann/AnnAcrobotTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::trainMain<wann::AnnAcrobotTask>(argc, argv,
        {"wann_acrobot", "ANN Acrobot", "p/acrobot_ann.json", "ann_acrobot"});
}
