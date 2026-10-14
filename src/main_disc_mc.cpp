// wann_disc_mc – WANN evolution of classical ANNs + rl-tools MountainCar (discrete actions).
#include "../include/wann/AnnDiscMCTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::trainMain<wann::AnnDiscMCTask>(argc, argv,
        {"wann_disc_mc", "ANN Mountain Car (discrete)", "p/disc_mc_ann.json", "ann_disc_mc"});
}
