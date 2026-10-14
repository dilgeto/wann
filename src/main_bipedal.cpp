// wann_bipedal – WANN evolution of classical ANNs + BipedalWalker (Hardcore by default).
#include "../include/wann/AnnBipedalTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::trainMain<wann::AnnBipedalTask>(argc, argv,
        {"wann_bipedal", "ANN BipedalWalker", "p/bipedal_ann.json", "ann_bipedal"});
}
