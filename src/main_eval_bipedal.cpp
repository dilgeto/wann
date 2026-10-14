// wann_bipedal_eval – evaluate a trained network on BipedalWalker.
#include "../include/wann/AnnBipedalTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::evalMain<wann::AnnBipedalTask>(argc, argv,
        {"wann_bipedal", "ANN BipedalWalker", "p/bipedal_ann.json", "ann_bipedal"});
}
