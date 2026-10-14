// wann_car – WANN evolution of classical ANNs + rl-tools CarTrack.
#include "../include/wann/AnnCarTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::trainMain<wann::AnnCarTask>(argc, argv,
        {"wann_car", "ANN Car (CarTrack)", "p/car_ann.json", "ann_car"});
}
