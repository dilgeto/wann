// wann_disc_mc_eval – evaluate a trained network on discrete Mountain Car.
#include "../include/wann/AnnDiscMCTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::evalMain<wann::AnnDiscMCTask>(argc, argv,
        {"wann_disc_mc", "ANN Mountain Car (discrete)", "p/disc_mc_ann.json", "ann_disc_mc"});
}
