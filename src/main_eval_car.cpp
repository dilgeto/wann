// wann_car_eval – evaluate a trained network on the car task.
#include "../include/wann/AnnCarTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::evalMain<wann::AnnCarTask>(argc, argv,
        {"wann_car", "ANN Car (CarTrack)", "p/car_ann.json", "ann_car"});
}
