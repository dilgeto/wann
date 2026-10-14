// wann_acrobot_eval – evaluate a trained network on Acrobot (see wann::evalMain for options).
#include "../include/wann/AnnAcrobotTask.h"
#include "../include/wann/Driver.h"
int main(int argc, char* argv[]) {
    return wann::evalMain<wann::AnnAcrobotTask>(argc, argv,
        {"wann_acrobot", "ANN Acrobot", "p/acrobot_ann.json", "ann_acrobot"});
}
