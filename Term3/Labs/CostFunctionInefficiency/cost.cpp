#include <functional>
#include <iostream>
#include "cost.h"
#include "cmath"


using namespace std;

float inefficiency_cost(int target_speed, int intended_lane, int final_lane, vector<int> lane_speeds) {
    /*
    Cost becomes higher for trajectories with intended lane and final lane that have traffic slower than target_speed.
    */
    float final_speed_d = target_speed - lane_speeds[final_lane];
    float intended_speed_d = target_speed - lane_speeds[intended_lane];
    float cost = (final_speed_d + intended_speed_d) / (target_speed * 2);
    
    return cost;
}