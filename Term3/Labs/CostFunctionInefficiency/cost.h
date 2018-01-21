#ifndef COST_H
#define COST_H
#include <vector>

float inefficiency_cost(int target_speed, int intended_lane, int final_lane, std::vector<int> lane_speeds);

#endif