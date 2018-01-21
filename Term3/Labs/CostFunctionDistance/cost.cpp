#include <functional>
#include <iostream>
#include "cost.h"
#include "cmath"


using namespace std;

float goal_distance_cost(int goal_lane, int intended_lane, int final_lane, float distance_to_goal) {
    /*
    The cost increases with both the distance of intended lane from the goal
    and the distance of the final lane from the goal. The cost of being out of the 
    goal lane also becomes larger as vehicle approaches the goal.
    */

    float intended_delta_d = goal_lane - intended_lane;
    float final_delta_d = goal_lane - final_lane;
    float delta_d = intended_delta_d + final_delta_d;
    float delta_s = distance_to_goal;
    float cost = 1 - exp(delta_d / delta_s);
    
    return cost;
}