#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <iterator>

#include "vehicle.h"

using namespace std;

/**
 * Initializes Vehicle
 */
Vehicle::Vehicle(int lane, double s, double v, double a)
{
    this->lane = lane;
    this->s = s;
    this->v = v;
    this->a = a;
    state = "KL";
    max_acceleration = -1;
}

Vehicle::~Vehicle() {}

double Vehicle::calculate_cost(vector<Vehicle> trajectory, map<int, vector<vector<double>>> predictions, string state)
{
    auto trajectory_data = get_helper_data(trajectory, predictions);
    double dfgl = distance_from_goal_lane(trajectory, predictions, trajectory_data);
    double ic = inefficiency_cost(trajectory, predictions, trajectory_data);
    double cc = collision_cost(trajectory, predictions, trajectory_data);
    double clc = change_lane_cost(trajectory, predictions, trajectory_data);
    double sc = state_cost(state);
    return dfgl + ic + cc + clc + sc;
}

TrajectoryData Vehicle::get_helper_data(vector<Vehicle> trajectory, map<int, vector<vector<double>>> predictions)
{
    auto current_snapshot = trajectory[0];
    auto first = trajectory[1];
    auto last = trajectory[trajectory.size() - 1];
    double end_distance_to_goal = this->goal_s - last.s;
    int end_lanes_from_goal = abs(this->goal_lane - last.lane);
    double dt = double(trajectory.size());
    int proposed_lane = first.lane;
    double avg_speed = (last.s - current_snapshot.s) / dt;

    vector<double> accels;
    double closest_approach = 9999999;
    int collides = -1;

    for (int i = 1; i <= PLANNING_HORIZON && i < trajectory.size(); ++i)
    {
        auto snapshot = trajectory[i];
        accels.push_back(snapshot.a);
        for (auto f = predictions.begin(); f != predictions.end(); ++f)
        {
            if (f->first == -1)
            {
                // that is the prediction for the ego car
                continue;
            }
            if (i >= f->second.size())
            {
                // not enough prediction data in this lane for the
                // vehicle being observed
                continue;
            }
            // state is: 0: lane, 1: s, 2: v, 3: a
            auto state = f->second[i];
            if (state[0] != snapshot.lane)
            {
                // car is in a different lane
                continue;
            }
            auto previous_state = f->second[i - 1];
            bool vehicle_collides = check_collision(snapshot, previous_state[1], state[1]);
            if (collides == -1 && vehicle_collides)
            {
                collides = i;
            }
            auto dist = abs(state[1] - snapshot.s);
            if (dist < closest_approach)
            {
                closest_approach = dist;
            }
        }
    }
    double max_accel = accels[0];
    double rms_acceleration = 0;
    for (int i = 0; i < accels.size(); ++i)
    {
        if (accels[i] > max_accel)
        {
            max_accel = accels[i];
        }
        rms_acceleration += pow(accels[i], 2);
    }
    rms_acceleration /= accels.size();

    TrajectoryData trajectory_data;
    trajectory_data.proposed_lane = proposed_lane;
    trajectory_data.avg_speed = avg_speed;
    trajectory_data.max_accel = max_accel;
    trajectory_data.rms_acceleration = rms_acceleration;
    trajectory_data.closest_approach = closest_approach;
    trajectory_data.end_distance_to_goal = end_distance_to_goal;
    trajectory_data.end_lanes_from_goal = end_lanes_from_goal;
    trajectory_data.collides = collides;
    return trajectory_data;
}

bool Vehicle::check_collision(Vehicle snapshot, double s_previous, double s_now)
{
    if (s_previous <= snapshot.s)
    {
        return s_now >= (snapshot.s - CHECK_COLLISION_PREFERRED_BUFFER_BACK);
    }
    if (s_previous > snapshot.s)
    {
        return s_now <= (snapshot.s + CHECK_COLLISION_PREFERRED_BUFFER_FRONT);
    }
    if (s_previous >= (snapshot.s - CHECK_COLLISION_PREFERRED_BUFFER_BACK))
    {
        auto v_target = s_now - s_previous;
        return v_target > snapshot.v;
    }
    return false;
}

double Vehicle::distance_from_goal_lane(vector<Vehicle> trajectory, map<int, vector<vector<double>>> predictions, TrajectoryData data)
{
    double distance = abs(data.end_distance_to_goal);
    distance = max(distance, 1.0);
    double time_to_goal = distance / data.avg_speed;
    double multiplier = 5.0 * data.end_lanes_from_goal / time_to_goal;
    return multiplier * REACH_GOAL;
}

double Vehicle::inefficiency_cost(vector<Vehicle> trajectory, map<int, vector<vector<double>>> predictions, TrajectoryData data)
{
    double diff = this->target_speed - data.avg_speed;
    double pct = diff / this->target_speed;
    double multiplier = pow(pct, 2);
    return multiplier * EFFICIENCY;
}

double Vehicle::collision_cost(vector<Vehicle> trajectory, map<int, vector<vector<double>>> predictions, TrajectoryData data)
{
    if (data.collides != -1)
    {
        double exponent = pow(data.collides, 2);
        double multiplier = exp(-exponent);
        return multiplier * COLLISION;
    }
    return 0;
}

double Vehicle::change_lane_cost(vector<Vehicle> trajectory, map<int, vector<vector<double>>> predictions, TrajectoryData data)
{
    // Penalizes lane changes AWAY from the goal lane and rewards
    // lane changes TOWARDS the goal lane.
    if (data.end_lanes_from_goal > 0)
    {
        return COMFORT;
    }
    return 0;
}

double Vehicle::state_cost(string state)
{
    if (state.compare("CS") == 0)
    {
        return -EFFICIENCY;
    }
    else if (state.compare("KL") == 0)
    {
        return -EFFICIENCY;
    }
    else if (state.compare("PLCL") == 0)
    {
        return EFFICIENCY;
    }
    else if (state.compare("PLCR") == 0)
    {
        return EFFICIENCY;
    }
    return 0;
}

map<int, vector<vector<double>>> deep_copy(map<int, vector<vector<double>>> predictions)
{
    map<int, vector<vector<double>>> pred_copy;
    for (auto p = predictions.begin(); p != predictions.end(); ++p)
    {
        vector<vector<double>> q_copy;
        for (auto q = p->second.begin(); q != p->second.end(); ++q)
        {
            vector<double> r_copy;
            for (auto r = q->begin(); r != q->end(); ++r)
            {
                r_copy.push_back(*r);
            }
            q_copy.push_back(r_copy);
        }
        pred_copy.insert({p->first, q_copy});
    }
    return pred_copy;
}

vector<Vehicle> deep_copy(vector<Vehicle> trajectory)
{
    vector<Vehicle> trajectory_copy;
    for (auto v = trajectory.begin(); v != trajectory.end(); ++v)
    {
        trajectory_copy.push_back(*v);
    }
    return trajectory_copy;
}

vector<string> get_possible_states(Vehicle* v, bool usePrepareStates=true)
{
    vector<string> states = { "KL" };
    if (v->v > 10 && v->lane > 0)
    {
        if (usePrepareStates
            && (v->state == "KL" || v->state == "PLCL"))
        {
            states.push_back("PLCL");
        }
        if (!usePrepareStates || v->state == "PLCL")
        {
            states.push_back("LCL");
        }
    }
    if (v->v > 10 && v->lane < (v->lanes_available - 1))
    {
        if (usePrepareStates &&
            (v->state == "KL" || v->state == "PLCR"))
        {
            states.push_back("PLCR");
        }
        if (!usePrepareStates || v->state == "PLCR")
        {
            states.push_back("LCR");
        }
    }
    return states;
}

void Vehicle::update_state(map<int, vector<vector<double>>> predictions, int horizon)
{
    /*
    Updates the "state" of the vehicle by assigning one of the
    following values to 'self.state':

    "KL" - Keep Lane
     - The vehicle will attempt to drive its target speed, unless there is 
       traffic in front of it, in which case it will slow down.

    "LCL" or "LCR" - Lane Change Left / Right
     - The vehicle will IMMEDIATELY change lanes and then follow longitudinal
       behavior for the "KL" state in the new lane.

    "PLCL" or "PLCR" - Prepare for Lane Change Left / Right
     - The vehicle will find the nearest vehicle in the adjacent lane which is
       BEHIND itself and will adjust speed to try to get behind that vehicle.

    INPUTS
    - predictions 
    A dictionary. The keys are ids of other vehicles and the values are arrays
    where each entry corresponds to the vehicle's predicted location at the 
    corresponding timestep. The FIRST element in the array gives the vehicle's
    current position. Example (showing a car with id 3 moving at 2 m/s):

    {
      3 : [
        {"s" : 4, "lane": 0},
        {"s" : 6, "lane": 0},
        {"s" : 8, "lane": 0},
        {"s" : 10, "lane": 0},
      ]
    }

    */
    vector<string> states = get_possible_states(this);
    map<string, double> costs;
    for (auto s = states.begin(); s != states.end(); s++)
    {
        vector<vector<Vehicle>> trajectories;
        auto simil0 = this->clone();
        simil0.state = *s;
        auto pred_copy = deep_copy(predictions);
        simil0.realize_state(pred_copy);
        // seed the trajectories with the simil0
        trajectories.push_back({ simil0.clone() });
        for (int i = 1; i <= horizon; ++i)
        {
            // pop the closest level of predictions from pred copy
            for (auto p = pred_copy.begin(); p != pred_copy.end(); ++p)
            {
                auto pv = p->second;
                if (pv.size() > 0)
                {
                    // pop the 0th element form the list of predictions for this item
                    pv.erase(pv.begin());
                }
            }
            // extend each of the known trajectories at this horizon step
            vector<vector<Vehicle>> new_trajectories;
            for (auto trajectory = trajectories.begin(); trajectory != trajectories.end(); ++trajectory)
            {
                // get a simil cloned from the last vehicle in the trajectory
                auto simil1 = trajectory->back().clone();
                // fan out all possible states at this future state
                auto simil_states = get_possible_states(&simil1);
                for (auto s1 = simil_states.begin(); s1 != simil_states.end(); s1++)
                {
                    if (*s1 != "KL")
                    {
                        // this is a different trajectory branch, extend our trajectories set
                        auto new_trajectory_branch = deep_copy(*trajectory);
                        auto simil2 = simil1.clone();
                        simil2.state = *s1;
                        simil2.realize_state(pred_copy);
                        simil2.increment(1);
                        new_trajectory_branch.push_back(simil2);
                        new_trajectories.push_back(new_trajectory_branch);
                    }
                }
                // extend the current trajectory by the "KL" branch
                simil1.state = "KL";
                simil1.realize_state(pred_copy);
                simil1.increment(1);
                trajectory->push_back(simil1);
            }
            // add new fanned-out trajectories to the trajectory set for this state
            for (auto trajectory = new_trajectories.begin(); trajectory != new_trajectories.end(); ++trajectory)
            {
                trajectories.push_back(*trajectory);
            }
        }

        double lowestCost = (double)INFINITY_COST;
        for (auto trajectory = trajectories.begin(); trajectory != trajectories.end(); ++trajectory)
        {
            auto cost = calculate_cost(*trajectory, predictions, *s);
            if (cost < lowestCost)
            {
                lowestCost = cost;
            }
        }
        costs.insert({*s, lowestCost});
    }
    string newState = states[0];
    for (auto s = states.begin(); s != states.end(); s++)
    {
        if (costs[*s] < costs[newState])
        {
            newState = *s;
        }
    }
    this->state = newState;
}

void Vehicle::configure(
    double target_speed,
    int lanes_available,
    double max_acceleration,
    int goal_lane,
    int goal_s
)
{
    /*
    Called by simulator before simulation begins. Sets various
    parameters which will impact the ego vehicle. 
    */
    this->target_speed = target_speed;
    this->lanes_available = lanes_available;
    this->max_acceleration = max_acceleration;
    this->goal_lane = goal_lane;
    this->goal_s = goal_s;
}

string Vehicle::display()
{

    ostringstream oss;

    oss << "s:    " << this->s << "\n";
    oss << "lane: " << this->lane << "\n";
    oss << "v:    " << this->v << "\n";
    oss << "a:    " << this->a << "\n";

    return oss.str();
}

void Vehicle::increment(double dt = 1)
{
    this->s += this->v * dt;
    this->v += this->a * dt;
}

vector<double> Vehicle::state_at(int t)
{
    /*
    Predicts state of vehicle in t seconds (assuming constant acceleration)
    */
    double s = this->s + (this->v * t) + (this->a * t * t / 2);
    double v = this->v + (this->a * t);
    return {static_cast<double>(this->lane), s, v, this->a};
}

void Vehicle::realize_state(map<int, vector<vector<double>>> predictions)
{
    /*
    Given a state, realize it by adjusting acceleration and lane.
    Note - lane changes happen instantaneously.
    */
    string state = this->state;
    if (state.compare("CS") == 0)
    {
        realize_constant_speed();
    }
    else if (state.compare("KL") == 0)
    {
        realize_keep_lane(predictions);
    }
    else if (state.compare("LCL") == 0)
    {
        realize_lane_change(predictions, "L");
    }
    else if (state.compare("LCR") == 0)
    {
        realize_lane_change(predictions, "R");
    }
    else if (state.compare("PLCL") == 0)
    {
        realize_prep_lane_change(predictions, "L");
    }
    else if (state.compare("PLCR") == 0)
    {
        realize_prep_lane_change(predictions, "R");
    }
}

void Vehicle::realize_constant_speed()
{
    a = 0;
}

double Vehicle::_max_accel_for_lane(map<int, vector<vector<double>>> predictions, int lane, double s)
{

    double delta_v_til_target = target_speed - v;
    double max_acc = min(max_acceleration, delta_v_til_target);

    map<int, vector<vector<double>>>::iterator it = predictions.begin();
    vector<vector<vector<double>>> in_front;
    vector<vector<vector<double>>> behind_in_proximity;
    while (it != predictions.end())
    {
        vector<vector<double>> v = it->second;

        if (v[0][0] == lane)
        {
            if ((v[0][1] > s))
            {
                in_front.push_back(v);
            }
            else if ((v[0][1] >= s - KEEP_LANE_PREFERRED_BUFFER))
            {
                behind_in_proximity.push_back(v);
            }
        }
        it++;
    }

    if (in_front.size() > 0)
    {
        int min_s = 100000;
        vector<vector<double>> leading = {};
        for (int i = 0; i < in_front.size(); i++)
        {
            if ((in_front[i][0][1] - s) < min_s)
            {
                min_s = (in_front[i][0][1] - s);
                leading = in_front[i];
            }
        }

        double next_pos = leading[1][1];
        double my_next = s + this->v;
        double separation_next = next_pos - my_next;
        double available_room = separation_next - KEEP_LANE_PREFERRED_BUFFER;
        max_acc = min(max_acc, available_room);
    }
    if (behind_in_proximity.size() > 0)
    {
        int max_s = -100000;
        vector<vector<double>> trailing = {};
        for (int i = 0; i < behind_in_proximity.size(); i++)
        {
            if ((behind_in_proximity[i][0][1] - s) > max_s)
            {
                max_s = (behind_in_proximity[i][0][1] - s);
                trailing = behind_in_proximity[i];
            }
        }

        double next_pos = trailing[1][1];
        double my_next = s + this->v;
        double separation_next = my_next - next_pos;
        double available_room = separation_next - KEEP_LANE_PREFERRED_BUFFER;
        max_acc = max(max_acc, available_room);
    }
    if (max_acc < 0)
    {
        max_acc = max(max_acc, -max_acceleration);
    }
    return max_acc;
}

void Vehicle::realize_keep_lane(map<int, vector<vector<double>>> predictions)
{
    this->a = _max_accel_for_lane(predictions, this->lane, this->s);
}

void Vehicle::realize_lane_change(map<int, vector<vector<double>>> predictions, string direction)
{
    int delta = -1;
    if (direction.compare("R") == 0)
    {
        delta = 1;
    }
    this->lane += delta;
    // ensure the new lane is still within boundaries
    if (this->lane >= this->lanes_available)
    {
        this->lane = this->lanes_available - 1;
    }
    else if (this->lane < 0)
    {
        this->lane = 0;
    }
    this->goal_lane = this->lane;
    this->a = _max_accel_for_lane(predictions, this->lane, this->s);
}

void Vehicle::realize_prep_lane_change(map<int, vector<vector<double>>> predictions, string direction)
{
    int delta = -1;
    if (direction.compare("R") == 0)
    {
        delta = 1;
    }
    int proposed_lane = this->lane + delta;
    double proposed_lane_a = _max_accel_for_lane(predictions, proposed_lane, this->s);
    double current_lane_a = _max_accel_for_lane(predictions, this->lane, this->s);
    this->a = min(proposed_lane_a, current_lane_a);
}

vector<vector<double>> Vehicle::generate_predictions(int horizon)
{

    vector<vector<double>> predictions;
    for (int i = 0; i <= horizon; i++)
    {
        vector<double> check1 = state_at(i);
        vector<double> lane_s = {check1[0], check1[1]};
        predictions.push_back(lane_s);
    }
    return predictions;
}