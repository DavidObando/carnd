#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <iterator>

#include "vehicle.h"

/**
 * Initializes Vehicle
 */
Vehicle::Vehicle(int lane, int s, int v, int a) {
    this->lane = lane;
    this->s = s;
    this->v = v;
    this->a = a;
    state = "CS";
    max_acceleration = -1;
}

Vehicle::~Vehicle() {}

double Vehicle::calculate_cost(vector<Vehicle> trajectory, map<int,vector<vector<int>>> predictions)
{
    /*auto trajectory_data = get_helper_data(trajectory, predictions);
    double cost = 0.0;
    cost += distance_from_goal_lane(trajectory, predictions, trajectory_data);
    return cost;*/
    return 0.0;
}

TrajectoryData Vehicle::get_helper_data(vector<Vehicle> trajectory, map<int,vector<vector<int>>> predictions)
{
    auto current_snapshot = trajectory[0];
    auto first = trajectory[1];
    auto last = trajectory[trajectory.size() - 1];
    int end_distance_to_goal = this->goal_s - last.s;
    int end_lanes_from_goal = abs(this->goal_lane - last.lane);
    double dt = double(trajectory.size());
    int proposed_lane = first.lane;
    double avg_speed = (last.s - current_snapshot.s) / dt;

    vector<int> accels;
    int closest_approach = 999999;
    int collides = -1;
    auto filtered = filter_predictions_by_lane(predictions, proposed_lane);

    for(int i = 1; i < PLANNING_HORIZON + 1; ++i)
    {
        auto snapshot = trajectory[i];
        accels.push_back(snapshot.a);
        for (auto f = filtered.begin(); f != filtered.end(); ++f)
        {
            auto state = f->second[i];
            auto last_state = f->second[i - 1];
            bool vehicle_collides = check_collision(snapshot, last_state[1], state[1]);
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
    int max_accel = accels[0];
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

map<int,vector<vector<int>>> Vehicle::filter_predictions_by_lane(map<int,vector<vector<int>>> predictions, int lane)
{
    map<int,vector<vector<int>>> filtered;
    for (auto p = predictions.begin(); p != predictions.end(); ++p)
    {
        auto v_id = p->first;
        auto predicted_traj = p->second;
        if (predicted_traj[0][0] == lane &&  v_id != -1 /*Road::ego_key*/)
        {
            filtered.insert({v_id, predicted_traj});
        }
    }
    return filtered;
}

bool  Vehicle::check_collision(Vehicle snapshot, int s_previous, int s_now)
{
    auto v_target = s_now - s_previous;
    if (s_previous < snapshot.s)
    {
        return s_now >= snapshot.s;
    }
    if (s_previous > snapshot.s)
    {
        return s_now <= snapshot.s;
    }
    if (s_previous == snapshot.s)
    {
        return v_target > snapshot.v;
    }
}

double Vehicle::distance_from_goal_lane(vector<Vehicle> trajectory, map<int,vector<vector<int>>> predictions, TrajectoryData data)
{
    double distance = abs(data.end_distance_to_goal);
    distance = max(distance, 1.0);
    double time_to_goal = distance / data.avg_speed;
    double multiplier = 5.0 * data.end_lanes_from_goal / time_to_goal;
    return multiplier * REACH_GOAL;
}

void Vehicle::update_state(map<int,vector < vector<int> > > predictions) {
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
    vector<string> states = {"KL"};
    if (lane > 0)
    {
        states.push_back("LCL");
    }
    if (lane < lanes_available - 1)
    {
        states.push_back("LCR");
    }
    map<int,vector<vector<int>>> pred_copy;
    for (auto p = predictions.begin(); p != predictions.end(); ++p)
    {
        vector<vector<int>> q_copy;
        for(auto q = p->second.begin(); q != p->second.end(); ++q)
        {
            vector<int> r_copy;
            for(auto r = q->begin(); r != q->end(); ++r)
            {
                r_copy.push_back(*r);
            }
            q_copy.push_back(r_copy);
        }
        pred_copy.insert({p->first, q_copy});
    }
    map<string,double> costs;
    const int horizon = 5;
    for (auto s = states.begin(); s != states.end(); s++)
    {
        vector<Vehicle> trajectories;
        auto simil0 = this->clone();
        simil0.state = *s;
        trajectories.push_back(simil0);
        for (int i = 0; i < horizon; ++i)
        {
            auto simil = this->clone();
            simil.state = *s;
            simil.realize_state(pred_copy);
            simil.increment(1);
            trajectories.push_back(simil);
            for (auto p = pred_copy.begin(); p != pred_copy.end(); ++p)
            {
                auto pv = p->second;
                if (pv.size() > 0)
                {
                    // pop the 0th element form the list of predictions for this item
                    pv.erase(pv.begin());
                }
            }
        }
        costs.insert({*s, calculate_cost(trajectories, predictions)});
    }
    string newState = states[0];
    for (auto s = states.begin(); s != states.end(); s++)
    {
        if(costs[*s] < costs[newState])
        {
            newState = *s;
        }
    }
    this->state = newState;
}

void Vehicle::configure(vector<int> road_data) {
    /*
    Called by simulator before simulation begins. Sets various
    parameters which will impact the ego vehicle. 
    */
    target_speed = road_data[0];
    lanes_available = road_data[1];
    max_acceleration = road_data[2];
    goal_lane = road_data[3];
    goal_s = road_data[4];
}

string Vehicle::display() {

    ostringstream oss;

    oss << "s:    " << this->s << "\n";
    oss << "lane: " << this->lane << "\n";
    oss << "v:    " << this->v << "\n";
    oss << "a:    " << this->a << "\n";

    return oss.str();
}

void Vehicle::increment(int dt = 1) {

    this->s += this->v * dt;
    this->v += this->a * dt;
}

vector<int> Vehicle::state_at(int t) {

    /*
    Predicts state of vehicle in t seconds (assuming constant acceleration)
    */
    int s = this->s + this->v * t + this->a * t * t / 2;
    int v = this->v + this->a * t;
    return {this->lane, s, v, this->a};
}

bool Vehicle::collides_with(Vehicle other, int at_time) {

    /*
    Simple collision detection.
    */
    vector<int> check1 = state_at(at_time);
    vector<int> check2 = other.state_at(at_time);
    return (check1[0] == check2[0]) && (abs(check1[1]-check2[1]) <= L);
}

Vehicle::collider Vehicle::will_collide_with(Vehicle other, int timesteps) {

    Vehicle::collider collider_temp;
    collider_temp.collision = false;
    collider_temp.time = -1; 

    for (int t = 0; t < timesteps+1; t++)
    {
          if( collides_with(other, t) )
          {
            collider_temp.collision = true;
            collider_temp.time = t; 
            return collider_temp;
        }
    }

    return collider_temp;
}

void Vehicle::realize_state(map<int,vector < vector<int> > > predictions) {

    /*
    Given a state, realize it by adjusting acceleration and lane.
    Note - lane changes happen instantaneously.
    */
    string state = this->state;
    if(state.compare("CS") == 0)
    {
        realize_constant_speed();
    }
    else if(state.compare("KL") == 0)
    {
        realize_keep_lane(predictions);
    }
    else if(state.compare("LCL") == 0)
    {
        realize_lane_change(predictions, "L");
    }
    else if(state.compare("LCR") == 0)
    {
        realize_lane_change(predictions, "R");
    }
    else if(state.compare("PLCL") == 0)
    {
        realize_prep_lane_change(predictions, "L");
    }
    else if(state.compare("PLCR") == 0)
    {
        realize_prep_lane_change(predictions, "R");
    }

}

void Vehicle::realize_constant_speed() {
    a = 0;
}

int Vehicle::_max_accel_for_lane(map<int,vector<vector<int> > > predictions, int lane, int s) {

    int delta_v_til_target = target_speed - v;
    int max_acc = min(max_acceleration, delta_v_til_target);

    map<int, vector<vector<int> > >::iterator it = predictions.begin();
    vector<vector<vector<int> > > in_front;
    while(it != predictions.end())
    {

        int v_id = it->first;

        vector<vector<int> > v = it->second;

        if((v[0][0] == lane) && (v[0][1] > s))
        {
            in_front.push_back(v);
        }
        it++;
    }

    if(in_front.size() > 0)
    {
        int min_s = 1000;
        vector<vector<int>> leading = {};
        for(int i = 0; i < in_front.size(); i++)
        {
            if((in_front[i][0][1]-s) < min_s)
            {
                min_s = (in_front[i][0][1]-s);
                leading = in_front[i];
            }
        }

        int next_pos = leading[1][1];
        int my_next = s + this->v;
        int separation_next = next_pos - my_next;
        int available_room = separation_next - preferred_buffer;
        max_acc = min(max_acc, available_room);
    }
    return max_acc;

}

void Vehicle::realize_keep_lane(map<int,vector< vector<int> > > predictions) {
    this->a = _max_accel_for_lane(predictions, this->lane, this->s);
}

void Vehicle::realize_lane_change(map<int,vector< vector<int> > > predictions, string direction) {
    int delta = -1;
    if (direction.compare("L") == 0)
    {
        delta = 1;
    }
    this->lane += delta;
    int lane = this->lane;
    int s = this->s;
    this->a = _max_accel_for_lane(predictions, lane, s);
}

void Vehicle::realize_prep_lane_change(map<int,vector<vector<int> > > predictions, string direction) {
    int delta = -1;
    if (direction.compare("L") == 0)
    {
        delta = 1;
    }
    int lane = this->lane + delta;

    map<int, vector<vector<int> > >::iterator it = predictions.begin();
    vector<vector<vector<int> > > at_behind;
    while(it != predictions.end())
    {
        int v_id = it->first;
        vector<vector<int> > v = it->second;

        if((v[0][0] == lane) && (v[0][1] <= this->s))
        {
            at_behind.push_back(v);
        }
        it++;
    }
    if(at_behind.size() > 0)
    {

        int max_s = -1000;
        vector<vector<int> > nearest_behind = {};
        for(int i = 0; i < at_behind.size(); i++)
        {
            if((at_behind[i][0][1]) > max_s)
            {
                max_s = at_behind[i][0][1];
                nearest_behind = at_behind[i];
            }
        }
        int target_vel = nearest_behind[1][1] - nearest_behind[0][1];
        int delta_v = this->v - target_vel;
        int delta_s = this->s - nearest_behind[0][1];
        if(delta_v != 0)
        {

            int time = -2 * delta_s/delta_v;
            int a;
            if (time == 0)
            {
                a = this->a;
            }
            else
            {
                a = delta_v/time;
            }
            if(a > this->max_acceleration)
            {
                a = this->max_acceleration;
            }
            if(a < -this->max_acceleration)
            {
                a = -this->max_acceleration;
            }
            this->a = a;
        }
        else
        {
            int my_min_acc = max(-this->max_acceleration,-delta_s);
            this->a = my_min_acc;
        }

    }

}

vector<vector<int> > Vehicle::generate_predictions(int horizon = 10) {

    vector<vector<int> > predictions;
    for( int i = 0; i < horizon; i++)
    {
      vector<int> check1 = state_at(i);
      vector<int> lane_s = {check1[0], check1[1]};
      predictions.push_back(lane_s);
      }
    return predictions;

}