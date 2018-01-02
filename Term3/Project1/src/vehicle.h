#ifndef VEHICLE_H
#define VEHICLE_H
#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>
#include <chrono>

using namespace std;

class TrajectoryData {
public:
    int proposed_lane;
    double avg_speed;
    int max_accel;
    double rms_acceleration;
    int closest_approach;
    int end_distance_to_goal;
    int end_lanes_from_goal;
    int collides;
};

class Vehicle {
public:

    struct collider{
        bool collision ; // is there a collision?
        int  time; // time collision happens
    };

    int L = 1;

    int preferred_buffer = 6; // impacts "keep lane" behavior.

    int lane;

    double s;

    double v;

    double a;

    chrono::time_point<chrono::system_clock> last_update;

    double target_speed;

    int lanes_available;

    double max_acceleration;

    int goal_lane;

    int goal_s;

    string state;

    /**
    * Constructor
    */
    Vehicle(int lane, double s, double v, double a);

    /**
    * Destructor
    */
    virtual ~Vehicle();

    void update_state(map<int, vector <vector<double> > > predictions, int horizon = PLANNING_HORIZON);

    void configure(double target_speed, int lanes_available, double max_acceleration, int goal_lane, int goal_s);

    string display();

    void increment(int dt);

    vector<double> state_at(int t);

    bool collides_with(Vehicle other, int at_time);

    collider will_collide_with(Vehicle other, int timesteps);

    void realize_state(map<int, vector < vector<double> > > predictions);

    void realize_constant_speed();

    double _max_accel_for_lane(map<int,vector<vector<double> > > predictions, int lane, double s);

    void realize_keep_lane(map<int, vector< vector<double> > > predictions);

    void realize_lane_change(map<int,vector< vector<double> > > predictions, string direction);

    void realize_prep_lane_change(map<int,vector< vector<double> > > predictions, string direction);

    vector<vector<double> > generate_predictions(int horizon = PLANNING_HORIZON);

    // new functions:

    Vehicle clone()
    {
        Vehicle c(this->lane, this->s, this->v, this->a);
        c.target_speed = this->target_speed;
        c.lanes_available = this->lanes_available;
        c.max_acceleration = this->max_acceleration;
        c.goal_lane = this->goal_lane;
        c.goal_s = this->goal_s;
        c.state = this->state;
        return c;
    };

    double calculate_cost(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions);
    TrajectoryData get_helper_data(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions);
    map<int,vector<vector<double>>> filter_predictions_by_lane(map<int,vector<vector<double>>> predictions, int lane);
    bool check_collision(Vehicle snapshot, int s_previous, int s_now);
    double distance_from_goal_lane(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions, TrajectoryData data);
    double inefficiency_cost(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions, TrajectoryData data);
    double collision_cost(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions, TrajectoryData data);
    double buffer_cost(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions, TrajectoryData data);
    double change_lane_cost(vector<Vehicle> trajectory, map<int,vector<vector<double>>> predictions, TrajectoryData data);

private:
    static const int DESIRED_BUFFER = 2; // timesteps
    static const int PLANNING_HORIZON = 3;

    static const long long REACH_GOAL = pow(10, 1);
    static const long long COMFORT    = pow(10, 2);
    static const long long DANGER     = pow(10, 4);
    static const long long EFFICIENCY = pow(10, 6);
    static const long long COLLISION  = pow(10, 8);

};

#endif
