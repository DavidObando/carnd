#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>
#include "json.hpp"
#include "spline.h"
#include "vehicle.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s)
{
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_first_of("}");
    if (found_null != string::npos)
    {
        return "";
    }
    else if (b1 != string::npos && b2 != string::npos)
    {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

double distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

    double closestLen = 100000; //large number
    int closestWaypoint = 0;

    for (int i = 0; i < maps_x.size(); i++)
    {
        double map_x = maps_x[i];
        double map_y = maps_y[i];
        double dist = distance(x, y, map_x, map_y);
        if (dist < closestLen)
        {
            closestLen = dist;
            closestWaypoint = i;
        }
    }

    return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

    int closestWaypoint = ClosestWaypoint(x, y, maps_x, maps_y);

    double map_x = maps_x[closestWaypoint];
    double map_y = maps_y[closestWaypoint];

    double heading = atan2((map_y - y), (map_x - x));

    double angle = abs(theta - heading);

    if (angle > pi() / 4)
    {
        closestWaypoint++;
    }

    return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
    int next_wp = NextWaypoint(x, y, theta, maps_x, maps_y);

    int prev_wp;
    prev_wp = next_wp - 1;
    if (next_wp == 0)
    {
        prev_wp = maps_x.size() - 1;
    }

    double n_x = maps_x[next_wp] - maps_x[prev_wp];
    double n_y = maps_y[next_wp] - maps_y[prev_wp];
    double x_x = x - maps_x[prev_wp];
    double x_y = y - maps_y[prev_wp];

    // find the projection of x onto n
    double proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y);
    double proj_x = proj_norm * n_x;
    double proj_y = proj_norm * n_y;

    double frenet_d = distance(x_x, x_y, proj_x, proj_y);

    //see if d value is positive or negative by comparing it to a center point

    double center_x = 1000 - maps_x[prev_wp];
    double center_y = 2000 - maps_y[prev_wp];
    double centerToPos = distance(center_x, center_y, x_x, x_y);
    double centerToRef = distance(center_x, center_y, proj_x, proj_y);

    if (centerToPos <= centerToRef)
    {
        frenet_d *= -1;
    }

    // calculate s value
    double frenet_s = 0;
    for (int i = 0; i < prev_wp; i++)
    {
        frenet_s += distance(maps_x[i], maps_y[i], maps_x[i + 1], maps_y[i + 1]);
    }

    frenet_s += distance(0, 0, proj_x, proj_y);

    return {frenet_s, frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
    int prev_wp = -1;

    while (s > maps_s[prev_wp + 1] && (prev_wp < (int)(maps_s.size() - 1)))
    {
        prev_wp++;
    }

    int wp2 = (prev_wp + 1) % maps_x.size();

    double heading = atan2((maps_y[wp2] - maps_y[prev_wp]), (maps_x[wp2] - maps_x[prev_wp]));
    // the x,y,s along the segment
    double seg_s = (s - maps_s[prev_wp]);

    double seg_x = maps_x[prev_wp] + seg_s * cos(heading);
    double seg_y = maps_y[prev_wp] + seg_s * sin(heading);

    double perp_heading = heading - pi() / 2;

    double x = seg_x + d * cos(perp_heading);
    double y = seg_y + d * sin(perp_heading);

    return {x, y};
}

double toMetersPerSecond(double milesPerHour)
{
    return milesPerHour / 2.23693629;
}

double toMilesPerHour(double metersPerSecond)
{
    return metersPerSecond * 2.23693629;
}

int toD(int lane)
{
    return (2 + (4 * lane));
}

bool inLane(double d, int lane)
{
    d -= toD(lane);
    return d >= -0.5 & d <= 0.5;
}

// Waypoint map to read from
const string map_file_ = "../data/highway_map.csv";
// The max s value before wrapping around the track back to 0
const double MAX_S = 6945.554;

int main()
{
    uWS::Hub h;

    // Load up map values for waypoint's x,y,s and d normalized normal vectors
    vector<double> map_waypoints_x;
    vector<double> map_waypoints_y;
    vector<double> map_waypoints_s;
    vector<double> map_waypoints_dx;
    vector<double> map_waypoints_dy;

    ifstream in_map_(map_file_.c_str(), ifstream::in);

    string line;
    while (getline(in_map_, line))
    {
        istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        map_waypoints_x.push_back(x);
        map_waypoints_y.push_back(y);
        map_waypoints_s.push_back(s);
        map_waypoints_dx.push_back(d_x);
        map_waypoints_dy.push_back(d_y);
    }

    // vehicle initialization parameters
    int lane = 1;
    double s = 0;
    double v = 0;
    double a = 0;
    Vehicle ego(lane, s, v, a);
    double target_vel = toMetersPerSecond(46.85);
    int lanes_available = 3;
    double max_acceleration = 1.6;
    ego.configure(target_vel, lanes_available, max_acceleration, lane, s + 200);
    ego.last_update = std::chrono::system_clock::now();
    bool is_initialized = false;
    bool is_changing_lane = false;

    map<int, Vehicle> other_vehicles;

    h.onMessage([&is_initialized, &is_changing_lane, &map_waypoints_x, &map_waypoints_y, &map_waypoints_s, &map_waypoints_dx, &map_waypoints_dy, &ego, &other_vehicles]
        (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        //auto sdata = string(data).substr(0, length);
        //cout << sdata << endl;
        if (length && length > 2 && data[0] == '4' && data[1] == '2')
        {

            auto s = hasData(data);
            auto right_now = std::chrono::system_clock::now();

            if (s != "")
            {
                auto j = json::parse(s);

                string event = j[0].get<string>();

                if (event == "telemetry")
                {
                    if (!is_initialized)
                    {
                        ego.last_update = std::chrono::system_clock::now();
                        is_initialized = true;
                    }
                    // j[1] is the data JSON object

                    // Main car's localization Data
                    double car_x = j[1]["x"];
                    double car_y = j[1]["y"];
                    double car_s = j[1]["s"];
                    double car_d = j[1]["d"];
                    double car_yaw = j[1]["yaw"];
                    double car_speed = j[1]["speed"];

                    // Previous path data given to the Planner
                    auto previous_path_x = j[1]["previous_path_x"];
                    auto previous_path_y = j[1]["previous_path_y"];
                    // Previous path's end s and d values
                    double end_path_s = j[1]["end_path_s"];
                    double end_path_d = j[1]["end_path_d"];

                    // Sensor Fusion Data, a list of all other cars on the same side of the road.
                    auto sensor_fusion = j[1]["sensor_fusion"];

                    double prev_size = previous_path_x.size();
                    if (prev_size == 0)
                    {
                        end_path_s = car_s;
                    }

                    std::chrono::duration<double> egodt = right_now - ego.last_update;
                    if (egodt.count() >= 1) // refresh the path planning after this amount of seconds
                    {
                        if (is_changing_lane)
                        {
                            if (inLane(car_d, ego.lane))
                            {
                                is_changing_lane = false;
                            }
                        }
                        if (ego.v == 0)
                        {
                            ego.v = car_speed;
                        }
                        ego.s = car_s;
                        ego.goal_s = ego.s + 200;
                        ego.last_update = right_now;

                        // update car map
                        map<int, vector<vector<double>>> predictions;
                        predictions[-1] = ego.generate_predictions();
                        for (int i = 0; i < sensor_fusion.size(); ++i)
                        {
                            int check_car_id = sensor_fusion[i][0];
                            double check_car_s = sensor_fusion[i][5];
                            if (check_car_s < (ego.s - 70) || check_car_s > (ego.s + 130))
                            {
                                // we're only "seeing" 70 meters back and 130 meters ahead
                                // this car is too far back or too far ahead to matter, skip
                                continue;
                            }
                            double check_car_vx = sensor_fusion[i][3];
                            double check_car_vy = sensor_fusion[i][4];
                            double check_car_v = sqrt((check_car_vx * check_car_vx) + (check_car_vy * check_car_vy));
                            int check_car_d = (int)(((float)sensor_fusion[i][6]) / 4);

                            map<int, Vehicle>::iterator it;
                            if ((it = other_vehicles.find(check_car_id)) != other_vehicles.end())
                            {
                                auto delta_t = (right_now - it->second.last_update).count();
                                it->second.a = 0; // assume zero acceleration
                                it->second.v = check_car_v;
                                it->second.s = check_car_s;
                                it->second.lane = check_car_d;
                                // refresh instance in cache
                                it->second.last_update = right_now;
                                predictions[check_car_id] = it->second.generate_predictions();
                            }
                            else
                            {
                                Vehicle check_car(check_car_d, check_car_s, check_car_v, 0);
                                check_car.last_update = right_now;
                                other_vehicles.insert(std::pair<int,Vehicle>(check_car_id, check_car));
                                predictions[check_car_id] = check_car.generate_predictions();
                            }
                        }
                        // clean up the car cache
                        vector<int> to_delete;
                        // mark for deletion
                        for (auto it = other_vehicles.begin(); it != other_vehicles.end(); ++it)
                        {
                            auto delta_t = (right_now - it->second.last_update).count();
                            if (delta_t > 5.0)
                            {
                                to_delete.push_back(it->first);
                            }
                        }
                        // purge
                        for (auto index = to_delete.begin(); index != to_delete.end(); ++index)
                        {
                            other_vehicles.erase(*index);
                        }

                        auto initial_lane = ego.lane;
                        ego.update_state(predictions);
                        ego.realize_state(predictions);
                        ego.increment(egodt.count());
                        is_changing_lane = is_changing_lane || initial_lane != ego.lane;
                    }

                    int PREVIOUS_PATH_POINTS_TO_TAKE = (int)(ego.v > 20 ? ego.v : 20);

                    vector<double> ptsx;
                    vector<double> ptsy;

                    // reference x, y, yaw states
                    double ref_x = car_x;
                    double ref_y = car_y;
                    double ref_yaw = deg2rad(car_yaw);

                    // if the previous state is almost empty, use the car as starting reference
                    if (prev_size < 2)
                    {
                        // use two points that make the path tangent to the car
                        double prev_car_x = car_x - cos(car_yaw);
                        double prev_car_y = car_y - sin(car_yaw);

                        ptsx.push_back(prev_car_x);
                        ptsx.push_back(car_x);
                        ptsy.push_back(prev_car_y);
                        ptsy.push_back(car_y);
                    }
                    // use the previous path's end point as starting reference
                    else
                    {
                        int reference_state_index = PREVIOUS_PATH_POINTS_TO_TAKE < prev_size ? PREVIOUS_PATH_POINTS_TO_TAKE : prev_size;
                        ref_x = previous_path_x[reference_state_index - 1];
                        ref_y = previous_path_y[reference_state_index - 1];

                        double ref_x_prev = previous_path_x[reference_state_index - 2];
                        double ref_y_prev = previous_path_y[reference_state_index - 2];
                        ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

                        //Use two points that make the path tangent to the previous path's end point
                        ptsx.push_back(ref_x_prev);
                        ptsx.push_back(ref_x);
                        ptsy.push_back(ref_y_prev);
                        ptsy.push_back(ref_y);
                    }

                    // In Frenet, add evenly spaced points ahead of the end of the previous path
                    vector<vector<int>> wp_params = {{10,30,10},{60,90,30}};
                    if (is_changing_lane)
                    {
                        wp_params = {{30,90,30}};
                    }
                    for (auto wp_param = wp_params.begin(); wp_param != wp_params.end(); wp_param++)
                    {
                        for (int i = (*wp_param)[0]; i <= (*wp_param)[1]; i += (*wp_param)[2])
                        {
                            double projected_s = end_path_s + i;
                            if (projected_s > MAX_S)
                            {
                                projected_s = projected_s - MAX_S;
                            }
                            vector<double> next_wp = getXY(projected_s, toD(ego.lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
                            ptsx.push_back(next_wp[0]);
                            ptsy.push_back(next_wp[1]);
                        }
                    }

                    for (int i = 0; i < ptsx.size(); ++i)
                    {
                        //shift car reference angle to 0 degrees
                        double shift_x = ptsx[i] - ref_x;
                        double shift_y = ptsy[i] - ref_y;

                        ptsx[i] = (shift_x * cos(0 - ref_yaw)) - (shift_y * sin(0 - ref_yaw));
                        ptsy[i] = (shift_x * sin(0 - ref_yaw)) + (shift_y * cos(0 - ref_yaw));

                        if (i > 0)
                        {
                            if (ptsx[i - 1] >= ptsx[i])
                            {
                                std::cout << "Error!" << std::endl;
                                std::cout << "i: " << i << std::endl;
                                for (int j = 0; i < ptsx.size(); ++j)
                                {
                                    std::cout << "ptsx[" << j << "]: " << ptsx[j] << std::endl;
                                    std::cout << "ptsy[" << j << "]: " << ptsy[j] << std::endl;
                                }
                            }
                        }
                    }

                    // create a spline
                    tk::spline s;

                    // set (x,y) points to the spline
                    s.set_points(ptsx, ptsy);

                    // Define the actual (x,y) points we'll use for the planner
                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    // Start with previous path points from last time
                    for (int i = 0; i < previous_path_x.size() && i < PREVIOUS_PATH_POINTS_TO_TAKE; ++i)
                    {
                        //Redefine reference state as previous path end point
                        ref_x = previous_path_x[i];
                        ref_y = previous_path_y[i];
                        next_x_vals.push_back(ref_x);
                        next_y_vals.push_back(ref_y);
                    }

                    // Calculate how to break up spline points so that we travel at our desired reference velocity
                    //double target_x = ego.v > 10.0 ? ego.v : 10.0;
                    double target_x = 30;
                    double target_y = s(target_x);
                    double target_dist = sqrt((target_x * target_x) + (target_y * target_y));

                    double div = ego.v * 0.02;
                    double N = fabs(div) > 0.01 ? target_dist / div : 3000.0;

                    // Fill up the rest of our path planner after filling it with previous points, here we will always output 50 points
                    double frenet_x_point = 0.0;
                    double x_add_on = target_x / N;
                    while (next_x_vals.size() <= 50)
                    {
                        double x_point = frenet_x_point + x_add_on;
                        double y_point = s(x_point);

                        frenet_x_point += x_add_on;

                        double x_ref = x_point;
                        double y_ref = y_point;

                        // rotate back to normal plane after rotating from earlier
                        x_point = (x_ref * cos(ref_yaw)) - (y_ref * sin(ref_yaw));
                        y_point = (x_ref * sin(ref_yaw)) + (y_ref * cos(ref_yaw));

                        x_point += ref_x;
                        y_point += ref_y;

                        next_x_vals.push_back(x_point);
                        next_y_vals.push_back(y_point);
                    }

                    json msgJson;
                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;

                    auto msg = "42[\"control\"," + msgJson.dump() + "]";

                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }
            }
            else
            {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    // We don't need this since we're not using HTTP but if it's removed the
    // program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                       size_t, size_t) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1)
        {
            res->end(s.data(), s.length());
        }
        else
        {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                           char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port))
    {
        std::cout << "Listening to port " << port << std::endl;
    }
    else
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
