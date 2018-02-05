**Path Planning project**

The goals / steps of this project are the following:
  * Design a path planner that is able to create smooth, safe paths for the car to follow along a 3 lane highway with traffic.
  * A successful path planner will be able to keep inside its lane, avoid hitting other cars, and pass slower moving traffic all by using localization, sensor fusion, and map data.

--------------

## Dependencies
Run either [`install-mac.sh`](./install-mac.sh) or [`install-ubuntu.sh`](install-ubuntu.sh). In Windows it's recommended to use the Linux subsystem and follow the Ubuntu instructions.

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Included in both [`install-mac.sh`](./install-mac.sh) and [`install-ubuntu.sh`](install-ubuntu.sh)
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* [libuv](https://github.com/libuv/libuv)
  * Included in both [`install-mac.sh`](./install-mac.sh) and [`install-ubuntu.sh`](install-ubuntu.sh)

## Basic Build Instructions

1. Make a build directory: `mkdir build && cd build`
2. Compile: `cmake .. && make`
3. Run it: `./path_planning`.

--------------

## [Rubric](https://review.udacity.com/#!/rubrics/1020/view) Points

### Compilation 

#### The code compiles correctly.
It does. It has been tested on Windows (MinGW) and Ubuntu on Windows (via the [WSL](https://msdn.microsoft.com/en-us/commandline/wsl/about)), as well as macOS. Follow the instructions above and you should be good!

### Valid Trajectories

#### The car is able to drive at least 4.32 miles without incident.
The car has been able to drive for over 20 miles without incidents being reported.

### The car drives according to the speed limit.
I've set up the vehicle model to have a maximum speed slightly below the 50 miles per hour limit imposed. In `main.cpp` line 228 this is specified:
```C++
double target_vel = toMetersPerSecond(46.85);
```
As you can see this speed limit is expressed in miles per hour (46.85 MPH) but specified to the model in meters per second. This is because the model of this program is set to reason in meters. The simulator mostly uses the metric system when reporting data, even when it displays miles per hour in the display HUD. Likewise, the simulator expects the path points to be expressed in meters given its internal map of the world is in meters.

Finally, it's worth noting that the sensor fusion data obtained from the simulator gives us the other vehicle's speed and location using the metric system. This is very handy as helps us maintain a very consistent world view without much data transformation.

### Max Acceleration and Jerk are not Exceeded.
This path planning code reasons about acceleration, but uses a simplified model in which lane changes are immediate. Basically, it's a simplified model where acceleration is seen as a property of the `s` dimention but not of the `d` dimention, in Frenet coordinate speak. I separated the lane shifting acceleration problem to the controller and left the velocity acceleration problem to the path planner. This made my code very easy to reason with for multiple reasons:
  1. I could run deep simulations up to 6 seconds in the future using the predictions for all the other vehicles in this simplified world view.
  2. It was easy to use the `PLCL` and `PLCR` (prepare lane change left/right) states to prepare the vehicle's speed to safe lane changes.
  3. Lane switching in the controller could be easily smoothed out with heuristics.

In general the approach works. When I was midway in this implementation, and had only done steps 1 and 2 listed above, there were clear problems with lane switching elevating the total acceleration above the maximum defined in the simulator, and sometimes even the jerk. The lane switching smoothing was done with heuristics about how to generate the points for the spline used to obtain projected Y values given an X coordinate. The points are seeded with the last two points from the previous rendered frame, and are extended in two different ways:
  1. When keeping the lane: previous 2 points, +10, +20, +30, +60, +90.
  2. While switching lanes: previous 2 points, +30, +60, +90.
As seen in main.cpp lines 410-429:
```C++
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
```
What this accomplies is the spline to be very stiff closer to the origin while driving in a lane, but becoming more flexible while switching lanes. The `is_changing_lane` flag is set to true whenever the vehicle model changes lane (again, for the model it's an immediate action) and is set back to false after the vehicle has reached coordinates close to the center of the lane.

This simple heuristic allows for fine control over the path while driving on a specific lane, and soft, non-jerky transitions between lanes.

--------------

The process to get here was iteritative, one in which I discovered and fixed issues one at a time. My starting point was the [Path Planning Walkthrough video](https://www.youtube.com/watch?v=7sI3VHFPP0w) linked from the Path Planning project description in Udacity's website. This allowed me to have basic control over the vehicle and to know where to begin with regards to the sensor fusion data.

Next I started to work on separating the vehicle behavior from the basic controller code I already had. I did this by taking parts of the code I had produced for the behavior planning lab in lesson 4 of term 3 and adjusting it for my needs.
