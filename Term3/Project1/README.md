# Path Planning project

The goals / steps of this project are the following:
  * Design a path planner that is able to create smooth, safe paths for the car to follow along a 3-lane highway with traffic.
  * A successful path planner will be able to keep inside its lane, avoid hitting other cars, and pass slower moving traffic all by using localization, sensor fusion, and map data.

[//]: # (Image References)

[stress-run]: ./images/stress-run.png "Stress run"
[safety-perimeter]: ./images/safety-perimeter.png "Safety perimeter"
[keep-lane]: ./images/keep-lane.gif "Keep lane"
[passing]: ./images/passing.gif "Passing"
[path-expansion]: ./images/path-expansion.png "Path expansion"

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


## [Rubric](https://review.udacity.com/#!/rubrics/1020/view) Points

### Compilation 

#### The code compiles correctly.
It does. It has been tested on Windows (via the [WSL](https://msdn.microsoft.com/en-us/commandline/wsl/about)), as well as macOS. Follow the instructions above and you should be good!

### Valid Trajectories

#### The car is able to drive at least 4.32 miles without incident.
The car has been able to drive for over 100 miles without incidents being reported during a stress run.

![Stress run][stress-run]

#### The car drives according to the speed limit.
I've set up the vehicle model to have a maximum speed slightly below the 50 miles per hour limit imposed. In `main.cpp` line 228 this is specified:
```C++
double target_vel = toMetersPerSecond(46.85);
```
As you can see this speed limit is expressed in miles per hour (46.85 MPH) but specified to the model in meters per second. This is because the model of this program is set to reason in meters. The simulator mostly uses the metric system when reporting data, even when it displays miles per hour in the display HUD. Likewise, the simulator expects the path points to be expressed in meters given its internal map of the world is in meters.

Finally, it's worth noting that the sensor fusion data obtained from the simulator gives us the other vehicle's speed and location using the metric system. This is very handy as helps us maintain a very consistent world view without much data transformation.

Path planning then takes this value into account when producing an optimal value for the acceleration of the vehicle at any given point in the prediction pipeline. Method `_max_accel_for_lane` in `vehicle.cpp` takes care of this by computing the maximum acceleration that we can produce (in meters per second) given a number of variables:
  - The current velocity.
  - The target velocity (the equivalent of 46.85 miles per hour).
  - The immediate velocity of the nearest car in front.
  - The immediate velocity of the nearest car behind.

All these are combined to produce an optimal value that results in non-jerky acceleration, as well as car safety when approaching cars from behind or when attempting to change lanes.

#### Max Acceleration and Jerk are not Exceeded.
This path planning code reasons about acceleration, but uses a simplified model in which lane changes are immediate. Basically, it's a simplified model where acceleration is seen as a property of the `s` dimension but not of the `d` dimension, in Frenet coordinate speak. I separated the lane shifting acceleration problem to the controller and left the velocity acceleration problem to the path planner. This made my code very easy to work with for multiple reasons:
  1. I could run deep simulations up to 6 seconds in the future using the predictions for all the other vehicles in this simplified world view.
  2. It was easy to use the `PLCL` and `PLCR` (prepare lane change left/right) states to prepare the vehicle's speed to safe lane changes.
  3. Lane switching in the controller could be easily smoothed out with heuristics.

In general the approach works. When I was midway in this implementation, and had only done steps 1 and 2 listed above, there were clear problems with lane switching elevating the total acceleration above the maximum defined in the simulator, and sometimes even the jerk. The lane switching smoothing was done with heuristics about how to generate the points for the spline used to obtain projected Y values given an X coordinate. The points are seeded with the last two points from the previous rendered frame, and are extended in two different ways:
  1. When keeping the lane: previous 2 points, +10, +20, +30, +60, +90.
  2. While switching lanes: previous 2 points, +30, +60, +90.
As seen in `main.cpp` lines 410-429:
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
What this accomplishes is the spline to be very stiff closer to the origin while driving in a lane, but becoming more flexible while switching lanes. The `is_changing_lane` flag is set to `true` whenever the vehicle model changes lane (again, for the model it's an immediate action) and is set back to `false` after the vehicle has reached coordinates close to the center of the target lane.

This simple heuristic allows for fine control over the path while driving on a specific lane, and soft, non-jerky transitions between lanes.

#### Car does not have collisions.
The tests so far indicate the path planning is good enough to avoid collisions most of the time. After over two hours of stress testing, the vehicle has yet to incur in any faults, including collisions.

Path planning accounts for collisions, and penalizes them heavily. Collision detection serves a fundamental role in determining what the most optimal state for the planner is. For example, if continuing at top speed on my current lane will result in a collision, I should find an alternative such as changing lanes or reducing the current velocity. Changing lanes might also result in a collision, so we have to be smart about detecting collisions in lane switching as well.

The proposed solution is to establish what a collision is in terms not of an abstract point in space (which is unrealistic, as a vehicle has dimensionality) but of an area that's safe for the car to operate. When determining whether a collision occurs or not, I don't check whether the `s` dimension of vehicle A and vehicle B is the same. Instead, I account for a front buffer and a rear buffer. Since path planning is only done from the perspective of our car (sometimes referred to as `ego` in the comments) the buffer is defined as follows in `vehicle.h`:
```C++
static const int CHECK_COLLISION_PREFERRED_BUFFER_FRONT = 40;
static const int CHECK_COLLISION_PREFERRED_BUFFER_BACK = 4;
```
This means `ego` wants to have 40 meters in front of it, and 4 meters behind it. If this condition isn't met, we flag the situation as a collision. In `vehicle.cpp`:
```C++
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
```
The above code snippet is comparing the position of a future predicted state of `ego` (`snapshot.s`) against the predicted positions of another car (`s_previous` and `s_now`). We decide there's a collision in three separate cases:
  1. When the other car was behind us and is now 4 meters behind us or closer.
  2. When the other car was ahead of us and is now 40 meters ahead of us or closer.
  3. When the other car is behind us, closer than 4 meters, and its velocity is such that it will ram into us in the next timestep.

Given these conditions we can imagine the safety boxes being drawn somewhat like this:
![Safety perimeter][safety-perimeter]

In effect, these safety boxes help determining if we can change lanes, and also what our cruising velocity can be, as we'll reduce our velocity so as to keep `ego` from crossing into the buffer area.

As a corollary, it would be great if the simulator had other rendering facilities separate from the green bubbles that display the path. I would very much like to draw these safety boxes on the ground, as well as ramifications of all the paths that I'm evaluating and their related cost. This has been very hard to observe given the toolset for this assignment and makes me wish for an improved version of the simulator that also takes other forms of data to display in real time.


#### The car stays in its lane, except for the time between changing lanes.
Yes.

![Keep lane][keep-lane]

#### The car is able to change lanes

Here's an example of these safety boxes in action while the car is attempting to cruise at maximum speed in traffic, and finds an opening through which it can change lanes avoiding a collision with anyone:

![Passing][passing]


### Reflection

The process to get here was iterative, one in which I discovered and fixed issues one at a time. My starting point was the [Path Planning Walkthrough video](https://www.youtube.com/watch?v=7sI3VHFPP0w) linked from the Path Planning project description in Udacity's website. This allowed me to have basic control over the vehicle and to know where to begin with regards to the sensor fusion data.

Next I started to work on separating the vehicle behavior from the basic controller code I already had. I did this by taking parts of the code I had produced for the behavior planning lab in lesson 4 of term 3 and adjusting it for my needs. The main evolution in this code compared with the lab is that this solution is truly a deep search of an optimal path that expands the action tree at every timestep, generating a large number of possible trajectories and selecting the one that yields the lowest cost.

#### Trajectories tree expansion
The original code simply selected a possible state (keep lane, lane change left, etc.) and proceeded to see what would happen 3 seconds in the future after executing this action. The outcome was a less-than-ideal and non-realistic representation of the future. The solution evolved to be what you see in `update_state` in `vehicle.cpp`. Here we're running a prediction engine that will go, 1 second at a time, up to 6 seconds in the future expanding the trajectories tree with all possible states at any given point.

I wish I was able to visualize these trajectories in the simulator to overlay the potential trajectories and their cost as they were happening, but alas, this is not supported. Had I been able to visualize what was going on I would have seen something like:

![Path expansion][path-expansion]

This gives us a set of trajectories (or trajectory expansion) that roughly looks like this:

Lane   |   T=1 |   T=2 |   T=3 |   T=4 |
 ------------ | :-----------: | -----------: | -----------: | -----------: |
Rightmost | `KL`, `LCL` | `KL`, `LCL` | `KL`, `LCL` | `KL`, `LCL` |
Center | | `KL`, `LCL`, `LCR` | `KL`, `LCL`, `LCR` | `KL`, `LCL`, `LCR` |
Leftmost | |  | `KL`, `LCR` | `KL`, `LCR` |

At time T=1, rightmost lane:
  - `KL`: We can stay in the same lane (red line)
  - `LCL`: We can change lane to the left (yellow line)

At time T=2, rightmost lane:
  - `KL`: We can stay in the same lane (red line)
  - `LCL`: We can change lane to the left (yellow line)

At time T=2, center lane:
  - `KL`: We can stay in the same lane (red line)
  - `LCL`: We can change lane to the left (yellow line)
  - `LCR`: We can change lane to the right (green line)

At time T=3, rightmost lane:
  - `KL`: We can stay in the same lane (red line)
  - `LCL`: We can change lane to the left (yellow line)

At time T=3, center lane:
  - `KL`: We can stay in the same lane (red line)
  - `LCL`: We can change lane to the left (yellow line)
  - `LCR`: We can change lane to the right (green line)

At time T=3, leftmost lane:
  - `KL`: We can stay in the same lane (red line)
  - `LCR`: We can change lane to the right (green line)

T=4 expands the same way as T=3 for all lanes.

In reality there are two more state: `PLCL` and `PLCR` that enable the car to do smooth lane changing by adjusting its velocity not only to the car in its current lane, but also to the velocity of cars in the lane to which we're planning to switch. I've omitted these states from the graphic and the table above as it makes things easier to visualize, even though it's technically different.

This path expansion with individual trajectory costing is truly powerful given that we can find the best (cheapest) trajectory that involves all possible operations up to 6 seconds in the future, and leverages the predictions of where the other vehicles will be during the cost calculation of each of the possible trajectories independently. A trajectory that results in a crash will be too costly compared to a clean trajectory. Two equal trajectories will be untied by preferring to stay in the current lane so as to minimize discomfort to the vehicle passengers.

The code can be greatly optimized for performance, but it's functional and can visibly generate predictions that are very approximate to what I as a driver would do.

The cost evaluation for each trajectory is done by a sum of multiple factors:
  - Distance from goal lane: as a way to afinitize the vehicle to its current lane on the long run.
  - Inefficiency cost: as a way to penalize driving too slow.
  - Collision cost: as a way to penalize collisions, the cost is higher the sooner the collision will occur.
  - Change lane cost: a small penalty for changing lanes.
  - State cost: rewards staying in the same lane, and unties prepare state to lane change state when both would otherwise yield the same cost (preferring change lane to perpetually staying in prepare).

After the cost of each of the possible trajectories is calculated, we select the state at T=1 that leads to it. The trajectory could be a complete zigzag around other vehicles, but all we need to know is what state we want to start with and let the trajectory execution unroll the steps as the vehicle makes progress on the road.

#### Vehicle controller
The controller that was implemented for this project is somewhat naive and produces trajectories that allow all rubric points to be achieved. In particular, it takes the output from the path planner and observes what the lane and velocity should be according to it. If a lane change is detected, a lane change operation is started. Velocity is accounted for to determine the distribution of the path points that are to be sent to the simulator.

The controller was based on the [Path Planning Walkthrough video](https://www.youtube.com/watch?v=7sI3VHFPP0w) linked from the Path Planning project description in Udacity's website, following the strategies and suggestions given there.

#### Difficulty of this project
My main struggle with this project was making sense of the sensor fusion data, and ensuring that my model was completely fed by measurements in meters, as initially I produced a model that kept the top velocity in miles per hour (this was a mistake). After ironing out these inconsistencies, the next big challenge was to ensure I could effectively expand the trajectory tree, calculate the cost for each separate trajectory, and obtain the cheapest one. I can tell that the solution I produced can be greatly optimized. For example, if half of my trajectories reuse the first one or  still re-calculate the cost associated for each trajectory independently. I could transform the code to be based more on a greedy strategy where I don't have to re-compute costs associated to common trails of separate trajectories.

Another challenge that I didn't quite complete was producing a more sophisticated controller. The one I delivered with this assignment is functional and does produce smooth lane changes as well as keeping the car in the lane at all times. That said, it would be interesting to mount a PID or MPC that manages the trajectory based on the output of the path planner. This isn't necessary for this assignment to hit all the rubric points, but in practice I don't see how a real self-driving car would do without a more complete controller that accounted for variables such as the dimensionality of the vehicle, tire and road conditions, and jerk minimizing strategies.

#### Final words
Thank you for reading this far. And thank you for reviewing this engaging and challenging project!
