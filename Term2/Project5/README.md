**MPC Controller project**

The goals / steps of this project are the following:
  * Build an MPC (Model Predictive Control) controller and tune its hyperparameters.
  * Take into account a 100 millisecond delay between decision and actuation.
  * Successfully drive the simulated vehicle.

## Dependencies

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
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
   * On Windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./mpc`.


## [Rubric](https://review.udacity.com/#!/rubrics/896/view) Points

### Compiling

#### Your code should compile.
It does. It has been tested on Windows (MinGW) and Ubuntu on Windows (via the [WSL](https://msdn.microsoft.com/en-us/commandline/wsl/about)), as well as macOS. Follow the instructions above and you should be good!

### Implementation

#### The Model
The model is composed of:
 0. Coefficients
 1. State
 2. Actuators
 3. Update equations

##### Coefficients
The coefficients are the set of waypoints fitted from a third-degree polinomial based on the `x` and `y` points where we want the vehicle to be. These waypoints are used to describe the desired path, and to run the MPC algorithm.

##### State
The state of the vehicle is composed of the following components:
  - `x`: X coordinate.
  - `y`: Y coordinate.
  - `psi`: the rotation of the car (in radians) where 0 degrees is aligned to the X axis, increasing the value counterclockwise.
  - `cte`: the cross track error.
  - `epsi`: the PSI error.

The state is used to described the car in the context of the world.

##### Actuators
The actuators are:
  - `delta`: the steering angle.
  - `alpha`: the acceleration.

This is our output. We'll take the coefficients derived from the path points, and the state of the vehicle in order to obtain what are the expected values of the actuators that we'll give the car. Note that these actuators have specific constraints we want to observe, such as what is the maximum angle we want the car to produce.

##### Update equations
The equations give information to a solver in order for it to find the most cost effective solution to the problem of finding actuator values that, in the future, affect the trajectory and velocity of the car to be as close as possible to the desired state.

There first part to these equations is the cost function, which as thus described:
```cpp
    // Reference State Cost
    // The part of the cost based on the reference state.
    for (int t = 0; t < N; t++) {
      fg[0] += 0.5 * CppAD::pow(vars[cte_start + t], 2);
      fg[0] += 5 * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators.
    for (int t = 0; t < N - 1; t++) {
      fg[0] += 5000 * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int t = 0; t < N - 2; t++) {
      fg[0] += 500 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }
```

This means that the cost is constructed in 3 steps:
  - reference state, where I'm only taking 0.5 of the squared cross track error, 5 times the squared psi error, and 1 time the squared velocity difference.
  - actuator values, where I'm especially minimizing the use of the steering wheel as much as possible.
  - difference between sequential actuators, where again I'm minimizing sudden changes in the steering wheel.

The second part of these equations is the constraints, defined as:
```cpp
    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int t = 1; t < N; t++) {
      // The state at time t+1
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // The state at time t.
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 * x0;
      AD<double> psides0 = CppAD::atan(coeffs[1] + (2 * coeffs[2] * x0) + (3 * coeffs[3] * (x0*x0)));

      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      AD<double> psi1c = psi0 + (v0 / Lf * delta0 * dt);
      fg[1 + x_start + t] = x1 - (x0 + (v0 * CppAD::cos(psi0) * dt));
      fg[1 + y_start + t] = y1 - (y0 + (v0 * CppAD::sin(psi0) * dt));
      fg[1 + psi_start + t] = psi1 - psi1c;
      fg[1 + v_start + t] = v1 - (v0 + (a0 * dt));
      fg[1 + cte_start + t] = cte1 - (f0 - y0 + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] = epsi1 - psi1c;
    }
```

The idea here is to inform the solver of how the constraints evolve given the provided values for the actuators.

#### Timestep Length and Elapsed Duration (N & dt)
I decided to use the following values:
```cpp
size_t N = 15;
double dt = 0.05;
```

I started with larger values for N, but that slowed down the solver enough that it impacted the latency of the system. I also started with a larger value for dt given that I thought a 0.1 second latency in the system would also mean that this had to be 0.1 seconds. I then understood that the system latency doesn't have much to do with the elapsed duration between points in the MPC algorithm, as these points are only used to describe a trajectory that approximates the waypoints.

I arrived at these final values after playing around with multiple combinations in which the multiplication of the timestep length by the elapsed duration (the "green line" in the simulator) was either too short to fully operate on the waypoints, or way longer than the waypoints at hand would show. When the green line was too short, it wouldn't find values that adequately express the waypoints change further down the road; and when it was too long compared to the wayponts, the solver isn't smart enough to detect this and the resulting trajectory could lead anywhere.

These values produced a nice line that closely approximated the known waypoints most of the time, and that lone was the best determinator of whether the car in the simulator would stay in the road or not.

#### Model Predictive Control with Latency
One important aspect of the project is to implement the MPC accounting for latency. One way to implement this is to run the MPC solver using the vehicle model starting from the current state for the duration of the latency. From the code:
```cpp
          // I wanted to predict the trajectory in a future state, and used this to assume a
          // future position before using the MPC solver.
          // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
          // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
          // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
          // v_[t+1] = v[t] + a[t] * dt
          // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
          // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
          double steering_angle = j[1]["steering_angle"];
          double throttle = j[1]["throttle"];
          const double latency = 0.1; //100 milliseconds
          const double Lf = 2.67;
          double velocity_seconds = v * 0.44704; //mph to meters per second
          double future_move = velocity_seconds * latency;
          double future_x = future_move; // we're assuming psi to be zero, so all the move will be in x
          double future_y = 0;
          double future_psi = velocity_seconds / Lf * deg2rad(steering_angle) * latency;
          double future_v = v + (throttle * latency);
          double future_cte = cte + (velocity_seconds * sin(epsi) * latency);
          double future_epsi = epsi + future_psi;
          state << future_x, future_y, future_psi, future_v, future_cte, future_epsi;
          // The above code replaces this line, which would work well assuming no latency:
          // state << 0, 0, 0, v, cte, epsi;
          auto vars = mpc.Solve(state, coeffs);
```

As you can see, I call the MPC solver in the last line of the snippet. The state I use is calculated by means of the equations of the model assuming a worldview where the car is located at `x`=0, `y`=0 and `phi`=0, and then calculating where the car will be at the posterior time of the specified latency.

### Simulation

#### The vehicle must successfully drive a lap around the track.
The car successfully drives around the track. Its tires never leave the drivable portion of the track surface, nor doe sthe car pop into any ledges or roll over any surfaces that would otherwise be considered unsafe if humans were in the vehicle. I'm running on a Surface Pro 4, Intel Core i7 @ 2.2 GHz, 16 GB RAM; and a MacBook Pro (Late 2013), Intel Core i7 @ 2.6 GHz, 16 GB RAM.
