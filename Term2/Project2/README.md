**Unscented Kalman Filter Project**

The goals / steps of this project are the following:
  * Implement an unscented Kalman filter using the CTRV motion model in C++.
  * Calibrate the process noise using normalized innovation squared (NIS) data.
  * Measure its accuracy with root-mean-square error (RMSE).

## How to run this project
1. Ensure you have installed the dependencies:
  * cmake: 3.5
    * All OSes: [click here for installation instructions](https://cmake.org/install/)
  * make: 4.1
    * Linux: make is installed by default on most Linux distros
    * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
    * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
  * gcc/g++: 5.4
    * Linux: gcc / g++ is installed by default on most Linux distros
    * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
    * Windows: recommend using [MinGW](http://www.mingw.org/)
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
   * On Windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./UnscentedKF path/to/input.txt path/to/output.txt`. You can find some sample inputs in 'data/'.
   * eg. `./UnscentedKF ../data/obj_pose-laser-radar-synthetic-input.txt ./output.txt`

## [Rubric](https://review.udacity.com/#!/rubrics/783/view) Points

### Compiling

#### Your code should compile.
It does. It has been tested on Windows (MinGW) and Ubuntu on Windows (via the [WSL](https://msdn.microsoft.com/en-us/commandline/wsl/about)). Follow the instructions above and you should be good!

### Accuracy

#### The px, py, vx, vy output coordinates must have an RMSE <= [.09, .10, .40, .30] when using the file: "obj_pose-laser-radar-synthetic-input.txt"
My code gave me this RMSE measurement for file `obj_pose-laser-radar-synthetic-inpu.txt`:
```
RMSE:
0.0698574
 0.102242
 0.345342
 0.236333
```
The accuracy of the program is within the specified parameters.

Also note that the way I adjusted the process noise parameters was by fine tunning them following normalized innovation squared (NIS) data. I started with an approximate value that I reasoned would make sense, and fine tuned to get the radar (3 degrees of freedom) and lidar (2 degrees of freedom) NIS values as close as possible to the expected 95th percentile values.

For the logitudinal acceleration `std_a_` I ended up with a value of `0.185` meters per second squared, which means that 95% of the time we expect acceleration of `-0.37` to `0.37` meters per second squared.

For the angular velocity `std_yawdd_` I ended up with a value of `0.25` rads per second squared, which means a full circle would be completed in 8 seconds (`2pi = 0.25pi rad/s * 8s`).

These are the values I get from the program:
```
NIS Lidar 95% = 5.91035
NIS Radar 95% = 7.82644
For reference: expected 95th percentile NIS for 2 degrees of freedom: 5.991
For reference: expected 95th percentile NIS for 3 degrees of freedom: 7.815
```
### Follows the Correct Algorithm

#### Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.
I took the template code offered by Udacity and filled in the `TODO` entries with the expected implementation. Basically, the implementation follows this structure:
  * `ukf.cpp`: exposes a `ProcessMeasurement` function as the entry point for the UKF. Also, implements the predict and update steps (differentiated by type of measurement - radar or lidar), with variable initialization as part of the constructor.
  * `tools.cpp`: provides utilitarian functions to transform measurements from cartesian to polar system, and calculate the RMSE of the program.
  * `main.cpp`: loads the data, feeds the UKF with the measurements, and keeps track of the ground truth to provide an RMSE measurement at the end of the execution. I also extended it so it'd print the 95th percentile of the NIS for radar and lidar.

#### Your Kalman Filter algorithm handles the first measurements appropriately.
When the fusion EKF is not initialized, the first measurement is used to initialize the state of `x`. If the measurement is radar data, we convert it from the polar to the cartesian coordiante system before feeding it to the kalman filter init funcion. Regardless of the measurement type, we assume zero velocity, zero yaw angle and zero yaw rate.

Also, during this call to the kalman filter's init function we provide initialization values for `P`, which is initialized with an identity matrix.

#### Your Kalman Filter algorithm first predicts then updates.
After the UKF has been initialized we determine the amount of time that has passed since the last measurement was taken. We take this value (in seconds) and use it to call the `Prediction` function. After the preduction is done, we proceed to update the UKF with the measurement package values we obtained.

#### Your Kalman Filter can handle radar and lidar measurements.
During the update phase we inspect the measurement package and depending on the package's sensor type we select which update process we'll use. For radar we call function `UpdateRadar` which knows how to deal with 3-value radar measurements (rho, phi, and rho dot). For lidar we call function `UpdateLidar` which deals with 2-value lidar measurements (px and py).

### Code Efficiency

#### Your algorithm should avoid unnecessary calculations.
The largest optimization that I did, compared to the code we saw in class, was how I normalized the yaw angles after calculating them. The base implementation had an iterative nature when a simple direct formula could be applied. I changed the code that looked like this:
```C++
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
```
to look more like this:
```C++
  if (deltaZ(1) > M_PI)
  {
    double factor = deltaZ(1) / (2 * M_PI);
    double newval = deltaZ(1) - (2 * M_PI * floor(factor));
    if (newval > M_PI)
    {
      newval -= 2 * M_PI;
    }
    deltaZ(1) = newval;
  }
  else if (deltaZ(1) < -M_PI)
  {
    double factor = deltaZ(1) / (2 * M_PI);
    double newval = deltaZ(1) + (2 * M_PI * floor(abs(factor)));
    if (newval < -M_PI)
    {
      newval += 2 * M_PI;
    }
    deltaZ(1) = newval;
  }
```
This change alone makes the code run so quickly that it typically completes processing the provided data file with 500 rows in a few milliseconds. No other major code optimizations were made. I tried to not sacrifice code readability, but for the most part the code has been streamlined to not do unnecessary calculations. In general, the philosophy is to avoid doing unnnecessary calculations when a simple caching strategy will save a lot of time.
