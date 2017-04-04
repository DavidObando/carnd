**Extended Kalman Filters Project**

The goals / steps of this project are the following:
  * Implement a 2-dimensional Kalman filter in C++
  * Measure its accuracy with root-mean-square error (RMSE)

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
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find some sample inputs in 'data/'.
   * eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`

## [Rubric](https://review.udacity.com/#!/rubrics/748/view) Points

### Compiling

#### Your code should compile.
It does. It has been tested on Windows (MinGW) and Ubuntu on Windows (via the [WSL](https://msdn.microsoft.com/en-us/commandline/wsl/about)). Follow the instructions above and you should be good!

### Accuracy

#### The px, py, vx, vy output coordinates have an RMSE <= [0.08, 0.08, 0.60, 0.60] when using the file: "sample-laser-radar-measurement-data-1.txt". 
My code gave me this RMSE measurement for file `sample-laser-radar-measurement-data-1.txt`:
```
Accuracy - RMSE:
 0.072925
0.0814852
 0.562949
 0.554507
```
The accuracy of the program is within the specified parameters.

#### The px, py, vx, vy output coordinates have an RMSE <= [0.20, 0.20, .50, .85] when using the file: "sample-laser-radar-measurement-data-2.txt".
My code gave me this RMSE measurement for file `sample-laser-radar-measurement-data-2.txt`:
```
Accuracy - RMSE:
0.194499
 0.19798
0.498665
0.872486
```
The accuracy of the program is within the specified parameters.

### Follows the Correct Algorithm

#### Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.
I took the template code offered by Udacity and filled in the `TODO` entries with the expected implementation. Basically, the implementation follows this structure:
  * `kalman_filter.cpp`: implements the predict and update steps, with an EKF implementation for the update step as well.
  * `FusionEKF.cpp`: consumes the flow of measurements and feeds the kalman filter accordingly.
  * `tools.cpp`: provides utilitarian functions to calculate a Jacobian matrix, transform measurements from cartesian to polar system, and calculate the RMSE of the program.
  * `main.cpp`: loads the data, feeds the FusionEKF with the measurements, and keeps track of the ground truth to provide an RMSE measurement at the end of the execution.

#### Your Kalman Filter algorithm handles the first measurements appropriately.
When the fusion EKF is not initialized, the first measurement is used to initialize the state of `x`. If the measurement is radar data, we convert it from the polar to the cartesian coordiante system before feeding it to the kalman filter init funcion. Regardless of the measurement type, we assume zero velocity on both axes.

Also, during this call to the kalman filter's init function we provide initialization values for `P`, `F`, `H`, `R` and `Q`. `P` is initialized with very high confidence of the position values but very low confidence of the velocity values. Both `F` and `Q` are initialized to default values but these are then immediately updated by subsequent measurement intake calls.

#### Your Kalman Filter algorithm first predicts then updates.
After the fusion EKF has been initialized we first calculate the `F`, `transposed F`, and `Q` matrices before executing the predict step. After the prediction has been made, we take the new measurement and update the kalman filter with it.

#### Your Kalman Filter can handle radar and lidar measurements.
The call to the update step of the kalman filter bifurcates depending on what type of measurement we're taking. Given that our prediction model is based off of a linear (cartesian) model, the lidar measurements use the normal kalman filter code path for the update step. On the other hand the radar measurements use an extended kalman filter update step, which calculates the Jacobian matrix for the current state of the filter at every update step. Also, given that the measurement covariance matrices (`R`) differ between lidar and radar measurements, it's set to the appropriate value before every call to update.

### Code Efficiency

#### Your algorithm should avoid unnecessary calculations.
I tried to not sacrifice code readability, but for the most part the code has been streamlined to not do unnecessary calculations. Transposed matrices that are immutable are cached in the kalman filter itself the same way we hold on to other matrices. In general, the philosophy is to avoid doing unnnecessary calculations when a simple caching strategy will save a lot of time.
