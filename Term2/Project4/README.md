**PID Controller project**

The goals / steps of this project are the following:
  * Build a PID controller and tune the PID hyperparameters.
  * Learn about the use of the crosstrack error (CTE) in a PID controller.
  * Successfully drive the simulated vehicle.

## How to run this project
1. Ensure you have installed the dependencies.
  * On Ubuntu 16.04, (including Bash on Ubuntu on Windows), run the `install-ubuntu.sh` script in this repository.
2. Dependencies on other platforms:
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
  * [uWebSockets](https://github.com/uWebSockets/uWebSockets)
    * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
    * If you install from source, checkout to commit `e94b6e1`, i.e.
      ```
      git clone https://github.com/uWebSockets/uWebSockets
      cd uWebSockets
      git checkout e94b6e1
      ```
  * Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.
3. Make a build directory: `mkdir build && cd build`
4. Compile: `cmake .. && make`
   * On Windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
5. Run it: `./pid`.

## [Rubric](https://review.udacity.com/#!/rubrics/824/view) Points

### Compiling

#### Your code should compile.
It does. It has been tested on Windows (MinGW) and Ubuntu on Windows (via the [WSL](https://msdn.microsoft.com/en-us/commandline/wsl/about)). Follow the instructions above and you should be good!

### Implementation

#### The PID procedure follows what was taught in the lessons.
As instructed in the lesson, the PID exposes two main steps:
  * Update error: takes the CTE from the current measurement and updates the PID's state.
  * Total error: does the twiddling of the tau coeficients, and returns the computed total error.

The CTE affects the error state of the PID controller, which is separated by P error, I error (the integral) and D error (the derivate). The total error is the sum of the product of these errors by their corresponding tau coeficients.

Additionally, this PID controller behaves as a finate state automaton with respect to the twiddling of the tau coeficients. It keeps a state and runs experiments on the effect of slightly modified coeficient values, perpetually evaluating if new coeficients lead to a reduced mean squared error in the overall execution over a brief period of time.

### Reflection

#### Describe the effect each of the P, I, D components had in your implementation.
After playing with the program's hyperparameters for while it's evident that:
  * `P` plays a key role in actually reaching the center at all. If we went for a value that's too low for P, the car never becomes stable and the CTE just goes out of control. However, `P` by itself gives us only a proverbial "drunkard's path".
  * `D` is fundamental to the sobreity of our controller. It ensures the controller reduces the CTE smoothly over time. Given that it operates on the delta of the current CTE and the previous one, the `D` hyperparameter is typically way larger than the `P` hyperparameter. In my case, it's about one order of magnitude larger.
  * `I` is "integral" to reducing the bias that one experiences in real-world scenarios where components of the vehicle or the sensors do not respond adequately and we must compensate in order to reduce the CTE appropriately. Note that I ended up picking a very small value for `I` given this simulator doesn't really have much of a skew.

#### Describe how the final hyperparameters were chosen.
The program starts with the following default hyperparameters:
```C++
  double initial_kp = 0.31;
  double initial_ki = 0.00055;
  double initial_kd = 3.85;
```
The hyperparameters were obtained by running this program a few times with zeroes and letting the twiddling work. The values are never static, the program updates them continually but I noticed these are okay values to start with given that they are semi stable (the program goes back to them eventually after a short while) and work well for the problem at hand.

### Simulation

#### The vehicle must successfully drive a lap around the track.
In my machine this code successfully runs the vehile around the track. I'm running on a Surface Pro 4, Intel Core i7 @ 2.2 GHz, 16 GB RAM. 

### A little rant
I'm left with the feeling that Udacity doesn't like students using Windows much. I'm curious how the student population looks with respect to machines, OS of choice, access to tools, etc.

I mainly focused on using Bash on Ubuntu on Windows given that the project included the `install-ubuntu.sh` script. That script by the way will only work on Ubuntu 16.04 LTS and more recent installs (I found out the hard way).
