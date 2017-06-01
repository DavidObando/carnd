#ifndef PID_H
#define PID_H

#include <time.h>

enum PIDEntry
{
  P = 0,
  I = 1,
  D = 2
};

class PID {
public:
  /*
  * Errors
  */
  double _error[3];

  /*
  * Coefficients
  */ 
  double _K[3];

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double kp, double ki, double kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

private:
  clock_t _last_error_check;
  long long _iterations;
  double _cumulative_cuadratic_error;
  double _best_error;
  enum TwiddleState
  {
    Start,
    IncreaseCheck,
    DecreaseCheck
  } _current_twiddle_state;
  PIDEntry _current_twiddle_variable;
  double _dK[3];
  double _tolerance = 0.00005;
  const int TwiddleBottomThreshold = 20;
  void Twiddle();
};

#endif /* PID_H */
