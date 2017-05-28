#include "PID.h"

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Kd = Kd;
    this->Ki = Ki;
    this->p_error = 0;
    this->d_error = 0;
    this->i_error = 0;
    _prev_cte = 0;
    _diff_cte = 0;
    _int_cte = 0;
}

void PID::UpdateError(double cte) {
    _diff_cte = _prev_cte - cte;
    _prev_cte = cte;
    _int_cte += cte;
    p_error = -Kp * cte;
    d_error = Kd * _diff_cte;
    i_error = Ki * _int_cte;
}

double PID::TotalError() {
    return p_error - d_error - i_error;
}

