#include "PID.h"
#include <limits>
#include <iostream>

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double kp, double ki, double kd) {
    _K[PIDEntry::D] = kd;
    _K[PIDEntry::P] = kp;
    _K[PIDEntry::I] = ki;
    _error[PIDEntry::D] = 0;
    _error[PIDEntry::P] = 0;
    _error[PIDEntry::I] = 0;
    _last_error_check = clock();

    _dK[PIDEntry::D] = kd > 0.00000005 ? kd * 0.1 : 1;
    _dK[PIDEntry::P] = kp > 0.00000005 ? kp * 0.1 : 1;
    _dK[PIDEntry::I] = ki > 0.00000005 ? ki * 0.1 : 1;
    _iterations = 0;
    _best_error = std::numeric_limits<double>::max();
    _cumulative_cuadratic_error = 0;
    _current_twiddle_state = TwiddleState::Start;
    _current_twiddle_variable = PIDEntry::D;
}

void PID::UpdateError(double cte) {
    clock_t now = clock();
    float delta_t = ((float)(now - _last_error_check)) / CLOCKS_PER_SEC;
    cout << "delta_t: " << delta_t << endl;
    _last_error_check = now;
    _error[PIDEntry::D] = (cte - _error[PIDEntry::P]) / delta_t;
    _error[PIDEntry::P] = cte;
    _error[PIDEntry::I] += cte;
    _iterations++;
    if (_iterations > TwiddleBottomThreshold) {
        _cumulative_cuadratic_error += cte * cte;
    }
}

double PID::TotalError() {
    Twiddle();
    return (-_K[PIDEntry::P] * _error[PIDEntry::P])
            - (_K[PIDEntry::D] * _error[PIDEntry::D])
            - (_K[PIDEntry::I] * _error[PIDEntry::I]);
}

void PID::Twiddle()
{
    if (_iterations > TwiddleBottomThreshold) {
        double k_sum = (_K[PIDEntry::P] + _K[PIDEntry::D] + _K[PIDEntry::I]);
        if (k_sum > 0.0000005 && k_sum < _tolerance) {
            return;
        }

        // let's twiddle!
        double current_error_point;
        switch(_current_twiddle_state) {
        case TwiddleState::Start:
            cout << "K:  [" << _K[0] << "," << _K[1] << "," << _K[2] << "]" << endl;
            cout << "dK: [" << _dK[0] << "," << _dK[1] << "," << _dK[2] << "]" << endl;
            _current_twiddle_variable = (PIDEntry)((_current_twiddle_variable + 1) % 3);
            _K[_current_twiddle_variable] += _dK[_current_twiddle_variable];
            _cumulative_cuadratic_error = 0;
            _iterations = 0;
            //_error[PIDEntry::D] = 0;
            //_error[PIDEntry::P] = 0;
            //_error[PIDEntry::I] = 0;
            _current_twiddle_state = TwiddleState::IncreaseCheck;
            break;
        case TwiddleState::IncreaseCheck:
            current_error_point = (_cumulative_cuadratic_error / _iterations);
            if (current_error_point < _best_error) {
                _best_error = current_error_point;
                _dK[_current_twiddle_variable] *= 1.1;
                _current_twiddle_state = TwiddleState::Start;
            } else {
                _K[_current_twiddle_variable] -= 2 * _dK[_current_twiddle_variable];
                _cumulative_cuadratic_error = 0;
                _iterations = 0;
                //_error[PIDEntry::D] = 0;
                //_error[PIDEntry::P] = 0;
                //_error[PIDEntry::I] = 0;
                _current_twiddle_state = TwiddleState::DecreaseCheck;
            }
            break;
        case TwiddleState::DecreaseCheck:
            current_error_point = (_cumulative_cuadratic_error / _iterations);
            if (current_error_point < _best_error) {
                _best_error = current_error_point;
                _dK[_current_twiddle_variable] *= 1.1;
            } else {
                _K[_current_twiddle_variable] += _dK[_current_twiddle_variable];
                _dK[_current_twiddle_variable] *= 0.9;
            }
            _current_twiddle_state = TwiddleState::Start;
            break;
        }
    }
}
