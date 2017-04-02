#include "kalman_filter.h"
#include <cmath>
#include <iostream>
using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Ft_ = F_.transpose();
  H_ = H_in;
  Ht_ = H_.transpose();
  R_ = R_in;
  Q_ = Q_in;
  long x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
}

void KalmanFilter::Predict() {
  x_ = (F_ * x_) /*+ u*/;
	P_ = (F_ * P_ * Ft_) + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - (H_ * x_);
  MatrixXd PHt = P_ * Ht_;
  MatrixXd S = (H_ * P_ * Ht_) + R_;
  MatrixXd K = PHt * S.inverse();
  x_ = x_ + (K * y);
  P_ = (I_ - (K * H_)) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd y = z - tools_.cartesian4ToPolar3(x_);
  // normalize ϕ in the y vector so that its angle is between −pi and pi; in other words, add or subtract 2pi from ϕ until it is between −pi and pi.
  if (y(1) < -M_PI) {
    y(1) = y(1) + (2 * M_PI);
  } else if (y(1) > M_PI) {
    y(1) = y(1) - (2 * M_PI);
  }
  MatrixXd Hj = tools_.calculateJacobian3x4(x_);
  MatrixXd Hjt = Hj.transpose();
  MatrixXd PHjt = P_ * Hjt;
  MatrixXd S = (Hj * PHjt) + R_;
  MatrixXd K = PHjt * S.inverse();
  x_ = x_ + (K * y);
  P_ = (I_ - (K * Hj)) * P_;
}
