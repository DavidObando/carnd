#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2; //TODO: this is off

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2; //TODO: this is off

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);
  //set weights
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double wi = 0;
    if (!i)
    {
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    }
    else
    {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }
  }
  NIS_radar_ = 0;
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_)
  {
    // x = [pos1 pos2 vel_abs yaw_angle yaw_rate]
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      VectorXd c = tools.polar3ToCartesian4(meas_package.raw_measurements_);
      x_(0) = c(0);
      x_(1) = c(1);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }
    x_(2) = 0; // initial velocity is assumed to be zero during initialization
    x_(3) = 0;
    x_(4) = 0;

    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1000, 0, 0,
          0, 0, 0, 1000, 0,
          0, 0, 0, 0, 1000;
    
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // Prediction
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  Prediction(dt);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    //predict sigma points x and y
    double p_xp = p_x, p_yp = p_y;
    //avoid division by zero
    if(fabs(yawd) > 0.001)
    {
      // non-zero case
      p_xp += (v/yawd) * (sin(yaw + (yawd * delta_t)) - sin(yaw));
      p_yp += (v/yawd) * (-cos(yaw + (yawd * delta_t)) + cos(yaw));
    }
    else
    {
      // zero case
      p_xp += v * cos(yaw) * delta_t;
      p_yp += v * sin(yaw) * delta_t;
    }
    // add noise to x and y
    double half_delta_t_squared = 0.5 * (delta_t * delta_t);
    p_xp += half_delta_t_squared * cos(yaw) * nu_a;
    p_yp += half_delta_t_squared * sin(yaw) * nu_a;
    // predict all other points
    double vp = v + (delta_t * nu_a);
    double yawp = yaw + (yawd * delta_t) + (half_delta_t_squared * nu_yawdd);
    double yawdp = yawd + (delta_t * nu_yawdd);
    //write predicted sigma points into right column
    Xsig_pred_(0,i) = p_xp;
    Xsig_pred_(1,i) = p_yp;
    Xsig_pred_(2,i) = vp;
    Xsig_pred_(3,i) = yawp;
    Xsig_pred_(4,i) = yawdp;
  }
  //predict state mean
  VectorXd x = VectorXd(n_x_);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    for (int h = 0; h < n_x_; ++h)
    {
      x(h) += weights_(i) * Xsig_pred_(h,i);
    }
  }
  x_ = x;
  //predict state covariance matrix
  MatrixXd P = MatrixXd(n_x_, n_x_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P += weights_(i) * x_diff * x_diff.transpose();
  }
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;

  // define measurement vector z  
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  // and calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    if (fabs(px) < 0.0001) {
      px = 0.0001;
    }
    if (fabs(py) < 0.0001) {
      py = 0.0001;
    }
    
    float px2 = px * px;
    float py2 = py * py;
    
    //check division by zero
    if (fabs(px2) < 0.0001) {
      px2 = 0.0001;
    }
    if (fabs(py2) < 0.0001) {
      py2 = 0.0001;
    }
    double yaw = Xsig_pred_(3,i);
    double rho = sqrt(px2+py2);
    double phi = atan2(py,px);
    double rhod = ((px*(cos(yaw)*v))+(py*(sin(yaw)*v)))/rho;

    Zsig(0,i) = rho;
    Zsig(1,i) = phi;
    Zsig(2,i) = rhod;

    z_pred(0) += weights_(i) * rho;
    z_pred(1) += weights_(i) * phi;
    z_pred(2) += weights_(i) * rhod;
  }
  //calculate measurement covariance matrix S
  //predict state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += (weights_(i) * z_diff * z_diff.transpose());
  }
  // add process noise
  MatrixXd R = MatrixXd(n_z, n_z);
  R << (std_radr_*std_radr_), 0, 0,
      0, (std_radphi_*std_radphi_), 0,
      0, 0, (std_radrd_*std_radrd_);
  S += R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
      VectorXd deltaX = Xsig_pred_.col(i) - x_;
      //angle normalization
      while (deltaX(3)> M_PI) deltaX(3)-=2.*M_PI;
      while (deltaX(3)<-M_PI) deltaX(3)+=2.*M_PI;
      VectorXd deltaZ = Zsig.col(i) - z_pred;
      Tc += weights_(i) * deltaX * deltaZ.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  x_ = x_ + (K * (z - z_pred));
  P_ = P_ - (K * S * K.transpose());

  //update NIS
  VectorXd z_delta = z - z_pred;
  NIS_radar_ = z_delta.transpose() * S.inverse() * z_delta;
}
