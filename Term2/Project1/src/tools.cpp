#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
	rmse << 0,0,0,0;
  // check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	if (!estimations.size()) {
    cout << "Invalid estimation vector size" << endl;
    return rmse;
	}
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size()) {
    cout << "Estimation vector size doesn't match ground_truth vector size" << endl;
    return rmse;
	}
	// ... your code here

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse = rmse + residual;
	}

	//calculate the mean
	rmse = rmse / estimations.size();

	//calculate the squared root
	// ... your code here
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

VectorXd Tools::cartesian4ToPolar3(const VectorXd &x_state) {
  VectorXd P(3);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

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

  // calculate
  float rho = sqrt((px2) + (py2));
  float phi = atan2(py, px);
  float rho_dot = ((px * vx) + (py * vy)) / rho;

  P << rho, phi, rho_dot;
  return P;
}

VectorXd Tools::polar3ToCartesian4(const VectorXd &x_state) {
  VectorXd C(4);
	//recover state parameters
  float rho = x_state(0);
  float phi = x_state(1);
  float rho_dot = x_state(2);

  if (fabs(rho) < 0.0001) {
    rho = 0.0001;
  }

  // calculate
  float px = rho * cos(phi);
  float py = rho * sin(phi);
  float vx = ((rho_dot / rho) - py) / px;
  float vy = 1;

  C << px, py, vx, vy;
  return C;
}

MatrixXd Tools::calculateJacobian3x4(const VectorXd& x_state) {
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

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

	/*if (!px2 & !py2) {
    cout << "Division by zero" << endl;
	  return Hj;
	}*/
	
	//compute the Jacobian matrix
	float h00 = px / sqrt(px2 + py2);
	float h01 = py / sqrt(px2 + py2);
	float h02 = 0.0;
	float h03 = 0.0;
	float h10 = -py / (px2 + py2);
	float h11 = px / (px2 + py2);
	float h12 = 0.0;
	float h13 = 0.0;
	float h20 = (py * ((vx * py) - (vy * px))) / (float)pow((px2 + py2), 3/2);
	float h21 = (px * ((vy * px) - (vx * py))) / (float)pow((px2 + py2), 3/2);
	float h22 = px / sqrt(px2 + py2);
	float h23 = py / sqrt(px2 + py2);
	
	Hj << h00, h01, h02, h03,
	      h10, h11, h12, h13,
	      h20, h21, h22, h23;

	return Hj;
}
