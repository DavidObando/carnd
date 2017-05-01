#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(5);
	rmse << 0,0,0,0,0;
	if (!estimations.size()) {
    std::cout << "Invalid estimation vector size" << std::endl;
    return rmse;
	}
	if (estimations.size() != ground_truth.size()) {
    std::cout << "Estimation vector size doesn't match ground_truth vector size" << std::endl;
    return rmse;
	}

	for(int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse = rmse + residual;
	}

	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();
	return rmse;
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