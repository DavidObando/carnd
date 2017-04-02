#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  Eigen::MatrixXd calculateJacobian3x4(const Eigen::VectorXd& x_state);

  /**
  * Helper to convert cartesian(4) to polar(3); equivalent to function h in radar update EKF
  */
  Eigen::VectorXd cartesian4ToPolar3(const Eigen::VectorXd &x_state);

  /**
  * Helper to convert polar(3) to cartesian(4)
  */
  Eigen::VectorXd polar3ToCartesian4(const Eigen::VectorXd &x_state);

};

#endif /* TOOLS_H_ */
