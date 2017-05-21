/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i < num_particles; ++i)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;
	for (int i = 0; i < num_particles; ++i)
	{
		auto particle = particles[i];
		double delta_x, delta_y, delta_theta;
		//check if yaw_rate is to be factored in
		if(fabs(yaw_rate) > 0.001)
		{
			// factor yaw_rate in
			delta_x = (velocity / yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t) - sin(particle.theta)));
			delta_y = (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t)));
		}
		else
		{
			// yaw_rate is zero
			delta_x = velocity * cos(particle.theta) * delta_t;
			delta_y = velocity * sin(particle.theta) * delta_t;
		}
		delta_theta = yaw_rate * delta_t;
		// create gaissian noise 
		std::normal_distribution<double> dist_x(particle.x + delta_x, std_pos[0]);
		std::normal_distribution<double> dist_y(particle.y + delta_y, std_pos[1]);
		std::normal_distribution<double> dist_theta(particle.theta + delta_theta, std_pos[2]);
		// assign
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); ++i)
	{
		auto o = observations[i];
		int champ_index = -1;
		double smallest_distance = std::numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); ++j)
		{
			auto p = predicted[j];
			double current_distance = dist(p.x, p.y, o.x, o.y);
			if (current_distance < smallest_distance)
			{
				smallest_distance = current_distance;
				champ_index = j;
			}
		}
		o.id = champ_index;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	std::vector<LandmarkObs> map_landmarks_obs;
	for (int i = 0; i < map_landmarks.landmark_list.size(); ++i)
	{
		LandmarkObs obs;
		obs.id = map_landmarks.landmark_list[i].id_i;
		obs.x = map_landmarks.landmark_list[i].x_f;
		obs.y = map_landmarks.landmark_list[i].y_f;
		map_landmarks_obs.push_back(obs);
	}

	for (int p = 0; p < particles.size(); ++p)
	{
		auto particle = particles[p];
		// transform the observation as if it were seen from the particle's perspective
		// it is originally represented in the car's coordinate system
		std::vector<LandmarkObs> transformed_obs;
		for (int i = 0; i < observations.size(); ++i)
		{
			auto car_observation = observations[i];
			LandmarkObs obs;
			obs.id = -1; // no association
			obs.x = particle.x + (car_observation.x * cos(particle.theta)) - (car_observation.y * sin(particle.theta));
			obs.y = particle.y + (car_observation.x * sin(particle.theta)) + (car_observation.y * cos(particle.theta));
			// only add observation if it's within sensor range from the particle
			if (dist(particle.x, particle.y, obs.x, obs.y) <= sensor_range)
			{
				transformed_obs.push_back(obs);
			}
		}
		dataAssociation(map_landmarks_obs, transformed_obs);
		// calculate the weight for this particle
		double weight = 0;
		double probability = 1;
		for (int i = 0; i < transformed_obs.size(); ++i)
		{
			auto observation = transformed_obs[i];
			auto associated_landmark = map_landmarks_obs[observation.id];
			double distance = dist(particle.x, particle.y, associated_landmark.x, associated_landmark.y);
		}
		particle.weight = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
