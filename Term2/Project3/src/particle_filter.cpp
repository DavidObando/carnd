#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[])
{
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

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	std::default_random_engine gen;
	// create gaissian noise
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);
	for (int i = 0; i < num_particles; ++i)
	{
		auto particle = &particles[i];
		double delta_x, delta_y, delta_theta;
		double theta = particle->theta;
		//check if yaw_rate is to be factored in
		if(fabs(yaw_rate) > 1.e-15)
		{
			// factor yaw_rate in
			double vel_over_yaw = (velocity / yaw_rate);
			double yaw_by_delta_t = (yaw_rate * delta_t);
			delta_x = vel_over_yaw * (sin(theta + yaw_by_delta_t) - sin(theta));
			delta_y = vel_over_yaw * (cos(theta) - cos(theta + yaw_by_delta_t));
			delta_theta = yaw_rate * delta_t;
		}
		else
		{
			// yaw_rate is zero
			double vel_by_delta_t = (velocity * delta_t);
			delta_x = vel_by_delta_t * cos(theta);
			delta_y = vel_by_delta_t * sin(theta);
			delta_theta = 0;
		}
		// assign
		particle->x += delta_x + dist_x(gen);
		particle->y += delta_y + dist_y(gen);
		particle->theta += delta_theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	for (int i = 0; i < observations.size(); ++i)
	{
		auto o = &observations[i];
		int champ_index = -1;
		double smallest_distance = std::numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); ++j)
		{
			auto p = &predicted[j];
			double current_distance = dist(p->x, p->y, o->x, o->y);
			if (current_distance < smallest_distance)
			{
				smallest_distance = current_distance;
				champ_index = j;
			}
		}
		o->id = champ_index;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
	std::vector<LandmarkObs> map_landmarks_obs;
	for (int i = 0; i < map_landmarks.landmark_list.size(); ++i)
	{
		LandmarkObs obs;
		obs.id = map_landmarks.landmark_list[i].id_i;
		obs.x = map_landmarks.landmark_list[i].x_f;
		obs.y = map_landmarks.landmark_list[i].y_f;
		map_landmarks_obs.push_back(obs);
	}

	// we'll calculate the multivariate Gaussian probability density function, in its
	// bivariate case as described in https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double std_x2 = (std_x * std_x);
	double std_y2 = (std_y * std_y);
	// we assume rho to be zero, so we'll omit it in the next statement:
	long double bivariate_base = 1 / (2 * M_PI * std_x * std_y);

	for (int p = 0; p < particles.size(); ++p)
	{
		auto particle = &particles[p];
		// transform the observation as if it were seen from the particle's perspective
		// it is originally represented in the car's coordinate system
		std::vector<LandmarkObs> transformed_obs;
		for (int i = 0; i < observations.size(); ++i)
		{
			auto car_observation = observations[i];
			LandmarkObs obs;
			obs.id = -1; // no association
			obs.x = particle->x + (car_observation.x * cos(particle->theta)) - (car_observation.y * sin(particle->theta));
			obs.y = particle->y + (car_observation.x * sin(particle->theta)) + (car_observation.y * cos(particle->theta));
			// only add observation if it's within sensor range from the particle
			if (dist(particle->x, particle->y, obs.x, obs.y) <= sensor_range)
			{
				transformed_obs.push_back(obs);
			}
		}
		dataAssociation(map_landmarks_obs, transformed_obs);

		if (transformed_obs.size() > 0)
		{
			// implementation of bivariate case of the Gaussian probability density function
			// https://en.wikipedia.org/wiki/Multivariate_normal_distribution
			long double particle_weight = 1;
			for (int i = 0; i < transformed_obs.size(); ++i)
			{
				auto observation = &transformed_obs[i];
				auto associated_landmark = &map_landmarks_obs[observation->id];
				long double x = observation->x;
				long double mu_x = associated_landmark->x;
				long double y = observation->y;
				long double mu_y = associated_landmark->y;
				long double x_minus_mu_x = (x - mu_x);
				long double x_minus_mu_x2 = x_minus_mu_x * x_minus_mu_x;
				long double y_minus_mu_y = (y - mu_y);
				long double y_minus_mu_y2 = y_minus_mu_y * y_minus_mu_y;
				long double a = x_minus_mu_x2 / std_x2;
				long double b = y_minus_mu_y2 / std_y2;
				// we assume rho to be zero, so we'll omit it in the next statement:
				particle_weight *= bivariate_base * exp((-0.5) * (a + b));
			}
			//std::cout << "Particle " << p << " now has weight " << particle_weight << std::endl;
			particle->weight = particle_weight;
		}
		else
		{
			particle->weight = 1.e-150;
		}
	}
}

void ParticleFilter::resample()
{
	std::vector<Particle> resample_bucket;

	int n = particles.size();
	int index = rand() % n;
	double beta = 0;
	double max_weight = std::numeric_limits<double>::min();
	for (int i = 0; i < n; ++i)
	{
		if (particles[i].weight > max_weight)
		{
			max_weight = particles[i].weight;
		}
	}
	double max_weight_2x = 2 * max_weight;
	for (int i = 0; i < n; ++i)
	{
		beta += ((double)rand() / RAND_MAX) * max_weight_2x;
		while (beta > particles[index].weight)
		{
			beta -= particles[index].weight;
			index = (index + 1) % n;
		}
		Particle particle;
		particle.id = i;
		particle.x = particles[index].x;
		particle.y = particles[index].y;
		particle.theta = particles[index].theta;
		particle.weight = particles[index].weight;
		resample_bucket.push_back(particle);
	}
	particles = resample_bucket;
}

void ParticleFilter::write(std::string filename)
{
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i)
	{
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
