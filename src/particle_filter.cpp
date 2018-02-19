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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Implementation for this method adapted from Lesson 15, Lecture 5 of Udacity SDCND 2018/02/16
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (is_initialized) return;

  /** I picked 25 particles even though just 10 was enough to pass. The error for the three choices was:
   *  10 pts: x 0.162, y 0.135, yaw 0.0005
   *  25 pts: x 0.131, y 0.130, yaw 0.0004
   *  50 pts: x 0.133, y 0.117, yaw 0.0004
   *
   *  Of course, there is some variability due to random sampling, but these values are in the ballpark
   *  of what I found on most trials. It seems like there are diminishing returns going from 25 to 50
   *  points, while there is more improvement in the error going from 10 to 25.
   */
  
  num_particles = 25;
  
  default_random_engine gen;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;
		
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);	 

    Particle particle;
    particle.id = i;
    particle.x = sample_x;
    particle.y = sample_y;
    particle.theta = sample_theta;
    particle.weight = 1.0;

    weights.push_back(1.0);

    particles.push_back(particle);
  }

  is_initialized = true;
}

// The following implementation is adapted from Lesson 15, lecture 8
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  normal_distribution<double> noise_x(0., std_x);
  normal_distribution<double> noise_y(0., std_y);
  normal_distribution<double> noise_theta(0., std_theta);

  for (int i = 0; i < particles.size(); i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    
    if (abs(yaw_rate) > 0.0001) {
      particles[i].x     += velocity / yaw_rate * ( sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[i].y     += velocity / yaw_rate * (-cos(theta + yaw_rate * delta_t) + cos(theta));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x     += velocity * delta_t * sin(theta);
      particles[i].y     += velocity * delta_t * cos(theta);
    }

    particles[i].x     += noise_x(gen);
    particles[i].y     += noise_y(gen);
    particles[i].theta += noise_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); ++i) {
    double closest = numeric_limits<double>::max();
    int id = -1;
    for (int j = 0; j < predicted.size(); ++j) {
      double this_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (closest > this_dist) {
	closest = this_dist;
	id = predicted[j].id;
      }
    }
    observations[i].id = id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  for (int part = 0; part < particles.size(); ++part) {

    Particle particle = particles[part];
    
    // Collect the landmarks within the sensor_range for
    // each particle
    vector<LandmarkObs> landmarks;

    for (int lm = 0; lm < map_landmarks.landmark_list.size(); ++lm) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[lm];

      double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);

      if (sensor_range > distance) {
        landmarks.push_back(LandmarkObs{ landmark.id_i, landmark.x_f, landmark.y_f });
      }
    }

    // transform the observations from car coordinates to map coordinates
    vector<LandmarkObs> observationsTransformed;

    for (int obs = 0; obs < observations.size(); ++obs) {
      float x = observations[obs].x;
      float y = observations[obs].y;
      float theta = particle.theta;
      
      float newx = cos(theta) * x - sin(theta) * y + particle.x;
      float newy = sin(theta) * x + cos(theta) * y + particle.y;

      LandmarkObs newObs = LandmarkObs { observations[obs].id, newx, newy };
      observationsTransformed.push_back(newObs);
    }

    // Compute landmark id's for each observation
    dataAssociation(landmarks, observationsTransformed);

    particles[part].associations.clear();
    particles[part].sense_x.clear();
    particles[part].sense_y.clear();

    // reset weight
    particles[part].weight = 1.0;

    for (int obst = 0; obst < observationsTransformed.size(); ++obst) {
      LandmarkObs observation = observationsTransformed[obst];

      if (observation.id >= 0) {

	// Find the landmark corresponding to the observation id
	Map::single_landmark_s landmark;
	for (int lmf = 0; lmf < map_landmarks.landmark_list.size(); ++lmf) {
	  if (map_landmarks.landmark_list[lmf].id_i == observation.id) {
	    landmark = map_landmarks.landmark_list[lmf];
	  }
	}

	double std_x = std_landmark[0];
        double std_y = std_landmark[1];

	// Compute normalized error terms for exp term
        double delta_x = observation.x - landmark.x_f;
	double delta_y = observation.y - landmark.y_f;
	double delta_x_norm = delta_x * delta_x / (2 * std_x * std_x);
	double delta_y_norm = delta_y * delta_y / (2 * std_y * std_y);
	
        double update_factor = exp(-delta_x_norm - delta_y_norm) / (2 * M_PI * std_x * std_y);

        particles[part].weight *= update_factor;

        // refill association vectors
        particles[part].associations.push_back(observation.id);
        particles[part].sense_x.push_back(observation.x);
        particles[part].sense_y.push_back(observation.y);
      } else {
        particles[part].weight *= 1.0e-25;
      }
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Update the weights vector for use in discrete_distribution
  weights.clear();
  for (int i = 0; i < particles.size(); ++i) {
    weights.push_back(particles[i].weight);
  }

  // Generate random discrete distribution based on weights
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());

  // Create a new vector of particles (we are selecting with resampling and don't want
  // to modify the original particles vector in flight
  vector<Particle> newParticles;
  for(int i = 0; i < particles.size(); ++i) {
    newParticles.push_back(particles[d(gen)]);
  }

  // Update particles based on resampling
  particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
