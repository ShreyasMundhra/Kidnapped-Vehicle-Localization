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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 201;
  
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
  
	for(unsigned int i = 0; i < num_particles; i++){
      Particle p;
      p.id = i;
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0;
      particles.push_back(p);
    }
  
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  	default_random_engine gen;
	for(unsigned int i = 0; i < num_particles; i++){
      Particle p = particles[i];
      double x_m, y_m, theta_m;
      
      if(fabs(yaw_rate) < 0.00001){
        x_m = p.x + velocity * delta_t * cos(p.theta);
        y_m = p.y + velocity * delta_t * sin(p.theta);
        theta_m = p.theta;
      }
      else{
        x_m = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
        y_m = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        theta_m = p.theta + yaw_rate * delta_t;
      }
      
      normal_distribution<double> dist_x(x_m, std_pos[0]);
      normal_distribution<double> dist_y(y_m, std_pos[1]);
      normal_distribution<double> dist_t(theta_m, std_pos[2]);
      
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_t(gen);
      
      particles[i] = p;
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  for(unsigned int i = 0; i < observations.size(); i++){
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    
    double min_dist = std::numeric_limits<double>::max();
    int assoc_id = -1;
    for(unsigned int j = 0; j < predicted.size(); j++){
      double distance = dist(obs_x, obs_y, predicted[j].x, predicted[j].y);
      if(distance < min_dist){
        min_dist = distance;
        assoc_id = predicted[j].id;
      }
    }
    observations[i].id = assoc_id;
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
  for(unsigned int i = 0; i < num_particles; i++){
    Particle p = particles[i];
    std::vector<LandmarkObs> predicted;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
      Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
      if(dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range){
        predicted.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }
    
    vector<LandmarkObs> transformed_obs;
    for(unsigned int j = 0; j < observations.size(); j++){
      LandmarkObs obs = observations[j];
      double x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
      double y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
      obs.x = x;
      obs.y = y;
      transformed_obs.push_back(obs);
    }
    
    dataAssociation(predicted, transformed_obs);
    
    double weight = 1.0;
    for(unsigned int j = 0; j < transformed_obs.size(); j++){
      double x = transformed_obs[j].x;
      double y = transformed_obs[j].y;
            
      double x_m, y_m;
      for(unsigned int k = 0; k < predicted.size(); k++){
        if(transformed_obs[j].id == predicted[k].id){
          x_m = predicted[k].x;
          y_m = predicted[k].y;
        }
      }
      
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      
      weight *= (1/(2*M_PI*std_x*std_y)) * exp(-0.5 * (pow((x-x_m)/std_x, 2) + pow((y-y_m)/std_y, 2)));
    }
    p.weight = weight;
    particles[i] = p;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> particles_new;
  
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
  default_random_engine gen;
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  int index = uniintdist(gen);;
  double beta = 0.0;
  
  double max_weight = *max_element(weights.begin(), weights.end());
    
  uniform_real_distribution<double> unirealdist(0.0, max_weight);
  
  for(unsigned int i = 0; i < num_particles; i++){
    beta += unirealdist(gen) * 2.0;
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    particles_new.push_back(particles[index]);
  }
  particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
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
