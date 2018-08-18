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
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	num_particles = 50;

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_t(theta, std[2]);

    for(int i = 0; i < num_particles; ++i) {
        Particle p; 

        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_t(gen);

        particles.push_back(p);
        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_t(0, std_pos[2]);

    double v_div_theta_dot = velocity / yaw_rate;
    double theta_dot_delta = yaw_rate * delta_t;

    for(int i = 0; i < num_particles; ++i) {
        Particle& p = particles[i];

        if(fabs(yaw_rate) == 0.0) {  // Yaw rate equal to 0
            p.x += velocity * delta_t * std::cos(p.theta);
            p.y += velocity * delta_t * std::sin(p.theta);
            // Theta does not change
        }
        else {                  // Yaw rate not equal to 0
            double theta_0 = p.theta;

            p.x += v_div_theta_dot * (std::sin(theta_0 + theta_dot_delta) - std::sin(theta_0));
            p.y += v_div_theta_dot * (std::cos(theta_0) - std::cos(theta_0 + theta_dot_delta));
            p.theta += theta_dot_delta;
        }

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_t(gen);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    const double gauss_norm_factor = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

    const double _2_sigx2 = 2 * std_landmark[0] * std_landmark[0];
    const double _2_sigy2 = 2 * std_landmark[1] * std_landmark[1];

    const std::vector<Map::single_landmark_s>& landmarks = map_landmarks.landmark_list;

    for (int i = 0; i < num_particles; ++i) {
        Particle& p = particles[i];
        // Calculate importance weight for current particle
        double new_weight = 1.0;

        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        for(int j = 0; j < observations.size(); ++j) {
            const LandmarkObs& obs = observations[j];
            // Transform observation into the map's frame of reference
            double xm = p.x + (std::cos(p.theta) * obs.x) - (std::sin(p.theta) * obs.y);
            double ym = p.y + (std::sin(p.theta) * obs.x) + (std::cos(p.theta) * obs.y);

            // Need to set these
            double shortest_dist = 999999.0;
            int best_index = -1;
            double ux = 0;
            double uy = 0;

            // Iterate over landmarks, find nearest landmark to observation, mark that one
            for(int k = 0; k < landmarks.size(); ++k) {
                const Map::single_landmark_s& landmark = landmarks[k];
                // Check that landmark is worth checking
                if(dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
                    double curr_dist = dist(xm, ym, landmark.x_f, landmark.y_f);
                    if(curr_dist < shortest_dist) {
                        shortest_dist = curr_dist;
                        ux = landmarks[k].x_f;
                        uy = landmarks[k].y_f;
                        best_index = landmarks[k].id_i;
                    }
                }
            }

            double exponent = -((pow((xm - ux), 2.0) / _2_sigx2) + (pow((ym - uy), 2.0) / _2_sigy2));

            new_weight *= gauss_norm_factor * exp(exponent);

            associations.push_back(best_index);
            sense_x.push_back(xm);
            sense_y.push_back(ym);
        }

        // Update particle's importance weight, multiplied through over all observations
        weights[i] = new_weight;

        SetAssociations(p, associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
	// Resamples particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles;
    std::vector<double> new_weights;
    new_particles.resize(num_particles);
    new_weights.resize(num_particles);

    std::default_random_engine gen;
    std::discrete_distribution<double> distribution(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; ++i) {
        double rand_i = distribution(gen);
        new_particles[i] = particles[rand_i];
        new_weights[i] = weights[rand_i];
    }
    particles = new_particles;
    weights = new_weights;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
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

std::vector<double> ParticleFilter::get_weights() {
    return weights;
}
