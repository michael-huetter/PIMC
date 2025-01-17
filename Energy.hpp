#pragma once 

#include "Potential.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cstddef>
#include <random>

class Energy {
public:
    Energy(std::vector<double> mass, double temperature, double step_size_com, double step_size_sbm, std::size_t numTimeSlices, std::size_t numParticles, std::size_t simulation_dimension);

    double compute_potential_energy(const std::vector<Eigen::MatrixXd>& positions,
                                    const std::vector<std::size_t>& e_states) const;
    double thermodynamic_estimator(const std::vector<Eigen::MatrixXd>& positions) const;
    double compute_kinetic_action(const std::vector<Eigen::MatrixXd>& positions) const;
    double compute_tot_energy_thermodynamic(const std::vector<Eigen::MatrixXd>& positions,
                                            const std::vector<std::size_t>& e_states) const;
    double compute_tot_action(const std::vector<Eigen::MatrixXd>& positions,
                              const std::vector<std::size_t>& e_states) const;
    double compute_pseudopotential(const std::vector<Eigen::MatrixXd>& positions,
                                   const std::vector<std::size_t>& e_states) const;   
    // TODO: Implement this function
    double compute_potential_energy_difference(const std::vector<Eigen::MatrixXd>& positions,
                                               const std::vector<std::size_t>& e_states,
                                               std::size_t timeSlice,
                                               std::size_t particle,
                                               const Eigen::RowVectorXd& old_position,
                                               const Eigen::RowVectorXd& new_position) const;                       
protected:
    // Simulation parameters
    std::vector<double> mass_;
    double temperature_;
    double step_size_com_;
    double step_size_sbm_;
    std::size_t numTimeSlices_;
    std::size_t numParticles_;
    std::size_t simulation_dimension_;

    // Random number generator
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_mcmc_move_;
    std::uniform_real_distribution<double> uniform_dist_metropolis_;
    std::uniform_int_distribution<std::size_t> timeSlice_dist_;
    std::uniform_int_distribution<std::size_t> particle_dist_;
    std::uniform_int_distribution<std::size_t> e_state_dist_;
    std::normal_distribution<double> normal_dist_;
private:
    // Potential matrix
    PotentialMatrix potential_matrix_;
};


