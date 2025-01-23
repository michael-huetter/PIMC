#pragma once

#include <vector>
#include <cstddef>
#include <Eigen/Dense>
#include <random>
#include "Energy.hpp"

class Beads : public Energy {
public:
    Beads(std::vector<double> mass, double temperature, double step_size_com, double step_size_sbm, 
          std::size_t numTimeSlices, std::size_t numParticles, std::size_t simulation_dimension, 
          std::size_t stage_length, std::size_t n_estates);

    std::size_t get_num_time_slices() const;
    std::size_t get_num_particles() const;
    std::size_t get_simulation_dimension() const;

    const Eigen::MatrixXd& get_positions(std::size_t timeSlice) const;
    double pos_estimator(const std::vector<Eigen::MatrixXd>& positions, std::size_t dim, std::size_t ptcl) const;
    void set_positions(std::size_t timeSlice, const Eigen::MatrixXd& positions);
    const std::vector<Eigen::MatrixXd>& get_all_positions() const;

    std::size_t get_e_state(std::size_t timeSlice) const;
    void set_e_state(std::size_t timeSlice, std::size_t e_state);
    const std::vector<std::size_t>& get_all_e_states() const;

    // MCMC update Methods
    void center_of_mass_move();
    void single_bead_move();
    void staging_move();
    void global_e_state_move();
    void local_e_state_move();

    // Helper functions
    void print_parameters() const;
    std::size_t get_rejected_com() const;
    std::size_t get_rejected_sbm() const;
    std::size_t get_rejected_global_e_state() const;
    std::size_t get_rejected_local_e_state() const;

protected:
    std::vector<Eigen::MatrixXd> positions_;
    std::vector<std::size_t> e_states_;
    std::size_t rejected_com_;
    std::size_t rejected_sbm_;
    std::size_t rejected_global_e_state_;
    std::size_t rejected_local_e_state_;
    std::size_t stage_length_;
    std::size_t n_estates_;
};