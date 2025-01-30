#pragma once

#include <vector>
#include <functional>
#include <random>
#include <iostream>
#include "Beads.hpp"

class MCMC {
public:
    MCMC(std::size_t num_beads, std::size_t num_particles, std::size_t simulation_dimension, 
        double temperature, std::vector<double> mass, std::size_t num_steps, double step_size_com, 
        double step_size_sbm, bool echange, std::size_t eCL, std::size_t eCG, std::size_t therm_skip, 
        std::size_t corr_skip, bool staging, std::size_t stage_length, bool virial_estimator,
        std::size_t n_estates);

    void run();                
    void write_to_csv(const std::vector<double>& array, const std::string& filename) const;
    std::vector<double> get_energy_trace() const;
    std::vector<std::vector<std::size_t>> get_e_state_trace() const;
    std::vector<double> get_position_trace() const;
    void print_parameters() const;
    std::vector<std::tuple<std::string, double>> get_acceptance_rates() const;
    void set_initial_positions(const std::vector<Eigen::MatrixXd>& positions);

private:
    std::size_t num_beads_;
    std::size_t num_particles_;
    std::size_t  simulation_dimension_;
    double temperature_;
    std::vector<double> mass_;
    std::size_t  num_steps_;
    double  step_size_com_;
    double  step_size_sbm_;
    bool echange_;
    std::size_t  eCL_;
    std::size_t  eCG_;
    std::size_t  therm_skip_;
    std::size_t  corr_skip_;
    bool staging_;
    std::size_t stage_length_;
    std::vector<double> energy_trace_;
    std::vector<std::vector<std::size_t>> e_state_trace_;
    std::vector<double> position_trace_;
    std::size_t rejected_com_;
    std::size_t rejected_sbm_;
    std::size_t rejected_global_e_state_;
    std::size_t rejected_local_e_state_;
    bool virial_estimator_;
    std::size_t n_estates_;
    Beads beads_;
};