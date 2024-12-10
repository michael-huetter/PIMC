#include "Beads.hpp"
#include "Energy.hpp"
#include <stdexcept>
#include <random>
#include <iostream>

Beads::Beads(std::vector<double> mass, double temperature, double step_size_com, double step_size_sbm, std::size_t numTimeSlices, std::size_t numParticles, std::size_t simulation_dimension, std::size_t stage_length)
    :   Energy(mass, temperature, step_size_com, step_size_sbm, numTimeSlices, numParticles, simulation_dimension),
        positions_(numTimeSlices, Eigen::MatrixXd::Zero(numParticles, simulation_dimension)),
        e_states_(numTimeSlices, 0),
        stage_length_(stage_length)
{
    rejected_com_ = 0;
    rejected_sbm_ = 0;
    rejected_global_e_state_ = 0;
}

std::size_t Beads::get_num_time_slices() const {
    return numTimeSlices_;
}

std::size_t Beads::get_num_particles() const {
    return numParticles_;
}

std::size_t Beads::get_simulation_dimension() const {
    return simulation_dimension_;
}

const Eigen::MatrixXd& Beads::get_positions(std::size_t timeSlice) const {
    if (timeSlice >= numTimeSlices_) {
        throw std::out_of_range("Time slice index out of bounds");
    }
    return positions_[timeSlice];
}

void Beads::set_positions(std::size_t timeSlice, const Eigen::MatrixXd& positions) {
    if (timeSlice >= numTimeSlices_) {
        throw std::out_of_range("Time slice index out of bounds");
    }
    if (positions.rows() != numParticles_ || positions.cols() != simulation_dimension_) {
        throw std::invalid_argument("Positions matrix has incorrect dimensions");
    }
    positions_[timeSlice] = positions;
}

const std::vector<Eigen::MatrixXd>& Beads::get_all_positions() const {
    return positions_;
}

std::size_t Beads::get_e_state(std::size_t timeSlice) const {
    if (timeSlice >= numTimeSlices_) {
        throw std::out_of_range("Time slice index out of bounds");
    }
    return e_states_[timeSlice];
}

void Beads::set_e_state(std::size_t timeSlice, std::size_t e_state) {
    if (timeSlice >= numTimeSlices_) {
        throw std::invalid_argument("Time slice index out of bounds");
    }
    e_states_[timeSlice] = e_state;
}

const std::vector<std::size_t>& Beads::get_all_e_states() const {
    return e_states_;
}

double Beads::pos_estimator(const std::vector<Eigen::MatrixXd>& positions, std::size_t dim, std::size_t ptcl) const {
    double q = 0.0;
    for (std::size_t t = 0; t < numTimeSlices_; ++t) {
        q += positions[t](ptcl, dim);
    }
    return q / numTimeSlices_;
}

// Helper functions

void Beads::print_parameters() const {
    std::cout << "Number of time slices: " << numTimeSlices_ << std::endl;
    std::cout << "Number of particles: " << numParticles_ << std::endl;
    std::cout << "Simulation dimension: " << simulation_dimension_ << std::endl;
    std::cout << "Temperature: " << temperature_ << std::endl;
    std::cout << "Step size for center of mass moves: " << step_size_com_ << std::endl;
    std::cout << "Step size for single bead moves: " << step_size_sbm_ << std::endl;
    std::cout << "Stage length: " << stage_length_ << std::endl;
    std::cout << "Masses: ";
    for (const auto& m : mass_) {
        std::cout << m << " ";
    }
    std::cout << std::endl;
}
std::size_t Beads::get_rejected_com() const {
    return rejected_com_;
}
std::size_t Beads::get_rejected_sbm() const {
    return rejected_sbm_;
}
std::size_t Beads::get_rejected_global_e_state() const {
    return rejected_global_e_state_;
}

// MCMC moves

void Beads::center_of_mass_move() {
    std::vector<Eigen::MatrixXd> positions_old = positions_;
    double action_old = compute_potential_energy(positions_old, e_states_) / temperature_;

    Eigen::VectorXd displacement(simulation_dimension_);
    for (std::size_t d = 0; d < simulation_dimension_; ++d) {
        displacement(d) = uniform_dist_mcmc_move_(rng_) * step_size_com_;
    }
    for (std::size_t t = 0; t < numTimeSlices_; ++t) {
        positions_[t].rowwise() += displacement.transpose();
    }

    double action_new = compute_potential_energy(positions_, e_states_) / temperature_;
    double metropolis_ratio = std::exp(action_old - action_new);
    double random_number = uniform_dist_metropolis_(rng_);
    if (random_number > metropolis_ratio) {
        positions_ = positions_old;
        rejected_com_++;
    }
}

void Beads::single_bead_move() {
    std::vector<Eigen::MatrixXd> positions_old = positions_;
    double action_old = compute_tot_action(positions_old, e_states_);

    std::size_t timeSlice = timeSlice_dist_(rng_);
    std::size_t particle = particle_dist_(rng_);
    Eigen::VectorXd displacement(simulation_dimension_);
    for (std::size_t d = 0; d < simulation_dimension_; ++d) {
        displacement(d) = uniform_dist_mcmc_move_(rng_) * step_size_sbm_;
    }
    positions_[timeSlice].row(particle) += displacement.transpose();

    double action_new = compute_tot_action(positions_, e_states_);
    double metropolis_ratio = std::exp(action_old - action_new);
    double random_number = uniform_dist_metropolis_(rng_);
    if (random_number > metropolis_ratio) {
        positions_ = positions_old;
        rejected_sbm_++;
    }
}

// TODO: implement checks 0 < stage_length_ < numTimeSlices_
void Beads::staging_move() {
    std::size_t alpha_start = timeSlice_dist_(rng_);
    std::size_t alpha_end = (alpha_start + stage_length_) % numTimeSlices_;
    std::vector<Eigen::MatrixXd> positions_old = positions_;
    double action_old = compute_potential_energy(positions_old, e_states_) / temperature_;
    double tau = 1 / (temperature_ * numTimeSlices_);
    std::size_t ptcl = particle_dist_(rng_); // apply staging move to a random particle

    for (std::size_t a = 1; a < stage_length_; ++a) {
        std::size_t tslice = (alpha_start + a) % numTimeSlices_;
        std::size_t tslicem1 = (tslice == 0) ? (numTimeSlices_ - 1) : (tslice - 1);
        double tau1 = static_cast<double>(stage_length_ - a) * tau;
        Eigen::RowVectorXd avex = (tau1 * positions_[tslicem1].row(ptcl) + tau * positions_[alpha_end].row(ptcl)) / (tau + tau1); 
        double sigma2 = mass_[ptcl] / (1.0 / tau + 1.0 / tau1);
        Eigen::RowVectorXd noise = Eigen::RowVectorXd::NullaryExpr(avex.size(), [&]() {
            return std::sqrt(sigma2) * normal_dist_(rng_);
        });
        positions_[tslice].row(ptcl) = avex + noise;
    }

    double action_new = compute_potential_energy(positions_, e_states_) / temperature_;
    double metropolis_ratio = std::exp(action_old - action_new);
    double random_number = uniform_dist_metropolis_(rng_);
    if (random_number > metropolis_ratio) {
        positions_ = positions_old;
        rejected_sbm_++;
    }
}

void Beads::global_e_state_move() {
    std::vector<std::size_t> e_states_old = e_states_;
    double action_old = compute_potential_energy(positions_, e_states_old) / temperature_;

    std::size_t rand_e_state = e_state_dist_(rng_);
    for (std::size_t t = 0; t < numTimeSlices_; ++t) {
        e_states_[t] = rand_e_state;
    }

    double action_new = compute_potential_energy(positions_, e_states_) / temperature_;
    double metropolis_ratio = std::exp(action_old - action_new);
    double random_number = uniform_dist_metropolis_(rng_);
    if (random_number > metropolis_ratio) {
        e_states_ = e_states_old;
        rejected_global_e_state_++;
    }
}

