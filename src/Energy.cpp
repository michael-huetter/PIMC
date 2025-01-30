#include "Energy.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

Energy::Energy(std::vector<double> mass, double temperature, double step_size_com, double step_size_sbm, 
               std::size_t numTimeSlices, std::size_t numParticles, std::size_t simulation_dimension,
               std::size_t n_estates)
    :   temperature_(temperature),
        step_size_com_(step_size_com),
        step_size_sbm_(step_size_sbm),
        numTimeSlices_(numTimeSlices),
        numParticles_(numParticles),
        simulation_dimension_(simulation_dimension),
        rng_(std::random_device()()), // TODO: seed this
        uniform_dist_mcmc_move_(-1.0, 1.0),
        uniform_dist_metropolis_(0.0, 1.0),
        timeSlice_dist_(0, numTimeSlices - 1),
        particle_dist_(0, numParticles - 1),
        e_state_dist_(0, n_estates - 1), 
        normal_dist_(0.0, 1.0), // normal dist with mean 0 and std 1
        potential_matrix_(n_estates),
        n_estates_(n_estates)

{
    mass_ = mass;
} 

double Energy::compute_potential_energy(const std::vector<Eigen::MatrixXd>& positions,
                                        const std::vector<std::size_t>& e_states) const
{
    double total_energy = 0.0;

    for (std::size_t t = 0; t < numTimeSlices_; ++t) {
        const Eigen::MatrixXd& pos = positions[t];
        std::size_t e_state = e_states[t];
        for (std::size_t p = 0; p < numParticles_; ++p) { // TODO: not just sum over particles
            Eigen::RowVectorXd position = pos.row(p);
            Eigen::MatrixXd pot_mat = potential_matrix_.compute(position);
            total_energy += pot_mat(e_state, e_state);
        }
    }
    return total_energy/numTimeSlices_;
}

// NOTE: Currently only implemented for diatomics (particles=1)
double Energy::compute_pseudopotential(const std::vector<Eigen::MatrixXd>& positions,
                                       const std::vector<std::size_t>& e_states) const
{
    std::size_t n = 2; // TODO: create input parameter for number of electronic states
    std::vector<Eigen::MatrixXd> S(numTimeSlices_, Eigen::MatrixXd(n, n));
    for (std::size_t j = 0; j < numTimeSlices_; ++j) {
        const Eigen::MatrixXd& pos = positions[j].row(0);
        Eigen::MatrixXd V = potential_matrix_.compute(pos);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
        S[j] = es.eigenvectors();
    }
    double phi = 1.0;
    for (std::size_t i = 0; i < numTimeSlices_; ++i) {
        std::size_t currentEState = e_states[i];
        std::size_t nextEState    = e_states[(i + 1) % numTimeSlices_];
        Eigen::VectorXd current_vector = S[i].col(currentEState);
        Eigen::VectorXd next_vector    = S[(i + 1) % numTimeSlices_].col(nextEState);
        double dot_product = current_vector.dot(next_vector);
        phi *= dot_product;
    }
    return std::log(std::abs(phi+1.0e-10));
}

double Energy::thermodynamic_estimator(const std::vector<Eigen::MatrixXd>& positions) const
{
    double total_energy = 0.0;
    for (std::size_t bead = 0; bead < numTimeSlices_; ++bead) {
        std::size_t neighbour = (bead + 1) % numTimeSlices_;
        for (std::size_t particle = 0; particle < numParticles_; ++particle) {
            double norm = 0.5 * mass_[particle] * temperature_ * temperature_ * numTimeSlices_;
            Eigen::RowVectorXd delR = positions[bead].row(particle) - positions[neighbour].row(particle); //check, does order make a diff (see old python code)
            total_energy -= norm * delR.squaredNorm();
        }
    }
    return 0.5 * simulation_dimension_ * numParticles_ * numTimeSlices_ * temperature_ + total_energy;
}

double Energy::virial_estimator(const std::vector<Eigen::MatrixXd>& positions,
                                const std::vector<std::size_t>& e_states) const
{
    std::vector<Eigen::RowVectorXd> Rc(numParticles_, Eigen::RowVectorXd::Zero(simulation_dimension_));
    for (std::size_t bead = 0; bead < numTimeSlices_; ++bead) {
        for (std::size_t particle = 0; particle < numParticles_; ++particle) {
            Rc[particle] += positions[bead].row(particle);
        }
    }
    for (std::size_t particle = 0; particle < numParticles_; ++particle) {
        Rc[particle] /= numTimeSlices_;
    }

    double total_energy = 0.0;
    for (std::size_t bead = 0; bead < numTimeSlices_; ++bead) {
        for (std::size_t particle = 0; particle < numParticles_; ++particle) {
            Eigen::RowVectorXd delR = positions[bead].row(particle) - Rc[particle];
            Eigen::RowVectorXd dVdR = potential_matrix_.gradAutoDiff(positions[bead].row(particle), e_states[0], e_states[0]);
            positions[bead].row(particle); // Hardcoded for harmonic oscillator atm
            total_energy += delR.dot(dVdR);
        }
    }
    total_energy *= 0.5 / numTimeSlices_;
    double final = 0.5 * simulation_dimension_ * numParticles_ * temperature_ + total_energy;
    return final;
}

double Energy::compute_kinetic_action(const std::vector<Eigen::MatrixXd>& positions) const
{
    double total_energy = 0.0;
    for (std::size_t bead = 0; bead < numTimeSlices_; ++bead) {
        std::size_t neighbour = (bead + 1) % numTimeSlices_;
        for (std::size_t particle = 0; particle < numParticles_; ++particle) {
            double norm = 0.5 * mass_[particle] * temperature_ * temperature_ * numTimeSlices_;
            Eigen::RowVectorXd delR = positions[bead].row(particle) - positions[neighbour].row(particle);
            total_energy += norm * delR.squaredNorm();
        }
    }
    return total_energy / (temperature_); // *numTimeslices_
}

double Energy::compute_tot_energy_thermodynamic(const std::vector<Eigen::MatrixXd>& positions,
                                                const std::vector<std::size_t>& e_states) const
{
    return compute_potential_energy(positions, e_states) + thermodynamic_estimator(positions);
}
double Energy::compute_tot_energy_virial(const std::vector<Eigen::MatrixXd>& positions,
                                         const std::vector<std::size_t>& e_states) const
{
    return compute_potential_energy(positions, e_states) + virial_estimator(positions, e_states);
}

double Energy::compute_tot_action(const std::vector<Eigen::MatrixXd>& positions,
                                  const std::vector<std::size_t>& e_states) const
{
    return compute_potential_energy(positions, e_states) / temperature_ + compute_kinetic_action(positions);
}