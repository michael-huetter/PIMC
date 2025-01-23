#include "Beads.hpp"
#include "Energy.hpp"
#include "mcmc.hpp"
#include <iostream>
#include <fstream>

MCMC::MCMC(std::size_t num_beads, std::size_t num_particles, std::size_t simulation_dimension, 
    double temperature, std::vector<double> mass, std::size_t num_steps, double step_size_com, 
    double step_size_sbm, bool echange, std::size_t eCL, std::size_t eCG, std::size_t therm_skip, 
    std::size_t corr_skip, bool staging, std::size_t stage_length, bool virial_estimator,
    std::size_t n_estates)
    :   num_beads_(num_beads),
        num_particles_(num_particles),
        simulation_dimension_(simulation_dimension),
        temperature_(temperature),
        num_steps_(num_steps),
        step_size_com_(step_size_com),
        step_size_sbm_(step_size_sbm),
        echange_(echange),
        eCL_(eCL),
        eCG_(eCG),
        therm_skip_(therm_skip),
        corr_skip_(corr_skip),
        staging_(staging),
        stage_length_(stage_length),
        virial_estimator_(virial_estimator),
        n_estates_(n_estates)
{
    mass_ = mass;
    rejected_com_ = 0;
    rejected_sbm_ = 0;
    rejected_global_e_state_ = 0;
    rejected_local_e_state_ = 0;
    energy_trace_ = std::vector<double>();
    e_state_trace_ = std::vector<std::vector<std::size_t>>();
    if (num_beads_ == 0 || num_particles_ == 0 || simulation_dimension_ == 0) {
        throw std::invalid_argument("Number of beads, number of particles, and simulation dimension must be positive");
    }
    if (mass_.size() != num_particles_) {
        throw std::invalid_argument("Mass vector has incorrect size");
    }
}

void MCMC::write_to_csv(const std::vector<double>& array, const std::string& filename) const {
    std::ofstream file(filename);  
    if (file.is_open()) {
        for (const auto& ele : array) {
            file << ele << '\n'; 
        }
        file.close();  
    } else {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
}

std::vector<double> MCMC::get_energy_trace() const {
    return energy_trace_;
}

std::vector<std::vector<std::size_t>> MCMC::get_e_state_trace() const {
    return e_state_trace_;
}

std::vector<double> MCMC::get_position_trace() const {
    return position_trace_;
}

void MCMC::print_parameters() const {
    Beads beads(mass_, temperature_, step_size_com_, step_size_sbm_, num_beads_, num_particles_, simulation_dimension_, stage_length_, n_estates_);
    beads.print_parameters();
    std::cout << "Number of steps: " << num_steps_ << std::endl;
    std::cout << "Thermalization steps: " << therm_skip_ << std::endl;
    std::cout << "Correlation steps: " << corr_skip_ << std::endl;
    std::cout << "Staging: " << staging_ << std::endl;
    std::cout << "Virial estimator: " << virial_estimator_ << std::endl;
    std::cout << "Number of electronic states: " << n_estates_ << std::endl;
}

std::vector<std::tuple<std::string, double>> MCMC::get_acceptance_rates() const {
    std::vector<std::tuple<std::string, double>> acceptance_rates;
    double acceptance_com = 1.0 - (static_cast<double>(rejected_com_) / static_cast<double>(num_steps_));
    double acceptance_sbm = 1.0 - (static_cast<double>(rejected_sbm_) / static_cast<double>(num_steps_));
    double acceptance_global_e_state = 1.0 - (static_cast<double>(rejected_global_e_state_) / static_cast<double>(num_steps_));
    double acceptance_local_e_state = 1.0 - (static_cast<double>(rejected_local_e_state_) / static_cast<double>(num_steps_));
    acceptance_rates.push_back(std::make_tuple("Center of mass moves", acceptance_com));
    acceptance_rates.push_back(std::make_tuple("Single bead/staging moves", acceptance_sbm));
    acceptance_rates.push_back(std::make_tuple("Global electronic state moves", acceptance_global_e_state));
    acceptance_rates.push_back(std::make_tuple("Local electronic state moves", acceptance_local_e_state));
    return acceptance_rates;
}

void MCMC::run() {
    
    std::cout << "\033[1;32m";
    bool non_adiabatic_effects = false;

    Beads beads(mass_, temperature_, step_size_com_, step_size_sbm_, num_beads_, num_particles_, simulation_dimension_, stage_length_, n_estates_);

    // MCMC loop
    for (std::size_t i = 0; i < num_steps_; ++i) {
        beads.center_of_mass_move();
        staging_ ? beads.staging_move() : beads.single_bead_move();
        if (echange_ && i % eCG_ == 0) {
            beads.global_e_state_move();
        }
        if (echange_ && non_adiabatic_effects && i % eCL_ == 0) {
            beads.local_e_state_move();
        }
        if (i % corr_skip_ == 0 & i > therm_skip_) {
            double tot_energy = virial_estimator_ ? beads.compute_tot_energy_virial(beads.get_all_positions(), beads.get_all_e_states()) : beads.compute_tot_energy_thermodynamic(beads.get_all_positions(), beads.get_all_e_states());
            energy_trace_.push_back(tot_energy);
            e_state_trace_.push_back(beads.get_all_e_states());
            position_trace_.push_back(beads.pos_estimator(beads.get_all_positions(), 2, 0));
        }
        if (i % 10'000 == 0) {
            std::cout << "\r" << "MCMC Progress: " << (static_cast<double>(i) / static_cast<double>(num_steps_)) * 100 << "%" << std::flush;
        }
    }

    rejected_com_ = beads.get_rejected_com();
    rejected_sbm_ = beads.get_rejected_sbm();
    rejected_global_e_state_ = beads.get_rejected_global_e_state();
    rejected_local_e_state_ = beads.get_rejected_local_e_state();

    std::cout << "\033[0m";
}