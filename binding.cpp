#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "mcmc.hpp"
#include "Potential.hpp"

namespace py = pybind11;

PYBIND11_MODULE(PIMC, m) {
    py::class_<MCMC>(m, "MCMC")
        .def(py::init<
            std::size_t,                // num_beads
            std::size_t,                // num_particles
            std::size_t,                // simulation_dimension
            double,                     // temperature
            std::vector<double>,        // mass    
            std::size_t,                // num_steps
            double,                     // step_size_com
            double,                     // step_size_sbm
            bool,                       // echange
            std::size_t,                // eCL
            std::size_t,                // eCG
            std::size_t,                // therm_skip
            std::size_t,                // corr_skip
            bool,                       // staging
            std::size_t                 // stage_length
        >(),
        py::arg("num_beads"),
        py::arg("num_particles"),
        py::arg("simulation_dimension"),
        py::arg("temperature"),
        py::arg("mass"),
        py::arg("num_steps"),
        py::arg("step_size_com"),
        py::arg("step_size_sbm"),
        py::arg("echange"),
        py::arg("eCL"),
        py::arg("eCG"),
        py::arg("therm_skip"),
        py::arg("corr_skip"),
        py::arg("staging"),
        py::arg("stage_length")
        )
        .def("run", &MCMC::run)
        .def("get_energy_trace", &MCMC::get_energy_trace)
        .def("get_e_state_trace", &MCMC::get_e_state_trace)
        .def("get_position_trace", &MCMC::get_position_trace)
        .def("print_parameters", &MCMC::print_parameters)
        .def("get_acceptance_rates", &MCMC::get_acceptance_rates);
    m.def("set_potential", [](py::object py_func) {
        PotentialMatrix::setComputeFunction(
            [py_func](const Eigen::RowVectorXd& position, std::size_t dim) {
                py::gil_scoped_acquire gil;
                py::object result = py_func(position, dim);
                return result.cast<Eigen::MatrixXd>();
            }
        );
    });
}