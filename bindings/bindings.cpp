#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "montecarlo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(montecarlo_engine, m) {
    m.doc() = "Optimized Montecarlo engine in C++";

    py::enum_<montecarlo::OptionType>(m, "OptionType")
        .value("Call", montecarlo::OptionType::Call)
        .value("Put", montecarlo::OptionType::Put)
        .export_values();

    py::class_<montecarlo::Result>(m, "Result")
        .def(py::init<double, double, double, double, double, double, double, std::unordered_map<int, int>>())
        .def_readonly("fairValue", &montecarlo::Result::fairValue)
        .def_readonly("CI_lower", &montecarlo::Result::CI_lower)
        .def_readonly("CI_upper", &montecarlo::Result::CI_upper)
        .def_readonly("delta", &montecarlo::Result::delta)
        .def_readonly("gamma", &montecarlo::Result::gamma)
        .def_readonly("vega", &montecarlo::Result::vega)
        .def_readonly("avg_S_T", &montecarlo::Result::avg_S_T)
        .def_readonly("counter", &montecarlo::Result::counter);

    m.def("run_montecarlo", &montecarlo::Montecarlo, "Executes the N simulations",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("S_0"), py::arg("volatility"), py::arg("K"), 
          py::arg("T"), py::arg("r"), py::arg("OT"), py::arg("N"));
}