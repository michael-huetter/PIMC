"""
Tests for the PIMC module.
"""

import PIMC
import numpy as np  
import unittest

class CustomTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.write(f"✅ PASSED: {test}\n")
        self.stream.flush()  # Ensure immediate output
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.write(f"❌ FAILED: {test}\n")
        self.stream.flush()
    def addError(self, test, err):
        super().addError(test, err)
        self.stream.write(f"⚠️ ERROR: {test}\n")
        self.stream.flush()
class CustomTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        """Create an instance of the custom TestResult."""
        return CustomTestResult(self.stream, self.descriptions, self.verbosity)


def HO3D(pos, dim):
    x, y, z = pos
    pot_mat = np.zeros((dim, dim))
    pot_mat[0, 0] = 0.5 * (x**2 + y**2 + z**2)
    pot_mat[1, 1] = 0.5 * (x**2 + y**2 + z**2) + 1
    return pot_mat

def PIMC_HO_3D(beads = 20, particles = 1, dim = 3, mass = [1.0], num_steps = 50_000, 
               step_size_com = 1.0, step_size_sbm = 0.1, echange = False, eCL = 1, eCG = 1, 
               therm_skip = 1000, corr_skip = 20, staging = True, stage_length = 18, T = 0.3, virial = False, n_estates = 2):
    
    PIMC.set_potential(HO3D)
    sim = PIMC.MCMC(beads, particles, dim, T, mass, num_steps, step_size_com, step_size_sbm, echange, eCL, eCG, therm_skip, corr_skip, staging, stage_length, virial, n_estates)
    sim.run()
    E_trace = sim.get_energy_trace()
    accept_rates = sim.get_acceptance_rates()

    return E_trace, accept_rates

def bounds(E_trace, E_exact):
    conv = np.abs(np.mean(E_trace)-E_exact)
    sigma = np.std(E_trace)
    SE = sigma/np.sqrt(len(E_trace))
    tol = 1.96*SE
    return conv, tol

class TestPIMC(unittest.TestCase):
    def test_HO_3D_classical_limit(self):  
        T = 0.3 
        E_trace, accept_rates =  PIMC_HO_3D(beads=1)
        E_exact = 3*T
        conv, tol = bounds(E_trace, E_exact)
        self.assertTrue( conv < tol, "Classical limit not reached.")
    def test_HO_3D(self):  
        T = 0.3 
        E_trace, accept_rates =  PIMC_HO_3D()
        E_exact = (3/2)/np.tanh(0.5/T)
        conv, tol = bounds(E_trace, E_exact)
        self.assertTrue( conv < tol, "Quanum limit not reached.")
    def test_HO_3D_adiabatic(self):  
        T = 3.0 
        dE = 1
        E_trace, accept_rates =  PIMC_HO_3D(T=3.0, echange=True)
        E_exact = (3/2)/np.tanh(0.5/T) + dE / (np.exp(dE/T) + 1)
        conv, tol = bounds(E_trace, E_exact)
        self.assertTrue( conv < tol, "Quanum limit not reached.")   


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPIMC)
    runner = CustomTestRunner(verbosity=2)
    result = runner.run(suite)