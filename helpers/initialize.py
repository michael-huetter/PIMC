import numpy as np
import PIMC

class input_parameters:
    initial_positions = None
    def __init__(
            self, 
            particles, 
            mass, 
            T, 
            beads=20, 
            dim=3, 
            num_steps=50_000, 
            step_size_com=1.0, 
            step_size_sbm=0.1, 
            echange=False, 
            eCL=1, 
            eCG=1, 
            therm_skip=1000, 
            corr_skip=20, 
            staging=False, 
            stage_length=16, 
            num_CPU=1, 
            virial=False, 
            n_estates=1
        ):
        self.beads = beads
        self.particles = particles
        self.dim = dim
        self.mass = mass
        self.num_steps = num_steps
        self.step_size_com = step_size_com
        self.step_size_sbm = step_size_sbm
        self.echange = echange
        self.eCL = eCL
        self.eCG = eCG
        self.therm_skip = therm_skip
        self.corr_skip = corr_skip
        self.staging = staging
        self.stage_length = stage_length
        self.T = T
        self.num_CPU = num_CPU
        self.virial = virial
        self.n_estates = n_estates
    def initial_positions(self, particle_coordinates):
        self.initial_positions = np.random.rand(self.beads, self.particles, self.dim) / 3
        for i in range(self.beads):
            for j in range(self.particles):
                    self.initial_positions[i][j] += particle_coordinates[j]
    def initialize_simulation(self):
        sim = PIMC.MCMC(
            self.beads, 
            self.particles, 
            self.dim, 
            self.T, 
            self.mass, 
            self.num_steps, 
            self.step_size_com, 
            self.step_size_sbm, 
            self.echange, 
            self.eCL, 
            self.eCG, 
            self.therm_skip, 
            self.corr_skip, 
            self.staging, 
            self.stage_length, 
            self.virial, 
            self.n_estates
        )
        sim.set_initial_positions(self.initial_positions)
        return sim
    def set_potential(self, potential):
        PIMC.set_potential(potential)


        


    