"""
In this file, we create codes for the system to be simulated.
"""
from numpy import random
import matplotlib.pyplot as plt
from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import plot
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
import matplotlib as mpl

# Set a seed.
selected_seed = 100
random.seed(selected_seed)

# Codes for parameters of plotting.
mpl.rcParams.update(mpl.rcParamsDefault)
font = {'size' : 17}
mpl.rc('font', **font)

# Write a class for simulation purpose.
class Plane:

    def __init__(self, power, alpha, beta, alt, vt, phi, theta, psi, tmax, step, return_state):
        self.power = power # engine power level (0 - 10)
        self.alpha = alpha # Trim Angle of Attack (rad)
        self.beta = beta # Side slip angle (rad)
        self.alt = alt # altitude (ft)
        self.vt = vt # initial velocity (ft/sec)
        self.phi = phi # Roll angle from wings level (rad)
        self.theta = theta # Pitch angle from nose level (rad)
        self.psi = psi # Yaw angle from North (rad)
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        self.init_state = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
        self.tmax = tmax
        self.step = step
        self.return_state = return_state
        self.timestamps, self.nominal_traj = self.__generate_nominal(tmax, step, return_state) # The fixed nominal trajectory.

    def __generate_nominal(self, tmax, step, return_state):
        ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')
        # Generate the nominal trajectory.
        res = run_f16_sim(self.init_state, tmax, ap, step=step, extended_states=True)
        return plot.return_single(res, return_state)

    def simulate_nominal(self, nominal_sd = 3):
        # Add a gaussian noise to each timestep of the trajectory.
        new_traj = []
        for point in self.nominal_traj:
            new_point = random.normal(point, nominal_sd)
            new_traj.append(new_point)
        return new_traj

    def simulate_disturbed(self, shifted_distance = 0, shifted_direction = 0, disturbed_sd = 3.5):
        new_traj = []
        for point in self.nominal_traj:
            if shifted_direction: #up
                shifted_point = point + shifted_distance
            else: #down
                shifted_point = point - shifted_distance
            new_point = random.normal(shifted_point, disturbed_sd)
            new_traj.append(new_point)
        return new_traj

    def plot_simulated_trajectories(self, simulation_size = 100):
        for i in range(simulation_size):
            nominal_simulated = self.simulate_nominal()
            disturbed_simulated = self.simulate_disturbed()
            if i == 0:
                plt.plot(self.timestamps, nominal_simulated, color="blue", label = "Original Trajectories")
                plt.plot(self.timestamps, disturbed_simulated, color="orange", label = "Disturbed Trajectories")
            else:
                plt.plot(self.timestamps, nominal_simulated, color="blue")
                plt.plot(self.timestamps, disturbed_simulated, color="orange")
        plt.xlabel("Time (0.03 s)")
        plt.ylabel("Altitude (ft)")
        plt.title("Simulated Trajectories")
        plt.legend()
        plt.show()

    def plot_nominal_trajectory(self):
        x = self.simulate_nominal(nominal_sd=0)
        plt.plot(self.timestamps, x)
        plt.xlabel("Time (0.03 s)")
        plt.ylabel("Altitude (ft)")
        plt.title("Nominal Trajectory")
        plt.show()


    def plot_nominal_and_simulated_trajectories(self, simulation_size = 5):
        for i in range(simulation_size):
            nominal_simulated = self.simulate_nominal()
            disturbed_simulated = self.simulate_disturbed()
            if i == 0:
                plt.plot([i for i in range(20)], nominal_simulated[:20], color="blue", label = "Trajectories from $\mathcal{D}_0$")
                plt.plot([i for i in range(20)], disturbed_simulated[:20], color="orange", label = "Trajectories from $\mathcal{D}$")
            else:
                plt.plot([i for i in range(20)], nominal_simulated[:20], color="blue")
                plt.plot([i for i in range(20)], disturbed_simulated[:20], color="orange")
        x = self.simulate_nominal(nominal_sd=0)
        plt.plot([i for i in range(20)], x[:20], color = "green", label = "Nominal Trajectories")
        plt.xlabel("Time (0.03 s)")
        plt.ylabel("Altitude (ft)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Trajectories.pdf")
        plt.show()