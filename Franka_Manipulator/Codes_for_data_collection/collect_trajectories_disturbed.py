"""
References used:
https://www.w3schools.com/python/ref_random_seed.asp
official tutorial for isaac sim
https://www.w3schools.com/python/ref_random_uniform.asp
https://www.geeksforgeeks.org/os-module-python-examples/
https://www.geeksforgeeks.org/how-to-convert-python-dictionary-to-json/
"""


from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import json

# Declare global variables for the property of the system.
start_position = [0.52, 0.52, 0]
position_std = 0.1
end_position = [-0.5, -0.5, 0]


def compute_norm(x):
    return (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)


def save_trajectories(trajectories):
    # Change directories: Please set this to your own directory if you are using my codes.
    print(os.getcwd())
    os.chdir("C:/Users/nickz/Documents/Research/stl_runtime_verification_under_distribution_shift/Isaac_Sim_Case_Study/Data")
    # Now, save the json.
    # Add system property discriptions:
    trajectories["properties"] = {"start": start_position, "end": end_position, "start_position_std": position_std}
    with open("disturbed_10.json", "w") as outfile:
        json.dump(trajectories, outfile)


def process_raw_trajectories(trajectories, num_samples):
    processed_trajectories = dict()
    # First, extract the speed, which is irrelevant of the coordinate.
    for i in range(num_samples):
        processed_trajectories[i] = dict()
        processed_trajectories[i]["speed"] = [float(compute_norm(trajectories[i]["velocity"][j])) for j in range(len(trajectories[i]["velocity"]))]
        processed_trajectories[i]["position"] = [[float(trajectories[i]["position"][j][0] - 2 * i), float(trajectories[i]["position"][j][1]), float(trajectories[i]["position"][j][2])] for j in range(len(trajectories[i]["position"]))]
    return processed_trajectories


def generate_starting_positions(num_samples):
    return [[random.uniform(start_position[0] - position_std, start_position[0] + position_std), random.uniform(start_position[1] - position_std, start_position[1] + position_std), start_position[2]] for i in range(num_samples)]


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        # Set parameters here:
        self.num_samples = 100
        self.starting_positions = generate_starting_positions(self.num_samples)
        self.trajectories = dict()
        # Add robots:
        self.frankas = dict()
        for i in range(self.num_samples):
            self.frankas[i] = world.scene.add(Franka(prim_path=f"/World/Fancy_Franka{i}", name=f"fancy_franka{i}", position = [2 * i, 0, 0]))

        # Add cubes and tables:
        self.cubes = dict()
        self.tables = dict()
        self.goal_positions = dict()
        for i in range(self.num_samples):
            self.cubes[i] = world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/random_cube{i}",
                    name=f"fancy_cube{i}",
                    position=np.array([self.starting_positions[i][0] + 2 * i, self.starting_positions[i][1], self.starting_positions[i][2]]),
                    scale=np.array([0.03, 0.03, 0.03]),
                    color=np.array([0, 0, 1.0])
                )
            )
            self.goal_positions[i] = np.array([end_position[0] + 2 * i, end_position[1], end_position[2]])
            self.trajectories[i] = dict()
            self.trajectories[i]["position"] = []
            self.trajectories[i]["velocity"] = []
            self.trajectories[i]["orientation"] = []
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._francas = dict()
        self._fancy_cubes = dict()
        self._controllers = dict()
        for i in range(self.num_samples):
            self._francas[i] = self._world.scene.get_object(f"fancy_franka{i}")
            self._fancy_cubes[i] = self._world.scene.get_object(f"fancy_cube{i}")
            # Initialize a pick and place controller
            self._controllers[i] = PickPlaceController(
                name="pick_place_controller",
                gripper=self._francas[i].gripper,
                robot_articulation=self._francas[i]
            )
        self._world.add_physics_callback(f"sim_step", callback_fn=self.physics_step)
        for i in range(self.num_samples):
            # World has pause, stop, play..etc
            # Note: if async version exists, use it in any async function is this workflow
            self._francas[i].gripper.set_joint_positions(self._francas[i].gripper.joint_opened_positions)
        await self._world.play_async()
        return

    # This function is called after Reset button is pressed
    # Resetting anything in the world should happen here
    async def setup_post_reset(self):
        for i in range(self.num_samples):
            self._controllers[i].reset()
            self._francas[i].gripper.set_joint_positions(self._francas[i].gripper.joint_opened_positions)
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        for i in range(self.num_samples):
            cube_position, cube_orientation = self._fancy_cubes[i].get_world_pose()
            current_joint_positions = self._francas[i].get_joint_positions()
            velocity = self._fancy_cubes[i].get_linear_velocity()
            # Change trajectories.
            self.trajectories[i]["position"].append(cube_position)
            self.trajectories[i]["velocity"].append(velocity)
            self.trajectories[i]["orientation"].append(cube_orientation)
            actions = self._controllers[i].forward(
                picking_position=cube_position,
                placing_position=self.goal_positions[i],
                current_joint_positions=current_joint_positions,
            )
            self._francas[i].apply_action(actions)
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        if self._controllers[0].is_done():
            self._world.pause()
            # Process raw trajectories into the same coordinate.
            processed_trajectories = process_raw_trajectories(self.trajectories, self.num_samples)
            # Save trajectories.
            save_trajectories(processed_trajectories)
        return