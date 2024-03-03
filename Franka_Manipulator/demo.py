"""
In this file, we implement the codes for verifying the Franka Manipulator using the direct method.
"""

"""
Import necessary libraries.
"""
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import casadi
import cvxpy as cp
import matplotlib as mpl
import os

mpl.rcParams.update(mpl.rcParamsDefault)
font = {'size' : 17}
mpl.rc('font', **font)

import pylab as pl
params = {'legend.fontsize': 12}
pl.rcParams.update(params)


# Set a seed.
my_seed = 10
random.seed(my_seed)

def f(t):
    # We assume to use the total variation distance.
    return 0.5 * abs(t - 1)


def g(epsilon, beta, search_step=0.0007):
    # Check input.
    if beta < 0 or beta > 1:
        raise Exception("Input to the function g is out of range.")

    # Perform a sampling-based line search.
    z = 0
    while z <= 1:
        value = beta * f(z / beta) + (1 - beta) * f((1 - z) / (1 - beta))
        if value <= epsilon:
            return z
        z += search_step

    raise Exception("No return from function g.")


def g_inverse(epsilon, tau, search_step=0.0007):
    # Check input.
    if tau < 0 or tau > 1:
        raise Exception("Input to the function g_inverse is out of range.")

    beta = 1
    while beta >= 0:
        if beta != 1 and g(epsilon, beta) <= tau:
            return beta
        beta -= search_step

    raise Exception("No return from function g_inverse.")


def calculate_delta_n(delta, n, epsilon):
    inner = (1 + 1 / n) * g_inverse(epsilon, 1 - delta)
    return (1 - g(epsilon, inner))


def calculate_delta_tilde(delta_n, epsilon):
    answer = 1 - g_inverse(epsilon, 1 - delta_n)
    return answer


# Other Helper functions defined here.
def restructure_trajectories(trajectories):
    restructured = []
    for trajectory in trajectories:
        restructured_traj = []
        for t in range(len(trajectory["position"])):
            temp = [trajectory["position"][t][i] for i in [0, 1, 2]]
            temp.append(trajectory["speed"][t])
            restructured_traj.append(temp)
        restructured.append(restructured_traj)
    return restructured


def split_data_for_x_y(trajectories, observed_range, predicted_range):
    x = []
    y = []
    for trajectory in trajectories:
        x.append(trajectory[observed_range[0]: observed_range[1] + 1])
        y.append(trajectory[predicted_range[0]: predicted_range[1] + 1])
    return x, y


def split_list(my_list, chunk_size):
    start = 0
    end = len(my_list)
    step = chunk_size
    splitted = []
    for i in range(start, end, step):
        x = i
        splitted.append(my_list[x:x + step])
    return splitted


def extract_dimension(trajectory, dimension):
    return [trajectory[t][dimension] for t in range(len(trajectory))]


def test_model(splitted_test_trajectories, dimension, model, title):
    print(f"Start: Generating predictions for test data: {title}")
    test_x, test_y = splitted_test_trajectories[dimension]
    predicted = model.predict(test_x)
    print(f"End: Generating predictions for test data: {title}")
    print()
    return test_x, test_y, predicted


def train_model(splitted_trajectories, dimension, title):
    print(f"Start: Training a predictor for dimension {dimension}: {title}")
    train_x, calib_x, train_y, calib_y = splitted_trajectories[dimension]
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(30, input_shape=(len(train_x[0]), 1)))
    model.add(Dense(50))
    model.add(Dense(20))
    model.add(Dense(len(train_y[0])))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, epochs=10, batch_size=1, verbose=2)
    print(f"End: Training a predictor for {dimension}: {title}")
    print()

    print("Testing a sample from the calibration set.")
    calib_predicted = model.predict(calib_x)
    train_predicted = model.predict(train_x)
    sample_indices = [0, 1, 2]
    for sample_index in sample_indices:
        plt_ground = np.concatenate((calib_x[sample_index], calib_y[sample_index]))
        plt_predicted = np.concatenate((calib_x[sample_index], calib_predicted[sample_index]))
        time = [t for t in range(len(plt_ground))]
        plt.plot(time, plt_ground, label="Ground")
        plt.plot(time, plt_predicted, label="Predicted")
        plt.title(f"Sample {sample_index} Prediction from Trajectory Predictor on {title}")
        plt.legend()
        plt.show()

    return model, train_x, calib_x, train_y, calib_y, train_predicted, calib_predicted


def reassemble_x_y_trajectories(trajectories_x, trajectories_y):
    reassembled = []
    for i in range(len(trajectories_x)):
        reassembled.append(np.concatenate((trajectories_x[i], trajectories_y[i])))
    return reassembled


def recover_trajectories(x0, y0, x1, y1, x2, y2, x3, y3):
    whole0 = reassemble_x_y_trajectories(x0, y0)
    whole1 = reassemble_x_y_trajectories(x1, y1)
    whole2 = reassemble_x_y_trajectories(x2, y2)
    whole3 = reassemble_x_y_trajectories(x3, y3)

    trajectories = []
    for i in range(len(whole0)):
        positions = []
        speeds = []
        new_traj = dict()
        for j in range(len(whole0[i])):
            point = [whole0[i][j], whole1[i][j], whole2[i][j]]
            positions.append(point)
            speeds.append(whole3[i][j])
        new_traj["position"] = positions
        new_traj["speed"] = speeds
        trajectories.append(new_traj)
    return trajectories


def dist(x, y):
    # Calculate the L2 distance between 2 points.
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2) ** (1 / 2)


def dist_4d(x, y):
    # Calculate the L2 distance between 2 points in 4d.
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2 + (x[3] - y[3]) ** 2) ** (1 / 2)


def compute_robustness(trajectory, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position):
    """
    Specification: {F_[0, T] dist(p(t), goal) <= epsilon_1} and  {F_[0, T] dist(p(t), human) <= epsilon_3} and {G_[0, T] ([dist(p(t) , human) < epsilon_3] => s(t) <= epsilon_2)}

    Simplification:
    -->            min(max_[0, T] [epsilon_1 - dist(p(t), goal)], max_[0, T] [epsilon_3 - dist(p(t), human)], min_[0, T] (max(dist(p(t), human) - epsilon_3, epsilon_2 - s(t))))
    """
    positions = trajectory["position"]
    speeds = trajectory["speed"]
    robustness_1_values = []
    robustness_2_values = []
    robustness_3_values = []
    for t in range(len(positions)):
        robustness_1_values.append(epsilon_1 - dist(positions[t], goal_position))
        robustness_2_values.append(epsilon_3 - dist(positions[t], human_position))
        robustness_3_values.append(max(dist(positions[t], human_position) - epsilon_3, epsilon_2 - speeds[t]))
    robustness_1_value = max(robustness_1_values)
    robustness_2_value = max(robustness_2_values)
    robustness_3_value = min(robustness_3_values)
    return min(robustness_1_value, robustness_2_value, robustness_3_value)


def compute_robustness_for_plotting(trajectory, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position):
    positions = [point[:3] for point in trajectory]
    speeds = [point[3] for point in trajectory]
    robustness_1_values = []
    robustness_2_values = []
    robustness_3_values = []
    for t in range(len(positions)):
        robustness_1_values.append(epsilon_1 - dist(positions[t], goal_position))
        robustness_2_values.append(epsilon_3 - dist(positions[t], human_position))
        robustness_3_values.append(max(dist(positions[t], human_position) - epsilon_3, epsilon_2 - speeds[t]))
    robustness_1_value = max(robustness_1_values)
    robustness_2_value = max(robustness_2_values)
    robustness_3_value = min(robustness_3_values)
    return min(robustness_1_value, robustness_2_value, robustness_3_value)


def compute_phi1_for_plotting(trajectory, epsilon_1, goal_position):
    positions = [point[:3] for point in trajectory]
    robustness_1_values = []
    for t in range(len(positions)):
        robustness_1_values.append(epsilon_1 - dist(positions[t], goal_position))
    return max(robustness_1_values)


def compute_phi2_for_plotting(trajectory, epsilon_3, human_position):
    positions = [point[:3] for point in trajectory]
    robustness_2_values = []
    for t in range(len(positions)):
        robustness_2_values.append(epsilon_3 - dist(positions[t], human_position))
    return max(robustness_2_values)


def compute_phi3_for_plotting(trajectory, epsilon_2, epsilon_3, human_position):
    positions = [point[:3] for point in trajectory]
    speeds = [point[3] for point in trajectory]
    robustness_3_values = []
    for t in range(len(positions)):
        robustness_3_values.append(max(dist(positions[t], human_position) - epsilon_3, epsilon_2 - speeds[t]))
    return min(robustness_3_values)

def convert_to_tuple_form(trajectories):
    new_trajs = []
    for trajectory in trajectories:
        traj_pos = trajectory["position"]
        traj_s = trajectory["speed"]
        new_trajectory = []
        for tau in range(len(trajectory["position"])):
            traj_tuple = [traj_pos[tau][0], traj_pos[tau][1], traj_pos[tau][2], traj_s[tau]]
            new_trajectory.append(traj_tuple)
        new_trajs.append(new_trajectory)
    return new_trajs


def run_direct_vanilla_runtime_verification(delta, calib_ground_whole_trajectories, calib_pred_whole_trajectories,
                                            test_ground_whole_trajectories, test_pred_whole_trajectories, epsilon_1,
                                            epsilon_2, epsilon_3, goal_position, human_position):
    # Compute nonconformities.
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        nonconformity = compute_robustness(calib_pred_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3,
                                           goal_position, human_position) - compute_robustness(
            calib_ground_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3, goal_position, human_position)
        nonconformity_list.append(nonconformity)
    nonconformity_list.append(float("inf"))
    # Sort the nonconformities and find c.
    nonconformity_list.sort()

    p = int(np.ceil((len(calib_ground_whole_trajectories) + 1) * (1 - delta)))
    c = nonconformity_list[p - 1]

    # Compute the coverage.
    coverage_count = 0
    ground_robustnesses = []
    predicted_robustnesses = []
    lower_bounds = []
    for i in range(len(test_ground_whole_trajectories)):
        robustness_ground = compute_robustness(test_ground_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3,
                                               goal_position, human_position)
        robustness_predicted = compute_robustness(test_pred_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3,
                                                  goal_position, human_position)
        ground_robustnesses.append(robustness_ground)
        predicted_robustnesses.append(robustness_predicted)
        lower_bounds.append(robustness_predicted - c)
        if robustness_ground >= (robustness_predicted - c):
            coverage_count += 1

    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c, ground_robustnesses, lower_bounds


def run_direct_robust_runtime_verification(epsilon, delta, calib_ground_whole_trajectories,
                                           calib_pred_whole_trajectories, test_ground_whole_trajectories,
                                           test_pred_whole_trajectories, epsilon_1, epsilon_2, epsilon_3, goal_position,
                                           human_position):
    # Compute nonconformities.
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        nonconformity = compute_robustness(calib_pred_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3,
                                           goal_position, human_position) - compute_robustness(
            calib_ground_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3, goal_position, human_position)
        nonconformity_list.append(nonconformity)
    nonconformity_list.sort()

    # Calculate delta_tilde.
    delta_n = calculate_delta_n(delta, len(calib_ground_whole_trajectories), epsilon)
    delta_tilde = calculate_delta_tilde(delta_n, epsilon)
    # Compute c_tilde.
    p = int(np.ceil((len(calib_ground_whole_trajectories)) * (1 - delta_tilde)))
    c_tilde = nonconformity_list[p - 1]

    # Compute the coverage.
    coverage_count = 0
    ground_robustnesses = []
    predicted_robustnesses = []
    lower_bounds = []
    for i in range(len(test_ground_whole_trajectories)):
        robustness_ground = compute_robustness(test_ground_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3,
                                               goal_position, human_position)
        robustness_predicted = compute_robustness(test_pred_whole_trajectories[i], epsilon_1, epsilon_2, epsilon_3,
                                                  goal_position, human_position)
        ground_robustnesses.append(robustness_ground)
        predicted_robustnesses.append(robustness_predicted)
        lower_bounds.append(robustness_predicted - c_tilde)
        if robustness_ground >= (robustness_predicted - c_tilde):
            coverage_count += 1

    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c_tilde, ground_robustnesses, lower_bounds


def find_epsilon_from_nonconformity_3(hybrid_alphas, predicted_range, d_0_ground_whole, d_0_pred_whole, d_ground_whole, d_pred_whole, epsilon_1,
                                    epsilon_2, epsilon_3, goal_position, human_position, kde_calculation_bin_num=200000,
                                    kde_plot_bin_num=1000, smoothing_term = 0.00001):
    # Compute nonconformities for each predicate.
    d_0_nonconformity_list = []
    for i in range(len(d_0_ground_whole)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            for p in range(4):
                if p == 0:
                    nonconformity = ((epsilon_1 - dist(d_0_pred_whole[i]["position"][tau],
                                                       goal_position)) - (
                                             epsilon_1 - dist(d_0_ground_whole[i]["position"][tau],
                                                              goal_position))) / (
                                                hybrid_alphas[p][tau] + smoothing_term)
                elif p == 1:
                    nonconformity = ((epsilon_3 - dist(d_0_pred_whole[i]["position"][tau],
                                                       human_position)) - (
                                             epsilon_3 - dist(d_0_ground_whole[i]["position"][tau],
                                                              human_position))) / (
                                                hybrid_alphas[p][tau] + smoothing_term)
                elif p == 2:
                    nonconformity = ((dist(d_0_pred_whole[i]["position"][tau],
                                           human_position) - epsilon_3) - (
                                             dist(d_0_ground_whole[i]["position"][tau],
                                                  human_position) - epsilon_3)) / (
                                                hybrid_alphas[p][tau] + smoothing_term)
                else:
                    nonconformity = ((epsilon_2 - d_0_pred_whole[i]["speed"][tau]) - (
                            epsilon_2 - d_0_ground_whole[i]["speed"][tau])) / (
                                                hybrid_alphas[p][tau] + smoothing_term)
                local_nonconformity_list.append(nonconformity)
        d_0_nonconformity_list.append(max(local_nonconformity_list))

    # Compute nonconformities for each predicate.
    d_nonconformity_list = []
    for i in range(len(d_ground_whole)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            for p in range(4):
                if p == 0:
                    nonconformity = ((epsilon_1 - dist(d_pred_whole[i]["position"][tau],
                                                       goal_position)) - (
                                             epsilon_1 - dist(d_ground_whole[i]["position"][tau],
                                                              goal_position))) / (
                                            hybrid_alphas[p][tau] + smoothing_term)
                elif p == 1:
                    nonconformity = ((epsilon_3 - dist(d_pred_whole[i]["position"][tau],
                                                       human_position)) - (
                                             epsilon_3 - dist(d_ground_whole[i]["position"][tau],
                                                              human_position))) / (
                                            hybrid_alphas[p][tau] + smoothing_term)
                elif p == 2:
                    nonconformity = ((dist(d_pred_whole[i]["position"][tau],
                                           human_position) - epsilon_3) - (
                                             dist(d_ground_whole[i]["position"][tau],
                                                  human_position) - epsilon_3)) / (
                                            hybrid_alphas[p][tau] + smoothing_term)
                else:
                    nonconformity = ((epsilon_2 - d_pred_whole[i]["speed"][tau]) - (
                            epsilon_2 - d_ground_whole[i]["speed"][tau])) / (
                                            hybrid_alphas[p][tau] + smoothing_term)
                local_nonconformity_list.append(nonconformity)
        d_nonconformity_list.append(max(local_nonconformity_list))

    lower_bound = np.min(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    upper_bound = np.max(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    step_size = (upper_bound - lower_bound) / kde_plot_bin_num
    score_x_list = [x for x in np.arange(lower_bound, upper_bound + step_size, step_size)]
    kde_d_0 = gaussian_kde(d_0_nonconformity_list)
    kde_d = gaussian_kde(d_nonconformity_list)
    d_0_nonconformity_kde_list = kde_d_0.evaluate(score_x_list)
    d_nonconformity_kde_list = kde_d.evaluate(score_x_list)

    # Now, compute the total variation.
    step_size = (upper_bound - lower_bound) / kde_calculation_bin_num
    new_score_list = np.arange(lower_bound, upper_bound + step_size, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def find_epsilon_from_nonconformity_2(alphas, predicted_range, d_0_ground_whole, d_0_pred_whole, d_ground_whole, d_pred_whole, kde_calculation_bin_num=200000,
                                    kde_plot_bin_num=1000, smoothing_term = 0.00001):
    # Reformat the signals into tuple formats.
    d_0_ground_whole = convert_to_tuple_form(d_0_ground_whole)
    d_ground_whole = convert_to_tuple_form(d_ground_whole)
    d_0_pred_whole = convert_to_tuple_form(d_0_pred_whole)
    d_pred_whole = convert_to_tuple_form(d_pred_whole)

    d_0_nonconformity_list = []
    d_nonconformity_list = []
    for i in range(len(d_0_ground_whole)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            x_ground = d_0_ground_whole[i][tau]
            x_predicted = d_0_pred_whole[i][tau]
            local_nonconformity = dist_4d(x_ground, x_predicted) / (alphas[tau] + smoothing_term)
            local_nonconformity_list.append(local_nonconformity)
        d_0_nonconformity_list.append(max(local_nonconformity_list))

    for i in range(len(d_ground_whole)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            x_ground = d_ground_whole[i][tau]
            x_predicted = d_pred_whole[i][tau]
            local_nonconformity = dist_4d(x_ground, x_predicted) / (alphas[tau] + smoothing_term)
            local_nonconformity_list.append(local_nonconformity)
        d_nonconformity_list.append(max(local_nonconformity_list))
    lower_bound = np.min(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    upper_bound = np.max(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    step_size = (upper_bound - lower_bound) / kde_plot_bin_num
    score_x_list = [x for x in np.arange(lower_bound, upper_bound + step_size, step_size)]
    kde_d_0 = gaussian_kde(d_0_nonconformity_list)
    kde_d = gaussian_kde(d_nonconformity_list)
    d_0_nonconformity_kde_list = kde_d_0.evaluate(score_x_list)
    d_nonconformity_kde_list = kde_d.evaluate(score_x_list)

    # Now, compute the total variation.
    step_size = (upper_bound - lower_bound) / kde_calculation_bin_num
    new_score_list = np.arange(lower_bound, upper_bound + step_size, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def find_epsilon_from_nonconformity_1(d_0_ground_whole, d_0_pred_whole, d_ground_whole, d_pred_whole, epsilon_1,
                                    epsilon_2, epsilon_3, goal_position, human_position, kde_calculation_bin_num=200000,
                                    kde_plot_bin_num=1000):
    d_0_nonconformity_list = []
    d_nonconformity_list = []
    for i in range(len(d_0_ground_whole)):
        nonconformity = compute_robustness(d_0_pred_whole[i], epsilon_1, epsilon_2, epsilon_3, goal_position,
                                           human_position) - compute_robustness(d_0_ground_whole[i], epsilon_1,
                                                                                epsilon_2, epsilon_3, goal_position,
                                                                                human_position)
        d_0_nonconformity_list.append(nonconformity)
    for i in range(len(d_ground_whole)):
        nonconformity = compute_robustness(d_pred_whole[i], epsilon_1, epsilon_2, epsilon_3, goal_position,
                                           human_position) - compute_robustness(d_ground_whole[i], epsilon_1, epsilon_2,
                                                                                epsilon_3, goal_position,
                                                                                human_position)
        d_nonconformity_list.append(nonconformity)

    lower_bound = np.min(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    upper_bound = np.max(np.concatenate((d_0_nonconformity_list, d_nonconformity_list)))
    step_size = (upper_bound - lower_bound) / kde_plot_bin_num
    score_x_list = [x for x in np.arange(lower_bound, upper_bound + step_size, step_size)]
    kde_d_0 = gaussian_kde(d_0_nonconformity_list)
    kde_d = gaussian_kde(d_nonconformity_list)
    d_0_nonconformity_kde_list = kde_d_0.evaluate(score_x_list)
    d_nonconformity_kde_list = kde_d.evaluate(score_x_list)

    # Visualize the effects of kde.
    plt.hist(d_0_nonconformity_list, bins=np.arange(lower_bound, upper_bound + 0.0001, 0.0001))
    plt.title("Empirical Distribution of D_0 Nonconformity Scores")
    plt.show()

    plt.hist(d_nonconformity_list, bins=np.arange(lower_bound, upper_bound + 0.0001, 0.0001))
    plt.title("Empircal Distribution of D Nonconformity Scores")
    plt.show()

    plt.scatter(score_x_list, d_0_nonconformity_kde_list)
    plt.title("Estimated Distribution of D_0 Nonconformity Scores")
    plt.show()

    plt.scatter(score_x_list, d_nonconformity_kde_list)
    plt.title("Estimated Distribution of D Nonconformity Scores")
    plt.show()

    # Now, compute the total variation.
    step_size = (upper_bound - lower_bound) / kde_calculation_bin_num
    new_score_list = np.arange(lower_bound, upper_bound + step_size, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def plot_robustnesses(original_trajectories, disturbed_trajectories, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position):
    original_rhos = [compute_robustness_for_plotting(trajectory, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position) for
        trajectory in original_trajectories]
    disturbed_rhos = [
        compute_robustness_for_plotting(trajectory, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position) for
        trajectory in disturbed_trajectories]
    plt.hist(original_rhos, bins = np.arange(min(original_rhos), max(original_rhos) + 0.001, 0.001))
    plt.tight_layout()
    plt.savefig("robustness_original.pdf")
    plt.show()

    plt.hist(disturbed_rhos, bins = np.arange(min(disturbed_rhos), max(disturbed_rhos) + 0.001, 0.001))
    plt.tight_layout()
    plt.savefig("robustness_disturbed.pdf")
    plt.show()


def make_trajectory_plot(calib_ground_whole_trajectories_all, calib_pred_whole_trajectories_all, test_ground_whole_trajectories_all, test_pred_whole_trajectories_all, human_position):
    # Take 5 trajectories from each.
    calib_ground = [trajectory["position"] for trajectory in calib_ground_whole_trajectories_all[:5]]
    calib_pred = [trajectory["position"] for trajectory in calib_pred_whole_trajectories_all[:5]]
    test_ground = [trajectory["position"] for trajectory in test_ground_whole_trajectories_all[:5]]
    test_pred = [trajectory["position"] for trajectory in test_pred_whole_trajectories_all[:5]]

    # Extract dimensions
    original_xs = [extract_dimension(trajectory, 0) for trajectory in calib_ground]
    original_ys = [extract_dimension(trajectory, 1) for trajectory in calib_ground]
    original_zs = [extract_dimension(trajectory, 2) for trajectory in calib_ground]
    original_pred_xs = [extract_dimension(trajectory, 0) for trajectory in calib_pred]
    original_pred_ys = [extract_dimension(trajectory, 1) for trajectory in calib_pred]
    original_pred_zs = [extract_dimension(trajectory, 2) for trajectory in calib_pred]

    disturbed_xs = [extract_dimension(trajectory, 0) for trajectory in test_ground]
    disturbed_ys = [extract_dimension(trajectory, 1) for trajectory in test_ground]
    disturbed_zs = [extract_dimension(trajectory, 2) for trajectory in test_ground]
    disturbed_pred_xs = [extract_dimension(trajectory, 0) for trajectory in test_pred]
    disturbed_pred_ys = [extract_dimension(trajectory, 1) for trajectory in test_pred]
    disturbed_pred_zs = [extract_dimension(trajectory, 2) for trajectory in test_pred]

    # Plot for original trajectories:
    ax = plt.figure().add_subplot(projection='3d')
    colors = ["r", "g", "b", "k", "c"]
    for i in range(len(original_xs)):
        ax.plot(original_xs[i], original_ys[i], original_zs[i], color = colors[i])
        ax.plot(original_pred_xs[i], original_pred_ys[i], original_pred_zs[i], linestyle = "dashed", color = colors[i])
    ax.scatter(human_position[0], human_position[1], human_position[2], marker="x", s = 90, label = "Human Position")
    ax.set_xlim(-0.6, human_position[0] + 0.05)
    ax.set_ylim(human_position[1] - 0.05, 0.6)
    ax.set_zlim(np.amin(np.array(original_zs)), np.amax(np.array(original_zs)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("trajectories_original.pdf")
    plt.show()

    # Plot for disturbed trajectories:
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(len(disturbed_xs)):
        ax.plot(disturbed_xs[i], disturbed_ys[i], disturbed_zs[i], color = colors[i])
        ax.plot(disturbed_pred_xs[i], disturbed_pred_ys[i], disturbed_pred_zs[i], linestyle="dashed", color = colors[i])
    ax.scatter(human_position[0], human_position[1], human_position[2], marker="x", s = 90, label = "Human Position")
    ax.set_xlim(-0.6, human_position[0] + 0.05)
    ax.set_ylim(human_position[1] - 0.05, 0.6)
    ax.set_zlim(np.amin(np.array(disturbed_zs)), np.amax(np.array(disturbed_zs)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("trajectories_disturbed.pdf")
    plt.show()


def plot_trajectories(original_trajectories, disturbed_trajectories, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position):
    # Original:
    original_xs = [extract_dimension(trajectory, 0) for trajectory in original_trajectories]
    original_ys = [extract_dimension(trajectory, 1) for trajectory in original_trajectories]
    original_zs = [extract_dimension(trajectory, 2) for trajectory in original_trajectories]
    original_rhos = [compute_robustness_for_plotting(trajectory, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position) for trajectory in original_trajectories]
    original_rhos_phi1 = [compute_phi1_for_plotting(trajectory, epsilon_1, goal_position) for trajectory in original_trajectories]
    original_rhos_phi2 = [compute_phi2_for_plotting(trajectory, epsilon_3, human_position) for trajectory in original_trajectories]
    original_rhos_phi3 = [compute_phi3_for_plotting(trajectory, epsilon_2, epsilon_3, human_position) for trajectory in original_trajectories]

    # Disturbed:
    disturbed_xs = [extract_dimension(trajectory, 0) for trajectory in disturbed_trajectories]
    disturbed_ys = [extract_dimension(trajectory, 1) for trajectory in disturbed_trajectories]
    disturbed_zs = [extract_dimension(trajectory, 2) for trajectory in disturbed_trajectories]
    disturbed_rhos = [
        compute_robustness_for_plotting(trajectory, epsilon_1, epsilon_2, epsilon_3, goal_position, human_position) for
        trajectory in disturbed_trajectories]
    disturbed_rhos_phi1 = [compute_phi1_for_plotting(trajectory, epsilon_1, goal_position) for trajectory in
                          disturbed_trajectories]
    disturbed_rhos_phi2 = [compute_phi2_for_plotting(trajectory, epsilon_3, human_position) for trajectory in
                          disturbed_trajectories]
    disturbed_rhos_phi3 = [compute_phi3_for_plotting(trajectory, epsilon_2, epsilon_3, human_position) for trajectory in
                          disturbed_trajectories]

    # Plot for original trajectories:
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(len(original_trajectories)):
        ax.plot(original_xs[i], original_ys[i], original_zs[i], label= "$[\phi, \phi_1, \phi_2, \phi_3] : [{:.2f}, {:.2f}, {:.2f}, {:.2f}]$".format(original_rhos[i], original_rhos_phi1[i], original_rhos_phi2[i], original_rhos_phi3[i]))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(np.amin(np.array(original_zs)), np.amax(np.array(original_zs)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc = "upper right", fontsize = 7)
    plt.subplots_adjust(left=0, right=1, bottom=0.01, top=1)
    plt.tight_layout()
    plt.savefig("trajectories_original.pdf")
    plt.show()

    # Plot for disturbed trajectories:
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(len(disturbed_trajectories)):
        ax.plot(disturbed_xs[i], disturbed_ys[i], disturbed_zs[i],
                label="$[\phi, \phi_1, \phi_2, \phi_3] : [{:.2f}, {:.2f}, {:.2f}, {:.2f}]$".format(disturbed_rhos[i],
                                                                                                   disturbed_rhos_phi1[
                                                                                                       i],
                                                                                                   disturbed_rhos_phi2[
                                                                                                       i],
                                                                                                   disturbed_rhos_phi3[
                                                                                                       i]))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(np.amin(np.array(disturbed_zs)), np.amax(np.array(disturbed_zs)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.subplots_adjust(left=0, right=1, bottom=0.01, top=1)
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig("trajectories_disturbed.pdf")
    plt.show()


def detect_unsuccessful_trajectory(trajectory, goal_position, threshold = 0.3):
    final_position = trajectory[-1][:3]
    return dist(final_position, goal_position) >= threshold


def recover_train_trajectories(x0, y0, x1, y1, x2, y2, x3, y3):
    whole0 = reassemble_x_y_trajectories(x0, y0)
    whole1 = reassemble_x_y_trajectories(x1, y1)
    whole2 = reassemble_x_y_trajectories(x2, y2)
    whole3 = reassemble_x_y_trajectories(x3, y3)

    trajectories = []
    for i in range(len(whole0)):
        new_traj = []
        for j in range(len(whole0[i])):
            point = [whole0[i][j], whole1[i][j], whole2[i][j], whole3[i][j]]
            new_traj.append(point)
        trajectories.append(new_traj)
    return trajectories

def main():
    os.chdir("plots")

    # Define hyperparameters.
    observed_range = [0, 400]
    predicted_range = [401, 916]
    set_coverage_percentage = 0.9
    num_coverage_samples = 50
    delta = 0.2
    epsilon_1 = 0.2
    epsilon_2 = 0.6
    epsilon_3 = 0.9
    goal_position = [-0.5, -0.5, 0]
    human_position = [0.6, -0.6, 0]
    calib_size = 1500
    test_size = 200

    # Load data.
    print("Start: Loading data")
    # Load distribution unshifted data.
    original_trajectories = []
    for i in range(1, 21):
        with open(f"../Data/original_{i}.json", "r") as f:
            trajectories = json.load(f)
            for j in range(0, 100):
                original_trajectories.append(trajectories[str(j)])
    # Load data after distribution shift.
    disturbed_trajectories = []
    for i in range(1, 11):
        with open(f"../Data/disturbed_{i}.json", "r") as f:
            trajectories = json.load(f)
            for j in range(0, 100):
                disturbed_trajectories.append(trajectories[str(j)])

    # Restructure data.
    original_trajectories = restructure_trajectories(original_trajectories)
    disturbed_trajectories = restructure_trajectories(disturbed_trajectories)

    # Filter for unsuccessful trajectories:
    filtered_original_trajectories = []
    filtered_disturbed_trajectories = []

    for trajectory in original_trajectories:
        if not detect_unsuccessful_trajectory(trajectory, goal_position):
            filtered_original_trajectories.append(trajectory)

    for trajectory in disturbed_trajectories:
        if not detect_unsuccessful_trajectory(trajectory, goal_position):
            filtered_disturbed_trajectories.append(trajectory)

    original_trajectories = filtered_original_trajectories
    disturbed_trajectories = filtered_disturbed_trajectories
    print("Total number of original trajectories after filtering:", len(original_trajectories))
    print("Total number of disturbed trajectories after filtering:", len(disturbed_trajectories))
    # Plot data.
    # Shuffle trajectories.
    random.shuffle(original_trajectories)
    random.shuffle(disturbed_trajectories)
    # plot_trajectories(original_trajectories[:5], disturbed_trajectories[:5], epsilon_1, epsilon_2, epsilon_3, goal_position, human_position)
    plot_robustnesses(original_trajectories[:300], disturbed_trajectories[:300], epsilon_1, epsilon_2, epsilon_3, goal_position,
                      human_position)
    print("End: Loading data")
    print()

    # Train an LSTM predictor on the original trajectories.
    # Split the original data to training and calibration.
    print("Start: Splitting Data to Train Predictors")
    original_x, original_y = split_data_for_x_y(original_trajectories, observed_range, predicted_range)
    test_x, test_y = split_data_for_x_y(disturbed_trajectories, observed_range, predicted_range)
    # Train Test Split.
    train_x, calib_x, train_y, calib_y = train_test_split(original_x, original_y, test_size=set_coverage_percentage,
                                                          random_state=my_seed)
    # Split by dimensions.
    splitted_trajectories = []
    for i in range(0, 4):
        train_xi = [extract_dimension(trajectory, i) for trajectory in train_x]
        calib_xi = [extract_dimension(trajectory, i) for trajectory in calib_x]
        train_yi = [extract_dimension(trajectory, i) for trajectory in train_y]
        calib_yi = [extract_dimension(trajectory, i) for trajectory in calib_y]
        splitted_trajectories.append([train_xi, calib_xi, train_yi, calib_yi])
    splitted_test_trajectories = []
    for i in range(0, 4):
        test_xi = [extract_dimension(trajectory, i) for trajectory in test_x]
        test_yi = [extract_dimension(trajectory, i) for trajectory in test_y]
        splitted_test_trajectories.append([test_xi, test_yi])
    # Report the shape.
    for i in range(0, 4):
        print(f"Shape of train_x{i}:", np.shape(splitted_trajectories[i][0]))
        print(f"Shape of calib_x{i}:", np.shape(splitted_trajectories[i][1]))
        print(f"Shape of train_y{i}:", np.shape(splitted_trajectories[i][2]))
        print(f"Shape of calib_y{i}:", np.shape(splitted_trajectories[i][3]))
        print(f"Shape of test_x{i}:", np.shape(splitted_test_trajectories[i][0]))
        print(f"Shape of test_y{i}", np.shape(splitted_test_trajectories[i][1]))
    print("End: Splitting Data to Train Predictors")
    print()

    print("Start: Training a predictor for position prediction.")
    model0, train_x0, calib_x0, train_y0, calib_y0, train_predicted0, calib_predicted0 = train_model(splitted_trajectories, 0,
                                                                                   "Position_x")
    model1, train_x1, calib_x1, train_y1, calib_y1, train_predicted1, calib_predicted1 = train_model(splitted_trajectories, 1,
                                                                                   "Position_y")
    model2, train_x2, calib_x2, train_y2, calib_y2, train_predicted2, calib_predicted2 = train_model(splitted_trajectories, 2,
                                                                                   "Position_z")
    model3, train_x3, calib_x3, train_y3, calib_y3, train_predicted3, calib_predicted3 = train_model(splitted_trajectories, 3, "Speed")
    test_x0, test_y0, test_predicted0 = test_model(splitted_test_trajectories, 0, model0, "Position_x")
    test_x1, test_y1, test_predicted1 = test_model(splitted_test_trajectories, 1, model1, "Position_y")
    test_x2, test_y2, test_predicted2 = test_model(splitted_test_trajectories, 2, model2, "Position_z")
    test_x3, test_y3, test_predicted3 = test_model(splitted_test_trajectories, 3, model3, "Speed")
    print("End: Training a predictor for position prediction.")
    print()

    print("Start: Prepare Trajectories for Runtime Verification.")
    # Piece calibs together.
    calib_ground_whole_trajectories_all = recover_trajectories(calib_x0, calib_y0, calib_x1, calib_y1, calib_x2,
                                                               calib_y2, calib_x3, calib_y3)
    calib_pred_whole_trajectories_all = recover_trajectories(calib_x0, calib_predicted0, calib_x1, calib_predicted1,
                                                             calib_x2, calib_predicted2, calib_x3, calib_predicted3)
    # Prepare the test data.
    test_ground_whole_trajectories_all = recover_trajectories(test_x0, test_y0, test_x1, test_y1, test_x2, test_y2,
                                                              test_x3, test_y3)
    test_pred_whole_trajectories_all = recover_trajectories(test_x0, test_predicted0, test_x1, test_predicted1, test_x2,
                                                            test_predicted2, test_x3, test_predicted3)

    # Prepare the train data for alpha calculations.
    train_ground_whole_trajectories_all = recover_train_trajectories(train_x0, train_y0, train_x1, train_y1, train_x2,
                                                                     train_y2, train_x3, train_y3)
    train_pred_whole_trajectories_all = recover_train_trajectories(train_x0, train_predicted0, train_x1,
                                                                   train_predicted1, train_x2, train_predicted2,
                                                                   train_x3, train_predicted3)

    # Plot trajectories.
    make_trajectory_plot(calib_ground_whole_trajectories_all, calib_pred_whole_trajectories_all, test_ground_whole_trajectories_all, test_pred_whole_trajectories_all, human_position)

    # Shuffle the data.
    zipped = list(zip(calib_ground_whole_trajectories_all, calib_pred_whole_trajectories_all))
    random.shuffle(zipped)
    calib_ground_whole_trajectories_all, calib_pred_whole_trajectories_all = zip(*zipped)

    zipped = list(zip(test_ground_whole_trajectories_all, test_pred_whole_trajectories_all))
    random.shuffle(zipped)
    test_ground_whole_trajectories_all, test_pred_whole_trajectories_all = zip(*zipped)

    # Randomly sample 1000 states to calculate epsilon.
    calib_ground_whole_eps = calib_ground_whole_trajectories_all[:800]
    calib_pred_whole_eps = calib_pred_whole_trajectories_all[:800]
    test_ground_whole_eps = test_ground_whole_trajectories_all[:800]
    test_pred_whole_eps = test_pred_whole_trajectories_all[:800]
    print()

    print("Start: Compute Alphas for the indirect methods (New).")
    alphas = dict()
    for tau in range(predicted_range[0], predicted_range[1] + 1):
        local_alpha = []
        for i in range(len(train_ground_whole_trajectories_all)):
            local_alpha.append(
                dist_4d(train_ground_whole_trajectories_all[i][tau], train_pred_whole_trajectories_all[i][tau]))
        alphas[tau] = max(local_alpha)
    print("alphas", alphas)
    print("End: Compute Alphas for the indirect methods (New).")
    print()

    print("Start: Computing Alphas for the hybrid methods (New).")
    hybrid_alphas = dict()
    for pi in range(0, 4):
        hybrid_alphas[pi] = dict()
        if pi == 0:
            for tau in range(predicted_range[0], predicted_range[1] + 1):
                local_alpha = []
                for i in range(len(train_ground_whole_trajectories_all)):
                    rho_ground = epsilon_1 - dist(train_ground_whole_trajectories_all[i][tau][:3], goal_position)
                    rho_pred = epsilon_1 - dist(train_pred_whole_trajectories_all[i][tau][:3], goal_position)
                    local_alpha.append(abs(rho_pred - rho_ground))
                hybrid_alphas[pi][tau] = max(local_alpha)
        elif pi == 1:
            for tau in range(predicted_range[0], predicted_range[1] + 1):
                local_alpha = []
                for i in range(len(train_ground_whole_trajectories_all)):
                    rho_ground = epsilon_3 - dist(train_ground_whole_trajectories_all[i][tau][:3], human_position)
                    rho_pred = epsilon_3 - dist(train_pred_whole_trajectories_all[i][tau][:3], human_position)
                    local_alpha.append(abs(rho_pred - rho_ground))
                hybrid_alphas[pi][tau] = max(local_alpha)
        elif pi == 2:
            for tau in range(predicted_range[0], predicted_range[1] + 1):
                local_alpha = []
                for i in range(len(train_ground_whole_trajectories_all)):
                    rho_ground = dist(train_ground_whole_trajectories_all[i][tau][:3], human_position) - epsilon_3
                    rho_pred = dist(train_pred_whole_trajectories_all[i][tau][:3], human_position) - epsilon_3
                    local_alpha.append(abs(rho_pred - rho_ground))
                hybrid_alphas[pi][tau] = max(local_alpha)
        else:
            for tau in range(predicted_range[0], predicted_range[1] + 1):
                local_alpha = []
                for i in range(len(train_ground_whole_trajectories_all)):
                    rho_ground = epsilon_2 - train_ground_whole_trajectories_all[i][tau][3]
                    rho_pred = epsilon_2 - train_pred_whole_trajectories_all[i][tau][3]
                    local_alpha.append(abs(rho_pred - rho_ground))
                hybrid_alphas[pi][tau] = max(local_alpha)
    print("End: Computing Alphas for the hybrid methods (New).")
    print()

    print("Start: Finding the distribution shift.")
    shift1 = find_epsilon_from_nonconformity_1(calib_ground_whole_eps, calib_pred_whole_eps, test_ground_whole_eps,
                                               test_pred_whole_eps, epsilon_1,
                                               epsilon_2, epsilon_3, goal_position, human_position)
    shift2 = find_epsilon_from_nonconformity_2(alphas, predicted_range, calib_ground_whole_eps, calib_pred_whole_eps,
                                               test_ground_whole_eps,
                                               test_pred_whole_eps)
    shift3 = find_epsilon_from_nonconformity_3(hybrid_alphas, predicted_range, calib_ground_whole_eps,
                                               calib_pred_whole_eps, test_ground_whole_eps,
                                               test_pred_whole_eps, epsilon_1,
                                               epsilon_2, epsilon_3, goal_position, human_position)
    print("Epsilon from Direct Method:", shift1)
    print("Epsilon from Indirect Method (Variant I):", shift2)
    print("Epsilon from Indirect Method (Variant II):", shift3)
    epsilon = max(shift1, shift2, shift3)
    with open("direct_epsilon.json", "w") as outfile:
        outfile.write(str(epsilon))
    print("Used epsilon:", epsilon)
    print("End: Finding the distribution shift.")
    print()

    print(f"Start: Performing Runtime Verification with delta = {delta}")
    direct_vanilla = []
    direct_robust = []
    for i in range(num_coverage_samples):
        print("Running Sample:", i)
        # Shuffle the data.
        zipped = list(zip(calib_ground_whole_trajectories_all, calib_pred_whole_trajectories_all))
        random.shuffle(zipped)
        calib_ground_whole_trajectories_all, calib_pred_whole_trajectories_all = zip(*zipped)

        zipped = list(zip(test_ground_whole_trajectories_all, test_pred_whole_trajectories_all))
        random.shuffle(zipped)
        test_ground_whole_trajectories_all, test_pred_whole_trajectories_all = zip(*zipped)

        calib_ground_whole_trajectories = calib_ground_whole_trajectories_all[:calib_size]
        calib_pred_whole_trajectories = calib_pred_whole_trajectories_all[:calib_size]
        test_ground_whole_trajectories = test_ground_whole_trajectories_all[:test_size]
        test_pred_whole_trajectories = test_pred_whole_trajectories_all[:test_size]

        if i == 0:
            print("Final calib_ground_whole_trajectories length:", len(calib_ground_whole_trajectories))
            print("Final calib_pred_whole_trajectories length:", len(calib_pred_whole_trajectories))
            print("Final test_ground_whole_trajectories length:", len(test_ground_whole_trajectories))
            print("Final test_pred_whole_trajectories length:", len(test_pred_whole_trajectories))

        vanilla_coverage_direct, vanilla_direct_nonconformity_list, vanilla_direct_c, ground_robustnesses_vanilla_direct, lower_bound_vanilla_direct = run_direct_vanilla_runtime_verification(
            delta, calib_ground_whole_trajectories, calib_pred_whole_trajectories,
            test_ground_whole_trajectories, test_pred_whole_trajectories, epsilon_1,
            epsilon_2, epsilon_3, goal_position, human_position)
        robust_coverage_direct, direct_robust_nonconformity_list, robust_direct_c, ground_robustnesses_robust_direct, lower_bound_robust_direct  = run_direct_robust_runtime_verification(
            epsilon, delta, calib_ground_whole_trajectories,
            calib_pred_whole_trajectories, test_ground_whole_trajectories,
            test_pred_whole_trajectories, epsilon_1, epsilon_2, epsilon_3,
            goal_position, human_position)

        direct_vanilla.append(vanilla_coverage_direct)
        direct_robust.append(robust_coverage_direct)

        if i == 0:
            sorted_ground_robustnesses_vanilla_direct, sorted_lower_bound_vanilla_direct, sorted_lower_bound_robust_direct = zip(
                *sorted(zip(ground_robustnesses_vanilla_direct, lower_bound_vanilla_direct, lower_bound_robust_direct)))
            dot_sizes = [0.2 for i in range(len(sorted_ground_robustnesses_vanilla_direct))]
            plt.scatter([i for i in range(len(sorted_ground_robustnesses_vanilla_direct))],
                        sorted_ground_robustnesses_vanilla_direct, s=dot_sizes, label="$\\rho^\phi(X, \\tau_0)$")
            plt.scatter([i for i in range(len(sorted_ground_robustnesses_vanilla_direct))],
                        sorted_lower_bound_vanilla_direct, s=dot_sizes, label="$\\rho^*$ from the Direct Method")
            plt.scatter([i for i in range(len(sorted_ground_robustnesses_vanilla_direct))],
                        sorted_lower_bound_robust_direct, s=dot_sizes, label="$\\rho^*$ from the Robust Direct Method")
            plt.xlabel("Sample (Sorted on $\\rho^\phi(X, \\tau_0)$)")
            plt.ylabel("Robustness Value")
            plt.legend(markerscale=6)
            plt.tight_layout()
            plt.savefig("comparison_old_new_direct.pdf")
            plt.show()

    print("End: Performing Runtime Verification.")

    print("Start: Plotting coverages.")
    # Draw the coverage plot.
    plt.hist(direct_vanilla, bins=np.arange(min(np.concatenate((direct_vanilla, direct_robust))) - 0.1, max(np.concatenate((direct_vanilla, direct_robust))) + 0.005, 0.005), label="Direct Runtime Verification Method")
    plt.hist(direct_robust, bins=np.arange(min(np.concatenate((direct_vanilla, direct_robust))) - 0.1, max(np.concatenate((direct_vanilla, direct_robust))) + 0.005, 0.005), label="Robust Direct Runtime Verification Method")
    plt.xlabel("Coverage")
    plt.ylabel("Occurrence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("case_study_direct_methods_coverage.pdf")
    plt.show()

    # Plot for nonconformity on the last sample:
    plt.hist(vanilla_direct_nonconformity_list[:-1],
             bins=np.arange(min(vanilla_direct_nonconformity_list[:-1]), max(vanilla_direct_nonconformity_list[:-1]) + 0.001,
                            0.001))
    plt.axvline(x=vanilla_direct_c, color='b', label='$C$')
    plt.axvline(x=robust_direct_c, color='g', label='$\\tilde{C}$')
    plt.xlabel("Nonconformity Score")
    plt.ylabel("Occurrence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nonconformity_c_shift_direct_methods.pdf")
    plt.show()

    print("End: Plotting coverages.")


if __name__ == "__main__":
    main()