"""
In this file, we demonstrate our algorithms using the F-16 case study (The running example from the article).

Specifically, we demonstrate the direct and indirect method for stl runtime verification under distribution shift using conformal prediction
with f-divergence. The case will be on collision avoidance (altitude always >= 60). Suppose tau_0 = 0.
"""

"""
We include any necessary libraries here.
"""
import math
from numpy import deg2rad
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
from scipy.stats import gaussian_kde
import os
from plane import *
from algorithms import *

# Set a seed.
selected_seed = 100
random.seed(selected_seed)
torch.manual_seed(selected_seed)

# Codes for parameters of plotting.
mpl.rcParams.update(mpl.rcParamsDefault)
font = {'size' : 17}
mpl.rc('font', **font)

"""
Here we provide the codes for the predictor.
"""
class Predictor(nn.Module):
    # Write a class for the predictor.
    def __init__(self, set_inputsize, set_outputsize):
        super().__init__()
        self.lstm = nn.LSTM(input_size=set_inputsize, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, set_outputsize)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def generate_data_with_predictions(norm, predictor, plane, size, observed_range, predicted_range, disturbed = False):
    # Generate data.
    simulations = []
    for i in range(size):
        if disturbed:
            simulations.append(plane.simulate_disturbed())
        else:
            simulations.append(plane.simulate_nominal())

    # Normalize the data.
    simulations /= norm

    # Slice to form the input and output data.
    x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in simulations]))
    y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in simulations]))

    predictor.eval()
    with torch.no_grad():
        pred = predictor(x)

    # Recover original calib_whole_trajectories
    ground_whole_trajectories = recover_whole_trajectories(x, y)
    pred_whole_trajectories = recover_whole_trajectories(x, pred)
    return ground_whole_trajectories, pred_whole_trajectories


def generate_predictions_for_training(train_x, predictor):
    predictor.eval()
    with torch.no_grad():
        y_pred = predictor(train_x)
    prediction_trajectories = recover_whole_trajectories(train_x, y_pred)
    return prediction_trajectories


def recover_whole_trajectories(x_obs, x_un):
    x_obs = np.array(x_obs)
    x_un = np.array(x_un)
    whole_trajectories = np.array([np.append(x_obs[i], x_un[i]) for i in range(len(x_obs))])
    return whole_trajectories


def find_epsilon_from_nonconformity_1(d_0_samples, d_samples, predictor, observed_range, predicted_range, norm, kde_calculation_bin_num = 200000, kde_plot_bin_num = 1000):
    # Compute direct method nonconformities on d_0_samples.
    d_0_x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in d_0_samples]))
    d_0_y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in d_0_samples]))
    with torch.no_grad():
        d_0_pred = predictor(d_0_x)
    d_x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in d_samples]))
    d_y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in d_samples]))
    with torch.no_grad():
        d_pred = predictor(d_x)
    d_0_ground_whole = recover_whole_trajectories(d_0_x, d_0_y)
    d_0_pred_whole = recover_whole_trajectories(d_0_x, d_0_pred)
    d_ground_whole = recover_whole_trajectories(d_x, d_y)
    d_pred_whole = recover_whole_trajectories(d_x, d_pred)
    d_0_nonconformity_list = []
    d_nonconformity_list = []
    for i in range(len(d_0_ground_whole)):
        nonconformity = compute_robustness(d_0_pred_whole[i], norm) - compute_robustness(d_0_ground_whole[i], norm)
        d_0_nonconformity_list.append(nonconformity)
    for i in range(len(d_ground_whole)):
        nonconformity = compute_robustness(d_pred_whole[i], norm) - compute_robustness(d_ground_whole[i], norm)
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
    plt.hist(d_0_nonconformity_list, bins = np.arange(lower_bound, upper_bound + 0.0001, 0.0001))
    plt.title("Empirical Distribution of D_0 Nonconformity Scores")
    plt.show()

    plt.hist(d_nonconformity_list, bins = np.arange(lower_bound, upper_bound + 0.0001, 0.0001))
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
    new_score_list = np.arange(lower_bound, upper_bound, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def find_epsilon_from_nonconformity_2(alphas, d_0_samples, d_samples, predictor, observed_range, predicted_range, norm, kde_calculation_bin_num = 200000, kde_plot_bin_num = 1000, smoothing_term  = 0.00001):
    # Compute direct method nonconformities on d_0_samples.
    d_0_x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in d_0_samples]))
    d_0_y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in d_0_samples]))
    with torch.no_grad():
        d_0_pred = predictor(d_0_x)
    d_x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in d_samples]))
    d_y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in d_samples]))
    with torch.no_grad():
        d_pred = predictor(d_x)
    d_0_ground_whole = recover_whole_trajectories(d_0_x, d_0_y)
    d_0_pred_whole = recover_whole_trajectories(d_0_x, d_0_pred)
    d_ground_whole = recover_whole_trajectories(d_x, d_y)
    d_pred_whole = recover_whole_trajectories(d_x, d_pred)

    d_0_nonconformity_list = []
    for i in range(len(d_0_ground_whole)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            local_nonconformity = abs(d_0_ground_whole[i][tau] - d_0_pred_whole[i][tau]) / (alphas[tau] + smoothing_term)
            local_nonconformity_list.append(local_nonconformity)
        d_0_nonconformity_list.append(max(local_nonconformity_list))

    d_nonconformity_list = []
    for i in range(len(d_ground_whole)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            local_nonconformity = abs(d_ground_whole[i][tau] - d_pred_whole[i][tau]) / (
                        alphas[tau] + smoothing_term)
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
    new_score_list = np.arange(lower_bound, upper_bound, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def find_epsilon_from_nonconformity_3(hybrid_alphas, d_0_samples, d_samples, predictor, observed_range, predicted_range, norm,
                                      kde_calculation_bin_num=200000, kde_plot_bin_num=1000, smoothing_term=0.00001):
    # Compute direct method nonconformities on d_0_samples.
    d_0_x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in d_0_samples]))
    d_0_y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in d_0_samples]))
    with torch.no_grad():
        d_0_pred = predictor(d_0_x)
    d_x = torch.FloatTensor(np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in d_samples]))
    d_y = torch.FloatTensor(np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in d_samples]))
    with torch.no_grad():
        d_pred = predictor(d_x)
    d_0_ground_whole = recover_whole_trajectories(d_0_x, d_0_y)
    d_0_pred_whole = recover_whole_trajectories(d_0_x, d_0_pred)
    d_ground_whole = recover_whole_trajectories(d_x, d_y)
    d_pred_whole = recover_whole_trajectories(d_x, d_pred)

    d_0_nonconformity_list = []
    for i in range(len(d_0_ground_whole)):
        local_nonconformity = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            nonconformity = (h(d_0_pred_whole[i][tau], norm) - h(d_0_ground_whole[i][tau], norm)) / (hybrid_alphas[tau] + smoothing_term)
            local_nonconformity.append(nonconformity)
        d_0_nonconformity_list.append(max(local_nonconformity))

    d_nonconformity_list = []
    for i in range(len(d_ground_whole)):
        local_nonconformity = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            nonconformity = (h(d_pred_whole[i][tau], norm) - h(d_ground_whole[i][tau], norm)) / hybrid_alphas[tau]
            local_nonconformity.append(nonconformity)
        d_nonconformity_list.append(max(local_nonconformity))

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
    new_score_list = np.arange(lower_bound, upper_bound, step_size)
    d_0_pdf = kde_d_0.evaluate(new_score_list)
    d_pdf = kde_d.evaluate(new_score_list)
    divergence = 0
    for i in range(len(new_score_list) - 1):
        y_front = 0.5 * abs(d_0_pdf[i] - d_pdf[i])
        y_back = 0.5 * abs(d_0_pdf[i + 1] - d_pdf[i + 1])
        divergence += ((y_front + y_back) * step_size / 2)
    return divergence


def main():
    # Navigate to plots to generate and store the generated plots.
    os.chdir("plots")
    # Define hyperparameters:
    power = 9  # engine power level (0-10)
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)
    alt = 800  # altitude (ft)
    vt = 650  # initial velocity (ft/sec)
    phi = -math.pi / 8  # Roll angle from wings level (rad)
    theta = (-math.pi / 2) * 0.3  # Pitch angle from nose level (rad)
    psi = 0  # Yaw angle from North (rad)
    tmax = 3.51  # simulation time
    simulation_step = 1 / 30
    return_state = 'alt'
    delta = 0.2
    observed_range = [0, 100]
    predicted_range = [101, 105]
    training_size = 500 # For I1
    validation_size = 100
    calibration_size = 2000 # For I2
    testing_size = 100
    n_epochs = 300
    coverage_sample_size = 50
    epsilon_estimation_size = 1000

    # Let's try to plot the original and disturbed trajectory.
    print("Start: Plotting original vs. disturbed trajectories")
    plane = Plane(power, alpha, beta, alt, vt, phi, theta, psi, tmax, simulation_step, return_state)
    plane.plot_nominal_and_simulated_trajectories()
    print("End: Plotting original vs. disturbed trajectories")
    print()

    # Now, let's train a predictor using the original trajectories without distribution shift.
    # Generate training data.
    print("Start: Generating predictor training and validation data.")
    training_simulations = []
    for i in range(training_size):
        simulated = plane.simulate_nominal()
        training_simulations.append(simulated)

    validation_simulations = []
    for i in range(validation_size):
        simulated = plane.simulate_nominal()
        validation_simulations.append(simulated)

    print("Generated training data of shape:", np.shape(np.array(training_simulations)))
    print("Generated validation data of shape:", np.shape(np.array(validation_simulations)))

    # Normalize the data.
    norm = np.max(np.concatenate((training_simulations, validation_simulations)))
    training_simulations /= norm
    validation_simulations /= norm

    # Slice for training and validation data.
    train_x = torch.FloatTensor(
        np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in training_simulations]))
    train_y = torch.FloatTensor(
        np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in training_simulations]))
    validation_x = torch.FloatTensor(
        np.array([sample[observed_range[0]: observed_range[1] + 1] for sample in validation_simulations]))
    validation_y = torch.FloatTensor(
        np.array([sample[predicted_range[0]: predicted_range[1] + 1] for sample in validation_simulations]))

    print("Generated train_x of shape:", np.shape(np.array(train_x)))
    print("Generated train_y of shape:", np.shape(np.array(train_y)))
    print("Generated validation_x of shape:", np.shape(np.array(validation_x)))
    print("Generated validation_y of shape:", np.shape(np.array(validation_y)))
    print("End: Generating predictor training and validation data.")
    print()

    # Train a quantile regressor.
    print("Start: fitting a predictor.")
    predictor = Predictor(len(train_x[0]), len(train_y[0]))
    optimizer = optim.Adam(predictor.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(train_x, train_y), shuffle=True, batch_size=8)
    for epoch in range(n_epochs):
        predictor.train()
        for X_batch, y_batch in loader:
            y_pred = predictor(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        predictor.eval()
        with torch.no_grad():
            y_pred = predictor(train_x)
            train_rmse = np.sqrt(loss_fn(y_pred, train_y))
            y_pred = predictor(validation_x)
            validation_rmse = np.sqrt(loss_fn(y_pred, validation_y))
        print("Epoch %d: train RMSE %.4f, validation RMSE %.4f" % (epoch, train_rmse, validation_rmse))

    # Plot some trajectories.
    predictor.eval()
    with torch.no_grad():
        y_pred = predictor(train_x)

    print("End: fitting a predictor.")
    print()

    print("Start: Computing alphas for indirect methods.")
    train_pred_whole = generate_predictions_for_training(train_x, predictor)
    train_ground_whole = recover_whole_trajectories(train_x, train_y)
    alphas = dict()
    for tau in range(predicted_range[0], predicted_range[1] + 1):
        local_alpha = []
        for i in range(len(train_pred_whole)):
            local_alpha.append(abs(train_pred_whole[i][tau] - train_ground_whole[i][tau]))
        alphas[tau] = np.max(local_alpha)
    print("alphas", alphas)
    print("End: Computing alphas for indirect methods.")

    print("Start: Computing alphas for hybrid methods.")
    hybrid_alphas = dict()
    for tau in range(predicted_range[0], predicted_range[1] + 1):
        local_alpha = []
        for i in range(len(train_pred_whole)):
            local_alpha.append(abs(h(train_pred_whole[i][tau], norm) - h(train_ground_whole[i][tau], norm)))
        hybrid_alphas[tau] = np.max(local_alpha)
    print("End: Computing alphas for hybrid methods.")
    print()

    print("Start: Estimating Epslion.")
    d_0_samples = []
    d_samples = []
    for i in range(epsilon_estimation_size):
        d_0_samples.append(plane.simulate_nominal())
        d_samples.append(plane.simulate_disturbed())
    d_0_samples /= norm
    d_samples /= norm
    shift_1 = find_epsilon_from_nonconformity_1(d_0_samples, d_samples, predictor, observed_range, predicted_range,
                                                norm)
    print("Epsilon from direct method:", shift_1)
    shift_2 = find_epsilon_from_nonconformity_2(alphas, d_0_samples, d_samples, predictor, observed_range,
                                                predicted_range, norm)
    print("Epsilon from indirect method (Variant I):", shift_2)
    shift_3 = find_epsilon_from_nonconformity_3(hybrid_alphas, d_0_samples, d_samples, predictor, observed_range,
                                                predicted_range,
                                                norm)
    print("Epsilon from indirect method (Variant II):", shift_3)
    epsilon = max(shift_1, shift_2, shift_3)
    with open("epsilon.json", "w") as outfile:
        outfile.write(str(epsilon))
    print("Used epsilon:", epsilon)
    print("End: Estimating Epsilon.")
    print()

    print(f"Start: Perform verifications with delta = {delta}")
    print(f"Generating calibration data.")
    direct_vanilla = []
    direct_robust = []
    indirect_vanilla = []
    indirect_robust = []
    hybrid_vanilla = []
    hybrid_robust = []
    for i in range(coverage_sample_size):
        print(f"Generating testing and calibration data for sample {i + 1}")
        print()
        calib_ground_whole_trajectories, calib_pred_whole_trajectories = generate_data_with_predictions(norm, predictor, plane, calibration_size, observed_range, predicted_range,disturbed=False)
        test_ground_whole_trajectories, test_pred_whole_trajectories = generate_data_with_predictions(norm, predictor, plane, testing_size, observed_range, predicted_range, disturbed = True)

        print(f"Performing direct vanilla runtime verification on sample {i + 1}:")
        vanilla_coverage_direct, vanilla_direct_nonconformity_list, vanilla_direct_c = run_direct_vanilla_runtime_verification(
            norm, delta, calib_ground_whole_trajectories, calib_pred_whole_trajectories,
            test_ground_whole_trajectories, test_pred_whole_trajectories)
        direct_vanilla.append(vanilla_coverage_direct)
        print(f"Coverage for direct vanilla runtime verification on sample {i + 1} is", vanilla_coverage_direct)
        print()

        print(f"Performing direct robust runtime verification on sample {i + 1}:")
        robust_coverage_direct, vanilla_robust_nonconformity_list, vanilla_robust_c = run_direct_robust_runtime_verification(
            norm, delta, epsilon, calib_ground_whole_trajectories,
            calib_pred_whole_trajectories, test_ground_whole_trajectories,
            test_pred_whole_trajectories)
        direct_robust.append(robust_coverage_direct)
        print(f"Coverage for direct robust runtime verification on sample {i + 1} is", robust_coverage_direct)
        print()

        print(f"Performing indirect vanilla runtime verification on sample {i + 1}")
        vanilla_coverage, vanilla_indirect_nonconformity_list, vanilla_indirect_c = run_vanilla_indirect_runtime_verification(norm, alphas, delta, predicted_range, calib_ground_whole_trajectories,
                                                  calib_pred_whole_trajectories, test_ground_whole_trajectories,
                                                  test_pred_whole_trajectories, plot_normalize = (i == 0))
        indirect_vanilla.append(vanilla_coverage)
        print(f"Coverage for indirect vanilla runtime verification on sample {i + 1} is", vanilla_coverage)
        print()

        print(f"Performing indirect robust runtime verification on sample {i + 1}")
        robust_coverage,  robust_indirect_nonconformity_list, robust_indirect_c = run_robust_indirect_runtime_verification(norm, alphas, delta, epsilon, predicted_range, calib_ground_whole_trajectories,
                                                 calib_pred_whole_trajectories, test_ground_whole_trajectories,
                                                 test_pred_whole_trajectories)
        indirect_robust.append(robust_coverage)
        print(f"Coverage for indirect robust runtime verification on sample {i + 1} is", robust_coverage)
        print()

        print(f"Performing hybrid vanilla runtime verification on sample {i + 1}")
        vanilla_coverage_hybrid, vanilla_hybrid_nonconformity_list,  vanilla_hybrid_c = run_hybrid_vanilla_runtime_verification(hybrid_alphas, norm, predicted_range, observed_range, delta,
                                                calib_ground_whole_trajectories, calib_pred_whole_trajectories,
                                                test_ground_whole_trajectories, test_pred_whole_trajectories)
        hybrid_vanilla.append(vanilla_coverage_hybrid)
        print(f"Coverage for hybrid vanilla runtime verification on sample {i + 1} is", vanilla_coverage_hybrid)
        print()

        print(f"Performing hybrid robust runtime verification on sample {i + 1}")
        robust_coverage_hybrid, robust_hybrid_nonconformity_list, robust_hybrid_c = run_hybrid_robust_runtime_verification(hybrid_alphas, epsilon, norm, predicted_range, observed_range, delta,
                                               calib_ground_whole_trajectories, calib_pred_whole_trajectories,
                                               test_ground_whole_trajectories, test_pred_whole_trajectories)
        hybrid_robust.append(robust_coverage_hybrid)
        print(f"Coverage for hybrid robust runtime verification on sample {i + 1} is", robust_coverage_hybrid)
    print("End: Performing verifications")
    print()

    print("Plotting coverages:")
    # Draw the coverage plot.
    used_bins = []
    percentile_value = np.min([direct_vanilla, direct_robust, indirect_vanilla, indirect_robust, hybrid_vanilla, hybrid_robust]) - 0.1
    while percentile_value <= 1.02:
        used_bins.append(percentile_value)
        percentile_value += 0.005

    # Plot for runtime verification (direct).
    plt.hist(direct_vanilla, bins=used_bins, label = "Direct Method")
    plt.hist(direct_robust, bins=used_bins, label = "Robust Direct Method")
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_plot_direct.pdf")
    plt.show()

    plt.hist(indirect_vanilla, bins=used_bins, label = "Indirect Method: Variant I")
    plt.hist(indirect_robust, bins=used_bins, label="Robust Indirect Method: Variant I")
    plt.hist(hybrid_vanilla, bins=used_bins, label="Indirect Method: Variant II")
    plt.hist(hybrid_robust, bins=used_bins, label = "Robust Indirect Method: Variant II")
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_plot_indirect_paper.pdf")
    plt.show()

    # Plot for nonconformity on the last sample:
    plt.hist(vanilla_direct_nonconformity_list[:-1], bins = np.arange(min(vanilla_direct_nonconformity_list[:-1]), max(vanilla_direct_nonconformity_list[:-1]) + 0.0001, 0.0001))
    plt.axvline(x=vanilla_direct_c, color='b', label='$C$')
    plt.axvline(x=vanilla_robust_c, color='g', label = '$\\tilde{C}$')
    plt.xlabel("Nonconformity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("direct_nonconformities.pdf")
    plt.show()

    plt.hist(vanilla_hybrid_nonconformity_list[:-1], bins = np.arange(min(vanilla_hybrid_nonconformity_list[:-1]), max(vanilla_hybrid_nonconformity_list[:-1]) + 0.01, 0.01))
    plt.axvline(x = vanilla_hybrid_c, color = 'b', label = '$C$')
    plt.axvline(x = robust_hybrid_c, color = "g", label = '$\\tilde{C}$')
    plt.xlabel("Nonconformity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("indirect_2_nonconformities.pdf")
    plt.show()

    plt.hist(vanilla_indirect_nonconformity_list[:-1], bins = np.arange(min(vanilla_indirect_nonconformity_list[:-1]), max(vanilla_indirect_nonconformity_list[:-1]) + 0.01, 0.01))
    plt.axvline(x = vanilla_indirect_c, color = 'b', label = "$C$")
    if robust_indirect_c == float("inf"):
        plt.axvline(x = robust_indirect_c, color = 'g', label = "$\\tilde{C} = \infty$")
    else:
        plt.axvline(x=robust_indirect_c, color='g', label="$\\tilde{C}$")
    plt.xlabel("Nonconformity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("indirect_1_nonconformities.pdf")
    plt.show()

    print("End: Simulating for the STL Runtime Verification.")
    print()


if __name__ == "__main__":
    main()