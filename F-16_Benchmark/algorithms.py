"""
In this file, we provide the algoritms associated with robust runtime verification and vanilla runtime
verification (from ICCPS 2023).
"""
import numpy as np

"""
Set up helper functions for robust cp:
"""

def f(t):
    # We assume to use the total variation distance.
    return 0.5 * abs(t - 1)


def g(f, epsilon, beta, search_step = 0.0007):
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


def g_inverse(f, epsilon, tau, search_step = 0.0007):
    # Check input.
    if tau < 0 or tau > 1:
        raise Exception("Input to the function g_inverse is out of range.")

    beta = 1
    while beta >= 0:
        if beta != 1 and g(f, epsilon, beta) <= tau:
            return beta
        beta -= search_step

    raise Exception("No return from function g_inverse.")


def calculate_delta_n(delta, n, f, epsilon):
    inner = (1 + 1 / n) * g_inverse(f, epsilon, 1 - delta)
    return (1 - g(f, epsilon, inner))


def calculate_delta_tilde(delta_n, f, epsilon):
    answer = 1 - g_inverse(f, epsilon, 1 - delta_n)
    return answer


"""
Set up functions for computing STL robust semantics.
"""

def h(value, norm, collision_threshold = 60):
    return (value - collision_threshold / norm)


# Write a function to compute the worst-case robustness.
def compute_worst_case_robustness(Q_dict, trajectory, norm):
    h_values = []
    for tau in range(len(trajectory)):
        if tau not in Q_dict:
            h_values.append(h(trajectory[tau], norm))
        else:
            h_values.append(h(trajectory[tau] - Q_dict[tau], norm))
    return min(h_values)


def compute_robustness(trajectory, norm):
    h_values = []
    for value in trajectory:
        h_values.append(h(value, norm))
    return min(h_values)


"""
Set up functions for the runtime verification algorithms.
"""

def run_direct_vanilla_runtime_verification(norm, delta, calib_ground_whole_trajectories, calib_pred_whole_trajectories, test_ground_whole_trajectories, test_pred_whole_trajectories):
    # Compute nonconformities.
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        nonconformity = compute_robustness(calib_pred_whole_trajectories[i], norm) - compute_robustness(calib_ground_whole_trajectories[i], norm)
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
        robustness_ground = compute_robustness(test_ground_whole_trajectories[i], norm)
        robustness_predicted = compute_robustness(test_pred_whole_trajectories[i], norm)
        ground_robustnesses.append(robustness_ground)
        predicted_robustnesses.append(robustness_predicted)
        lower_bounds.append(robustness_predicted - c)
        if robustness_ground >= (robustness_predicted - c):
            coverage_count += 1

    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c


def run_direct_robust_runtime_verification(norm, delta, epsilon, calib_ground_whole_trajectories, calib_pred_whole_trajectories, test_ground_whole_trajectories, test_pred_whole_trajectories):
    # Compute nonconformities.
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        nonconformity = compute_robustness(calib_pred_whole_trajectories[i], norm) - compute_robustness(
            calib_ground_whole_trajectories[i], norm)
        nonconformity_list.append(nonconformity)
    nonconformity_list.sort()

    # Calculate delta_tilde.
    delta_n = calculate_delta_n(delta, len(calib_ground_whole_trajectories), f, epsilon)
    delta_tilde = calculate_delta_tilde(delta_n, f, epsilon)
    # Compute c_tilde.
    p = int(np.ceil((len(calib_ground_whole_trajectories)) * (1 - delta_tilde)))
    c_tilde = nonconformity_list[p - 1]

    # Compute the coverage.
    coverage_count = 0
    ground_robustnesses = []
    predicted_robustnesses = []
    lower_bounds = []
    for i in range(len(test_ground_whole_trajectories)):
        robustness_ground = compute_robustness(test_ground_whole_trajectories[i], norm)
        robustness_predicted = compute_robustness(test_pred_whole_trajectories[i], norm)
        ground_robustnesses.append(robustness_ground)
        predicted_robustnesses.append(robustness_predicted)
        lower_bounds.append(robustness_predicted - c_tilde)
        if robustness_ground >= (robustness_predicted - c_tilde):
            coverage_count += 1

    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c_tilde


def run_vanilla_indirect_runtime_verification(norm, alphas, delta, predicted_range, calib_ground_whole_trajectories, calib_pred_whole_trajectories, test_ground_whole_trajectories, test_pred_whole_trajectories, smoothing_term = 0.00001, plot_normalize = True):
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            local_nonconformity = abs(calib_ground_whole_trajectories[i][tau] - calib_pred_whole_trajectories[i][tau]) / (alphas[tau] + smoothing_term)
            local_nonconformity_list.append(local_nonconformity)
        nonconformity_list.append(max(local_nonconformity_list))
    nonconformity_list.append(float("inf"))
    nonconformity_list.sort()
    p = int(np.ceil((len(calib_ground_whole_trajectories) + 1) * (1 - delta)))
    c = nonconformity_list[p - 1]
    # Generate prediction region.
    prediction_region_dict = dict()
    for tau in range(predicted_range[0], predicted_range[1] + 1):
        prediction_region_dict[tau] = c * alphas[tau]

    # Compute Coverage.
    coverage_count = 0
    for i in range(len(test_ground_whole_trajectories)):
        actual_robustness = compute_robustness(test_ground_whole_trajectories[i], norm)
        predicted_worst_case_robustness = compute_worst_case_robustness(prediction_region_dict, test_pred_whole_trajectories[i], norm)
        if actual_robustness >= predicted_worst_case_robustness:
            coverage_count += 1
    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c


def run_robust_indirect_runtime_verification(norm, alphas, delta, epsilon, predicted_range, calib_ground_whole_trajectories,
                                                 calib_pred_whole_trajectories, test_ground_whole_trajectories,
                                                 test_pred_whole_trajectories, smoothing_term = 0.00001):
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        local_nonconformity_list = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            local_nonconformity = abs(
                calib_ground_whole_trajectories[i][tau] - calib_pred_whole_trajectories[i][tau]) / (
                                              alphas[tau] + smoothing_term)
            local_nonconformity_list.append(local_nonconformity)
        nonconformity_list.append(max(local_nonconformity_list))
    nonconformity_list.sort()
    # Calculate delta_tilde.
    delta_n = calculate_delta_n(delta, len(calib_ground_whole_trajectories), f, epsilon)
    delta_tilde = calculate_delta_tilde(delta_n, f, epsilon)
    # Compute c_tilde.
    p = int(np.ceil((len(calib_ground_whole_trajectories)) * (1 - delta_tilde)))
    c_tilde = nonconformity_list[p - 1]
    # Generate prediction region.
    prediction_region_dict = dict()
    for tau in range(predicted_range[0], predicted_range[1] + 1):
        prediction_region_dict[tau] = c_tilde * alphas[tau]

    # Compute Coverage.
    coverage_count = 0
    for i in range(len(test_ground_whole_trajectories)):
        actual_robustness = compute_robustness(test_ground_whole_trajectories[i], norm)
        predicted_worst_case_robustness = compute_worst_case_robustness(prediction_region_dict,
                                                                        test_pred_whole_trajectories[i], norm)
        if actual_robustness >= predicted_worst_case_robustness:
            coverage_count += 1
    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c_tilde


def run_hybrid_vanilla_runtime_verification(hybrid_alphas, norm, predicted_range, observed_range, delta,
                                                calib_ground_whole_trajectories, calib_pred_whole_trajectories,
                                                test_ground_whole_trajectories, test_pred_whole_trajectories, smoothing_term = 0.00001):
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        local_nonconformity = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            nonconformity = (h(calib_pred_whole_trajectories[i][tau], norm) - h(calib_ground_whole_trajectories[i][tau],
                                                                               norm)) / (hybrid_alphas[tau] + smoothing_term)
            local_nonconformity.append(nonconformity)
        nonconformity_list.append(max(local_nonconformity))
    nonconformity_list.append(float("inf"))
    nonconformity_list.sort()
    p = int(np.ceil((len(calib_ground_whole_trajectories) + 1) * (1 - delta)))
    c = nonconformity_list[p - 1]

    # Compute Coverage.
    coverage_count = 0
    for i in range(len(test_ground_whole_trajectories)):
        actual_robustness = compute_robustness(test_ground_whole_trajectories[i], norm)
        h_values = []
        for tau in range(len(test_pred_whole_trajectories[i])):
            if tau <= predicted_range[1] and tau >= predicted_range[0]:
                h_values.append(h(test_pred_whole_trajectories[i][tau], norm) - c * hybrid_alphas[tau])
            else:
                h_values.append(h(test_pred_whole_trajectories[i][tau], norm))
        worst_case_robustness = min(h_values)
        if actual_robustness >= worst_case_robustness:
            coverage_count += 1
    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c


def run_hybrid_robust_runtime_verification(hybrid_alphas, epsilon, norm, predicted_range, observed_range, delta,
                                               calib_ground_whole_trajectories, calib_pred_whole_trajectories,
                                               test_ground_whole_trajectories, test_pred_whole_trajectories, smoothing_term = 0.00001):
    nonconformity_list = []
    for i in range(len(calib_ground_whole_trajectories)):
        local_nonconformity = []
        for tau in range(predicted_range[0], predicted_range[1] + 1):
            nonconformity = (h(calib_pred_whole_trajectories[i][tau], norm) - h(calib_ground_whole_trajectories[i][tau],
                                                                                norm)) / (hybrid_alphas[tau] + smoothing_term)
            local_nonconformity.append(nonconformity)
        nonconformity_list.append(max(local_nonconformity))
    nonconformity_list.sort()
    delta_n = calculate_delta_n(delta, len(calib_ground_whole_trajectories), f, epsilon)
    delta_tilde = calculate_delta_tilde(delta_n, f, epsilon)
    p = int(np.ceil((len(calib_ground_whole_trajectories)) * (1 - delta_tilde)))
    c_tilde = nonconformity_list[p - 1]

    # Compute Coverage.
    coverage_count = 0
    for i in range(len(test_ground_whole_trajectories)):
        actual_robustness = compute_robustness(test_ground_whole_trajectories[i], norm)
        h_values = []
        for tau in range(len(test_pred_whole_trajectories[i])):
            if tau <= predicted_range[1] and tau >= predicted_range[0]:
                h_values.append(h(test_pred_whole_trajectories[i][tau], norm) - c_tilde * hybrid_alphas[tau])
            else:
                h_values.append(h(test_pred_whole_trajectories[i][tau], norm))
        worst_case_robustness = min(h_values)
        if actual_robustness >= worst_case_robustness:
            coverage_count += 1
    return coverage_count / len(test_ground_whole_trajectories), nonconformity_list, c_tilde