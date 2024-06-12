import math

import numpy as np
from optimizationFunctions import fixed_target_coverage, moving_target_coverage, H_plots, fixed_target_coverage_gradient, moving_target_coverage_gradient
#todo riguardare caso più droni -> problema risolto passando da alpha = 0.0001 a alpha = 0.01 che sembra ottimale
initial_condition_drones = np.array([[20, 20, 10], [-12, -16, 10], [5, 10, 20]])

target_positions = np.array([[1, 2], [15, 7], [20, 10], [-12, 15]])

#covariance = fixed_target_coverage_gradient(initial_condition_drones, target_positions, alpha_parameter=0.01,
#                                            graphics="ON", gradient_descent_method="adam", total_time_step=300)



#covariance = fixed_target_coverage(initial_condition_drones, target_positions, bounds_z=(5, 50), graphics="OFF", alpha_parameter=0.01)
print("fine ottimizzazione")

#H_plots(covariance[0], covariance[1])
#H_plots(covariance[2], covariance[3])


target_trajectories = [2, 5, 7, 3, 9, 17, 27, 45, 36, 21]
moving_object_covariance = moving_target_coverage_gradient(initial_condition_drones, target_trajectories,
                                                           bounds_z=(5, 50), graphics="ON", G_function_parameter=5,
                                                           alpha_parameter=0.01, gradient_descent_method="rmsprop")

H_plots(moving_object_covariance[0], moving_object_covariance[1])
H_plots(moving_object_covariance[2], moving_object_covariance[3])



