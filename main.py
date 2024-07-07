import math
import sys
import numpy as np
from optimizationFunctions import fixed_target_coverage, moving_target_coverage, H_plots,\
    fixed_target_coverage_gradient, moving_target_coverage_gradient, moving_target_coverage_gradient_KF,\
    moving_target_coverage_gradient_KF_multipleUpdate, moving_target_coverage_gradient_KF_multipleUpdate_prediction
#todo riguardare caso più droni -> problema risolto passando da alpha = 0.0001 a alpha = 0.01 che sembra ottimale

'''                     esempi e combinazioni                   '''
#esempio 1, target fissi
# in questo esempio l'ottimizzazione con minimize funziona meglio della discesa del gradiente
'''
initial_condition_drones = np.array([[20, 20, 10], [-12, -16, 10]])
target_positions = np.array([[30, 30], [30, 30]])
covariance = fixed_target_coverage(initial_condition_drones, target_positions, bounds_z=(5, 50), alpha_parameter=0.01, graphics="ON", G_function_parameter=6)
covariance = fixed_target_coverage_gradient(initial_condition_drones, target_positions, bounds_z=(5, 50), alpha_parameter=0.01, graphics="ON", G_function_parameter=6)
'''

# esempio 2, target fissi
# aumento progressivo dei target: utilizzando gradient_descent non si riesce a ottimizzare, ma utilizzando metodi un po' più complessi tipo adam, o ancora meglio rmsprop,
# riusciamo a trovare la soluzione che minimizza la covarianza. l'applicazione di optimize.minimize invece non riesce a trovare una soluzione,
# mentre applicando il gradiente di discesa (con rmsprop) si riesce a minimizzare complessivamente
'''
initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25]])
target_positions = np.array([[30, 30]])
target_positions = np.array([[30, 30], [-20, -20]])
target_positions = np.array([[30, 30], [-20, -20], [10, -10]])
covariance = fixed_target_coverage(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="ON", G_function_parameter=3)
covariance = fixed_target_coverage_gradient(initial_condition_drones, target_positions, bounds_z=(5, 100), 
                                            alpha_parameter=0.01, graphics="ON", G_function_parameter=3, 
                                            total_time_step=900, gradient_descent_method="rmsprop")
'''

# esempio 3, target fissi
# caso estremo in cui andiamo a confrontare il comportamento di 6 droni che devono posizionarsi su un solo target
'''
initial_condition_drones = np.array([[20, 20, 10], [-12, -16, 10], [-15, 27, 30], [0, 0, 40], [14, 45, 37], [-40, -40, 40]])
target_positions = np.array([[30, 30]])
covariance = fixed_target_coverage(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="ON", G_function_parameter=3)
covariance = fixed_target_coverage_gradient(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="ON", G_function_parameter=3, total_time_step=900, gradient_descent_method="rmsprop")

'''

#esempio 4, target fissi
# ho 6 droni che partono tutti dalla stessa posizione al centro della mappa e devono posizionarsi su 6 target sparsi per la mappa
'''
# posizione iniziale identica per tutti i droni
initial_condition_drones = np.array([[0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10], [0, 0, 10]])
# posizione iniziale leggermente differente per ogni drone
initial_condition_drones = np.array([[1, 0, 10], [0, 2, 10], [0, 0, 10], [-3, 0, 10], [0, -7, 10], [0, 0, 10]])
# si nota che il comportamento cambia drasticamente se le posizioni iniziali dei droni sono leggermente differenti

target_positions = np.array([[30, 30], [20, 40], [-18, 20], [+35, -26], [-25, -25], [-32, -40]])
covariance = fixed_target_coverage(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="ON", G_function_parameter=3)
covariance = fixed_target_coverage_gradient(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="ON", G_function_parameter=3, total_time_step=900, gradient_descent_method="rmsprop")
'''

#esempio 5, target mobili
# 1 drone e 4 target mobili, confronto tra i 2 metodi di ottimizzazione


'''                 Posizione iniziale dei droni                '''
#initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25]])
#initial_condition_drones = np.array([[20, 20, 20]])
initial_condition_drones = np.array([[-30, 0, 10], [20, 20, 20]])
#initial_condition_drones = np.array([[-10, 0, 50], [-30, -15, 50], [34, -28, 30]])


'''                 Posizioni dei target                        '''
#target_positions = np.array([[30, 30], [20, 40], [-18, 20], [+35, -26], [-25, -25], [-32, -40]])

'''                 Traiettorie dei target                      '''
#target_trajectories = [3, 6, 8, 9, 24, 26, 27, 7]
#target_trajectories = [2, 5, 7, 10, 13]
target_trajectories = [1, 2, 9]


'''                 Coverage con ottimizzazione                 '''
#covariance = fixed_target_coverage(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="ON", G_function_parameter=3)
#moving_object_covariance = moving_target_coverage(initial_condition_drones, target_trajectories, bounds_z=(5, 100), graphics="ON", G_function_parameter=3, alpha_parameter=0.01)
'''                 Coverage con discesa del gradiente          '''
#covariance = fixed_target_coverage_gradient(initial_condition_drones, target_positions, bounds_z=(5, 100), alpha_parameter=0.01, graphics="OFF", G_function_parameter=3, total_time_step=300, gradient_descent_method="rmsprop")

'''
moving_object_covariance = moving_target_coverage_gradient(initial_condition_drones, target_trajectories,
                                                              bounds_z=(5, 100), graphics="ON", G_function_parameter=5,
                                                              alpha_parameter=0.000001, gradient_descent_method="rmsprop")
'''




moving_object_covariance = moving_target_coverage_gradient_KF_multipleUpdate_prediction(initial_condition_drones, target_trajectories,
                                                              bounds_z=(5, 100), graphics="ON", G_function_parameter=5,
                                                              alpha_parameter=0.1, gradient_descent_method="rmsprop",
                                                              KF="UPDATE", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=3)



'''
moving_object_covariance = moving_target_coverage_gradient_KF(initial_condition_drones, target_trajectories,
                                                              bounds_z=(5, 100), graphics="ON", G_function_parameter=5,
                                                              alpha_parameter=0.1, gradient_descent_method="rmsprop",
                                                              KF="UPDATE", sigma_noise= 2)

'''
'''             Grafici per target fissi                        '''
#H_plots(covariance[0], covariance[1])
#H_plots(covariance[2], covariance[3])



'''             Grafici per target in movimento                 '''
H_plots(moving_object_covariance[0], moving_object_covariance[1])
H_plots(moving_object_covariance[2], moving_object_covariance[3])
