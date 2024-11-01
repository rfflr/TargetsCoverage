import math
import sys
import pandas as pd
import numpy as np
from optimizationFunctions import fixed_target_coverage, moving_target_coverage, H_plots, \
    fixed_target_coverage_gradient, moving_target_coverage_gradient, moving_target_coverage_gradient_KF, \
    moving_target_coverage_gradient_KF_multipleUpdate, moving_target_coverage_gradient_KF_multipleUpdate_prediction, \
    H_plots_kalman, RMSE_plot, h_trend_plots, h_targets_trend_plots, \
    moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv, calcola_media_rmse_per_time_step, \
    calcola_media_h_target_per_time_step, RMSE_multiple_plots, \
    fixed_target_coverage_gradient_KF_multipleUpdate_prediction, \
    fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv, \
    MONTECARLO_moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv, \
    DRONE_MONTECARLO_moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv, RMSE_multiple_plots2,\
    fromStringToList, Height_fromCsvToList, trasponi_matrice, HeightPlots, heightMean, HeightPlots2, RMSE_multiple_plots3, HeightPlots3

###configurazione A
#initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20]])
#target_trajectories = [2, 5, 7, 10, 13, 17, 18]

###configurazione B
#initial_condition_drones = np.array([[-10, 0, 50], [-30, -15, 50], [34, -28, 30]])
#target_trajectories = [2, 5, 7, 10, 13, 17, 18]

# configurazione C
initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20], [30, 20, 40], [35, 35, 5], [0, 35, 5]])
target_trajectories = [2, 5, 7, 10, 13, 17, 18]

#                 Posizione iniziale dei droni
# initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20]])
# initial_condition_drones = np.array([[20, 20, 20]])
# initial_condition_drones = np.array([[-30, 0, 10], [20, 20, 20]])
# initial_condition_drones = np.array([[-10, 0, 50], [-30, -15, 50], [34, -28, 30]])


#                 Posizioni dei target
#targets_positions = np.array([[30, 30], [20, 40], [-18, 20], [+35, +26], [+25, -25], [-32, -40]])

#                 Traiettorie dei target
# target_trajectories = [3, 6, 8, 9, 24, 26, 27, 7]
# target_trajectories = [2, 5, 7, 10, 13, 17, 18]
# target_trajectories = [1, 2, 9]

#                CONFIGURAZIONE PROBLEMA CON TARGET FISSI
#initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20]])
#targets_positions = np.array([[+20, +20], [8, 7], [30, 0], [14, 38], [-5, -10]])


'''
#METODO IN CUI VIENE MOSTRATO UN SINGOLO ESPERIMENTO (target FISSI)
fixed_object_covariance, RMSE, h_trend, h_targets_trend = fixed_target_coverage_gradient_KF_multipleUpdate_prediction(initial_condition_drones, targets_positions,
    bounds_z=(5, 100), graphics="ON", G_function_parameter=2,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="OFF", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=0, correction_in_viewing_range="ON")


#             Grafici per target in movimento
#H_plots_kalman(moving_object_covariance[0], moving_object_covariance[1], moving_object_covariance[2])
RMSE_plot(RMSE)
#h_trend_plots(h_trend)
h_targets_trend_plots(h_targets_trend)

'''


'''
# METODO PER IL SALVATAGGIO DEI CSV (target FISSI)
path = "C:/CSV_tesi/dataset_fixed_PREDICTION10.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=200, bounds_z=(5, 100), graphics="OFF",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="PREDICTION",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)


df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''


#METODO IN CUI VIENE MOSTRATO UN SINGOLO ESPERIMENTO (target MOBILI)
moving_object_covariance, RMSE, h_trend, h_targets_trend = moving_target_coverage_gradient_KF_multipleUpdate_prediction(
    initial_condition_drones, target_trajectories,
    bounds_z=(5, 100), graphics="ON", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="OFF", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=5, correction_in_viewing_range="ON")


#             Grafici per target in movimento
#H_plots_kalman(moving_object_covariance[0], moving_object_covariance[1], moving_object_covariance[2])
RMSE_plot(RMSE)
#h_trend_plots(h_trend)
h_targets_trend_plots(h_targets_trend)



'''
#METODO PER IL SALVATAGGIO DEI CSV
path = "C:/CSV_tesi/dataset_configC_PREDICTION10.csv"


moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="PREDICTION", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=10, correction_in_viewing_range="ON", file_path=path)


df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''

'''
#METODO PER PRENDERE I CSV DEGLI RMSE CALCOLATI NELLE VARIE STRATEGIE E SOVRAPPORLI IN UN UNICO GRAFICO
path_OFF = "C:/CSV_tesi/dataset_configA_OFF.csv"
path_UPDATE = "C:/CSV_tesi/dataset_configA_UPDATE.csv"
path_PREDICTION5 = "C:/CSV_tesi/dataset_configA_PREDICTION5.csv"
path_PREDICTION10 = "C:/CSV_tesi/dataset_configA_PREDICTION10.csv"
df_OFF = pd.read_csv(path_OFF)
df_UPDATE = pd.read_csv(path_UPDATE)
df_PREDICTION5 = pd.read_csv(path_PREDICTION5)
df_PREDICTION10 = pd.read_csv(path_PREDICTION10)

mean_RMSE_OFF = calcola_media_rmse_per_time_step(df_OFF)
mean_RMSE_UPDATE = calcola_media_rmse_per_time_step(df_UPDATE)
mean_RMSE_PREDICTION5 = calcola_media_rmse_per_time_step(df_PREDICTION5)
mean_RMSE_PREDICTION10 = calcola_media_rmse_per_time_step(df_PREDICTION10)

RMSE_multiple_plots(mean_RMSE_OFF, mean_RMSE_UPDATE, mean_RMSE_PREDICTION5, mean_RMSE_PREDICTION10)
'''

'''
#METODO PER IL SALVATAGGIO DEI .CSV DI OFF, UPDATE, PREDICTION5 E PREDICTION10
path = "C:/CSV_tesi/dataset_configA_OFF.csv"


moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="OFF", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=10, correction_in_viewing_range="ON", file_path=path)

print("OFF completato")

path = "C:/CSV_tesi/dataset_configA_UPDATE.csv"


moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="UPDATE", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=10, correction_in_viewing_range="ON", file_path=path)

print("UPDATE completato")

path = "C:/CSV_tesi/dataset_configA_PREDICTION5.csv"


moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="PREDICTION", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=5, correction_in_viewing_range="ON", file_path=path)


print("PREDICTION5 completato")

path = "C:/CSV_tesi/dataset_configA_PREDICTION10.csv"


moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="PREDICTION", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=10, correction_in_viewing_range="ON", file_path=path)

print("PREDICTION10 completato")
'''


'''
#COSTRUZIONE DEI GRAFICI A PARTIRE DAI .CSV DI OFF, UPDATE, PREDICTION5 E PREDICTION10

# grafico OFF
path = "C:/CSV_tesi/dataset_configA_OFF.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)



#grafici UPDATE
path = "C:/CSV_tesi/dataset_configA_UPDATE.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)


#grafici PREDICTION5
path = "C:/CSV_tesi/dataset_configA_PREDICTION5.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)



# grafici PREDICTION10
path = "C:/CSV_tesi/dataset_configA_PREDICTION10.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''


'''
########    COSTRUZIONE MULTIPLA DEI CSV (TARGET FISSI)

path = "C:/CSV_tesi/dataset_fixed_OFF.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=10, bounds_z=(5, 100), graphics="OFF",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="OFF",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("OFF completato")

path = "C:/CSV_tesi/dataset_fixed_UPDATE.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=10, bounds_z=(5, 100), graphics="OFF",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="UPDATE",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("UPDATE completato")

path = "C:/CSV_tesi/dataset_fixed_PREDICTION5.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=10, bounds_z=(5, 100), graphics="OFF",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="PREDICTION",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=5,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("PREDICTION5 completato")

path = "C:/CSV_tesi/dataset_fixed_PREDICTION10.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=10, bounds_z=(5, 100), graphics="OFF",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="PREDICTION",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("PREDICTION10 completato")
'''


'''
# COSTRUZIONE DEI GRAFICI A PARTIRE DAI CSV (TARGET FISSI)

#grafici OFF
path = "C:/CSV_tesi/dataset_fixed_OFF.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)


# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)




#grafici UPDATE
path = "C:/CSV_tesi/dataset_fixed_UPDATE.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


#TODO l'errore sta qui
# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)



#grafici PREDICTION5
path = "C:/CSV_tesi/dataset_fixed_PREDICTION5.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)


#grafici PREDICTION10
path = "C:/CSV_tesi/dataset_fixed_PREDICTION10.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''



'''
#     PROVE MONTECARLO DOVE SI VARIA LE POSIZIONI DEI TARGET FISSI          
#CONDIZIONI INIZIALI DEI DRONI FISSATE
initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20]])
#POSIZIONE INIZIALE DEI TARGET VARIABILE (NOTA: IN QUESTO CASO è FITTIZIA, SERVE SOLO PER DARE UNA DIMENSIONE ALL'ARRAY)
targets_positions = np.array([[+20, +20], [8, 7], [30, 0], [14, 38], [-5, -10]])

#CREAZIONE DEI CSV PER I TARGET FISSI FACENDO UNA MEDIA MONTECARLO PER RICAVARE I GRAFICI DI RMSE

########    COSTRUZIONE MULTIPLA DEI CSV (TARGET FISSI)

path = "C:/CSV_tesi/dataset_fixed_OFF.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=20, bounds_z=(5, 100), graphics="OFF",
                                                                 random_targets_position="ON",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="OFF",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("OFF completato")

path = "C:/CSV_tesi/dataset_fixed_UPDATE.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=20, bounds_z=(5, 100), graphics="OFF",
                                                                 random_targets_position="ON",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="UPDATE",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("UPDATE completato")



path = "C:/CSV_tesi/dataset_fixed_PREDICTION5.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=20, bounds_z=(5, 100), graphics="OFF",
                                                                 random_targets_position="ON",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="PREDICTION5",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=5,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("PREDICTION5 completato")


path = "C:/CSV_tesi/dataset_fixed_PREDICTION10.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=20, bounds_z=(5, 100), graphics="OFF",
                                                                 random_targets_position="ON",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="PREDICTION10",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path)

print("PREDICTION10 completato")
'''

'''
#                CONFIGURAZIONE PROBLEMA CON TARGET FISSI        
initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20]])
targets_positions = np.random.randint(-40, 41, size=(5,2))

#METODO IN CUI VIENE MOSTRATO UN SINGOLO ESPERIMENTO (target FISSI)
fixed_object_covariance, RMSE, h_trend, h_targets_trend = fixed_target_coverage_gradient_KF_multipleUpdate_prediction(
    initial_condition_drones, targets_positions,
    bounds_z=(5, 100), graphics="ON", G_function_parameter=2,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="PREDICTION", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=5, correction_in_viewing_range="ON")


#             Grafici per target in movimento
#H_plots_kalman(moving_object_covariance[0], moving_object_covariance[1], moving_object_covariance[2])
RMSE_plot(RMSE)
#h_trend_plots(h_trend)
h_targets_trend_plots(h_targets_trend)

'''


#todo USARE PER GRAFICI MONTECARLO, TARGET FISSI
'''
# METODO PER IL SALVATAGGIO DEI CSV (target FISSI) MONTECARLO
# utilizzato per ricavare i dataset: dataset_fixedMC_OFF.csv, dataset_fixedMC_UPDATE.csv, dataset_fixedMC_PREDICTION5.csv, dataset_fixedMC_PREDICTION10.csv

path = "C:/CSV_tesi/dataset_fixedMC_PREDICTION10.csv"

fixed_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(initial_condition_drones, targets_positions,
                                                                 N_iteration4csv=200, bounds_z=(5, 100), graphics="OFF",
                                                                 G_function_parameter=2,
                                                                 alpha_parameter=0.01,
                                                                 gradient_descent_method="rmsprop",
                                                                 learning_rate_parameter=0.2, KF="PREDICTION",
                                                                 sigma_noise=2, sigma_w=0.001,
                                                                 forward_prediction_steps=10,
                                                                 correction_in_viewing_range="ON", file_path=path,
                                                                 random_targets_position="ON")


df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''


#todo USARE PER FARE GRAFICI MONTECARLO, TARGET MOBILI
'''
# METODO PER IL SALVATAGGIO DEI CSV (target FISSI) MONTECARLO
path = "C:/CSV_tesi/dataset_configAMC_PREDICTION10.csv"


MONTECARLO_moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4, random_trajectories="ON",
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="PREDICTION", sigma_noise=2, sigma_w=0.001, forward_prediction_steps=10, correction_in_viewing_range="ON", file_path=path)


df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''


'''
#METODO MONTECARLO PER I DRONI, PER IL SALVATAGGIO DEI CSV

initial_condition_drones = np.array([[20, -20, 10], [-12, -16, 10], [15, -18, 25], [10, 35, 30], [-20, 20, 20]])
target_trajectories = [2, 5, 7, 10, 13, 17, 18]



path = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION5_dev05.csv"


DRONE_MONTECARLO_moving_target_coverage_gradient_KF_multipleUpdate_prediction_4csv(
    initial_condition_drones, target_trajectories, N_iteration4csv=200,
    bounds_z=(5, 100), graphics="OFF", G_function_parameter=4,
    alpha_parameter=0.01, gradient_descent_method="rmsprop", learning_rate_parameter=0.2,
    KF="PREDICTION", sigma_noise=0.5, sigma_w=0.001, forward_prediction_steps=5, correction_in_viewing_range="ON", file_path=path)


df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''


#per 1 grafico
'''
path = "C:/CSV_tesi/dataset_configC_PREDICTION10.csv"

df = pd.read_csv(path)
medie_rmse = calcola_media_rmse_per_time_step(df)
print(medie_rmse)
print(type(medie_rmse))
RMSE_plot(medie_rmse)


# Convertiamo la colonna 'h_target_instant' se necessario
if isinstance(df['h_target_instant'].iloc[0], str):
    df['h_target_instant'] = df['h_target_instant'].apply(eval)

# Calcoliamo le medie degli elementi di h_target_instant per ogni time_step
medie_h_target = calcola_media_h_target_per_time_step(df)

print(medie_h_target)
h_targets_trend_plots(medie_h_target)
'''


#per i grafici RMSE sovrapposti
'''
#todo per i casi dove ho droni crescenti devo rifare la funzione

path_OFF = "C:/CSV_tesi/dataset_configAMC_OFF.csv"
path_UPDATE = "C:/CSV_tesi/dataset_configAMC_UPDATE.csv"
path_PREDICTION5 = "C:/CSV_tesi/dataset_configAMC_PREDICTION5.csv"
path_PREDICTION10 = "C:/CSV_tesi/dataset_configAMC_PREDICTION10.csv"
df_OFF = pd.read_csv(path_OFF)
df_UPDATE = pd.read_csv(path_UPDATE)
df_PREDICTION5 = pd.read_csv(path_PREDICTION5)
df_PREDICTION10 = pd.read_csv(path_PREDICTION10)

mean_RMSE_OFF = calcola_media_rmse_per_time_step(df_OFF)
mean_RMSE_UPDATE = calcola_media_rmse_per_time_step(df_UPDATE)
mean_RMSE_PREDICTION5 = calcola_media_rmse_per_time_step(df_PREDICTION5)
mean_RMSE_PREDICTION10 = calcola_media_rmse_per_time_step(df_PREDICTION10)

RMSE_multiple_plots(mean_RMSE_OFF, mean_RMSE_UPDATE, mean_RMSE_PREDICTION5, mean_RMSE_PREDICTION10)
'''


'''
#confronto RMSE della progressione di più droni
path_2drones = "C:/CSV_tesi/dataset_configA_2droneMC_UPDATE.csv"
path_3drones = "C:/CSV_tesi/dataset_configA_3droneMC_UPDATE.csv"
path_4drones = "C:/CSV_tesi/dataset_configA_4droneMC_UPDATE.csv"
path_5drones = "C:/CSV_tesi/dataset_configA_5droneMC_UPDATE.csv"
path_6drones = "C:/CSV_tesi/dataset_configA_6droneMC_UPDATE.csv"
path_7drones = "C:/CSV_tesi/dataset_configA_7droneMC_UPDATE.csv"
path_8drones = "C:/CSV_tesi/dataset_configA_8droneMC_UPDATE.csv"
path_9drones = "C:/CSV_tesi/dataset_configA_9droneMC_UPDATE.csv"

df_2drones = pd.read_csv(path_2drones)
df_3drones = pd.read_csv(path_3drones)
df_4drones = pd.read_csv(path_4drones)
df_5drones = pd.read_csv(path_5drones)
df_6drones = pd.read_csv(path_6drones)
df_7drones = pd.read_csv(path_7drones)
df_8drones = pd.read_csv(path_8drones)
df_9drones = pd.read_csv(path_9drones)

mean_RMSE_2drones = calcola_media_rmse_per_time_step(df_2drones)
mean_RMSE_3drones = calcola_media_rmse_per_time_step(df_3drones)
mean_RMSE_4drones = calcola_media_rmse_per_time_step(df_4drones)
mean_RMSE_5drones = calcola_media_rmse_per_time_step(df_5drones)
mean_RMSE_6drones = calcola_media_rmse_per_time_step(df_6drones)
mean_RMSE_7drones = calcola_media_rmse_per_time_step(df_7drones)
mean_RMSE_8drones = calcola_media_rmse_per_time_step(df_8drones)
mean_RMSE_9drones = calcola_media_rmse_per_time_step(df_9drones)

RMSE_multiple_plots2(mean_RMSE_2drones, mean_RMSE_3drones, mean_RMSE_4drones, mean_RMSE_5drones, mean_RMSE_6drones, mean_RMSE_7drones, mean_RMSE_8drones, mean_RMSE_9drones)
'''



'''
path_2drones = "C:/CSV_tesi/dataset_configA_2droneMC_UPDATE.csv"
df_2drones = pd.read_csv(path_2drones)
#print(df_2drones.columns)
posizione_istantanea_droni = 'drone_position_instant'
tempo = df_2drones['time']
posizione_droni = df_2drones[posizione_istantanea_droni]
print(posizione_droni[0])

'''






'''
#todo sistemare
newlist_posizionedroni = []
for i in len(posizione_droni):
    new_posizione = fromStringToList(posizione_droni[i])
    print(new_posizione)
    newlist_posizionedroni.append(new_posizione)

print(newlist_posizionedroni)
    
#print(posizione_droni)
#lista_unita = list(zip(tempo, posizione_droni))
#print(lista_unita)


#new_posizione_droni = fromStringToList(posizione_droni)
#print(new_posizione_droni)
#print(type(new_posizione_droni))

'''
'''
path_2drones = "C:/CSV_tesi/dataset_configA_2droneMC_UPDATE.csv"
df_2drones = pd.read_csv(path_2drones)
# print(df_2drones.columns)
tempo = df_2drones['time']
posizione_droni = df_2drones['drone_position_instant']
# print(posizione_droni[0])
new_position_drones = fromStringToList(posizione_droni[0])
# print(len(new_position_drones))
# print(len(tempo))
reduced_dataset = []
for i in range(351):
    reduced_dataset.append([i])
# print(reduced_dataset)


n_total_z = int(len(new_position_drones) / 3)

for i in range(351):
    for j in range(n_total_z):
        reduced_dataset[i].append(0)

for i in range(len(tempo)):
    print(i, tempo[i])
    for j in range(351):
        if j == tempo[i]:
            new_position_drones = fromStringToList(posizione_droni[i])
            for z in range(n_total_z):
                reduced_dataset[j][z + 1] = reduced_dataset[j][z + 1] + new_position_drones[3*z + 2]


for i in range(351):
    for j in range(n_total_z):
        reduced_dataset[i][j + 1] = reduced_dataset[i][j + 1] / 200

print(reduced_dataset)
'''


'''
# CODICE UTILIZZATO PER EFFETTUARE UN CONFRONTO FRA LE ALTEZZE MEDIE DEI DRONI, ALL'AUMENTARE DEL NUMERO DI DRONI
matrice_altezze_ridotta2 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_2droneMC_UPDATE.csv")
matrice_altezze_ridotta3 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_3droneMC_UPDATE.csv")
matrice_altezze_ridotta4 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_4droneMC_UPDATE.csv")
matrice_altezze_ridotta5 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_UPDATE.csv")
matrice_altezze_ridotta6 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_6droneMC_UPDATE.csv")
matrice_altezze_ridotta7 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_7droneMC_UPDATE.csv")
matrice_altezze_ridotta8 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_8droneMC_UPDATE.csv")
matrice_altezze_ridotta9 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_9droneMC_UPDATE.csv")

mean2drones = heightMean(matrice_altezze_ridotta2)
mean3drones = heightMean(matrice_altezze_ridotta3)
mean4drones = heightMean(matrice_altezze_ridotta4)
mean5drones = heightMean(matrice_altezze_ridotta5)
mean6drones = heightMean(matrice_altezze_ridotta6)
mean7drones = heightMean(matrice_altezze_ridotta7)
mean8drones = heightMean(matrice_altezze_ridotta8)
mean9drones = heightMean(matrice_altezze_ridotta9)

fullMatrix_altezze_matrice = [mean2drones, mean3drones, mean4drones, mean5drones, mean6drones, mean7drones, mean8drones, mean9drones]

HeightPlots2(fullMatrix_altezze_matrice)

'''

'''
#confronto RMSE di UPDATE, PREDICTION1, ... , PREDICTION5
path_5drones_UPDATE = "C:/CSV_tesi/dataset_configA_5droneMC_UPDATE.csv"
path_5drones_PREDICTION1 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION1.csv"
path_5drones_PREDICTION2 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION2.csv"
path_5drones_PREDICTION3 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION3.csv"
path_5drones_PREDICTION4 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION4.csv"
path_5drones_PREDICTION5 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION5.csv"

df_5drones_UPDATE = pd.read_csv(path_5drones_UPDATE)
df_5drones_PREDICTION1 = pd.read_csv(path_5drones_PREDICTION1)
df_5drones_PREDICTION2 = pd.read_csv(path_5drones_PREDICTION2)
df_5drones_PREDICTION3 = pd.read_csv(path_5drones_PREDICTION3)
df_5drones_PREDICTION4 = pd.read_csv(path_5drones_PREDICTION4)
df_5drones_PREDICTION5 = pd.read_csv(path_5drones_PREDICTION5)


mean_RMSE_5drones_UPDATE = calcola_media_rmse_per_time_step(df_5drones_UPDATE)
mean_RMSE_5drones_PREDICTION1 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION1)
mean_RMSE_5drones_PREDICTION2 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION2)
mean_RMSE_5drones_PREDICTION3 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION3)
mean_RMSE_5drones_PREDICTION4 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION4)
mean_RMSE_5drones_PREDICTION5 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION5)

RMSE_multiple_plots3(mean_RMSE_5drones_UPDATE, mean_RMSE_5drones_PREDICTION1, mean_RMSE_5drones_PREDICTION2, mean_RMSE_5drones_PREDICTION3, mean_RMSE_5drones_PREDICTION4, mean_RMSE_5drones_PREDICTION5)
'''

'''
#ALTEZZE MEDIE DEI DRONI, CASO DI 5 DRONI IN CUI CONFRONTO LE STRATEGIE DA UPDATE A PREDICTION5
matrice_altezze_ridotta_UPDATE = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_UPDATE.csv")
matrice_altezze_ridotta_PREDICTION1 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION1.csv")
matrice_altezze_ridotta_PREDICTION2 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION2.csv")
matrice_altezze_ridotta_PREDICTION3 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION3.csv")
matrice_altezze_ridotta_PREDICTION4 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION4.csv")
matrice_altezze_ridotta_PREDICTION5 = Height_fromCsvToList("C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION5.csv")

mean5drones_UPDATE = heightMean(matrice_altezze_ridotta_UPDATE)
mean5drones_PREDICTION1 = heightMean(matrice_altezze_ridotta_PREDICTION1)
mean5drones_PREDICTION2 = heightMean(matrice_altezze_ridotta_PREDICTION2)
mean5drones_PREDICTION3 = heightMean(matrice_altezze_ridotta_PREDICTION3)
mean5drones_PREDICTION4 = heightMean(matrice_altezze_ridotta_PREDICTION4)
mean5drones_PREDICTION5 = heightMean(matrice_altezze_ridotta_PREDICTION5)

fullMatrix_altezze_matrice = [mean5drones_UPDATE, mean5drones_PREDICTION1, mean5drones_PREDICTION2, mean5drones_PREDICTION3, mean5drones_PREDICTION4, mean5drones_PREDICTION5]

HeightPlots3(fullMatrix_altezze_matrice)
'''

'''
#confronto RMSE di UPDATE, PREDICTION1, ... , PREDICTION5 nel caso in cui ho deviazione standard = 0,5
path_5drones_UPDATE = "C:/CSV_tesi/dataset_configA_5droneMC_UPDATE_dev05.csv"
path_5drones_PREDICTION1 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION1_dev05.csv"
path_5drones_PREDICTION2 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION2_dev05.csv"
path_5drones_PREDICTION3 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION3_dev05.csv"
path_5drones_PREDICTION4 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION4_dev05.csv"
path_5drones_PREDICTION5 = "C:/CSV_tesi/dataset_configA_5droneMC_PREDICTION5_dev05.csv"

df_5drones_UPDATE = pd.read_csv(path_5drones_UPDATE)
df_5drones_PREDICTION1 = pd.read_csv(path_5drones_PREDICTION1)
df_5drones_PREDICTION2 = pd.read_csv(path_5drones_PREDICTION2)
df_5drones_PREDICTION3 = pd.read_csv(path_5drones_PREDICTION3)
df_5drones_PREDICTION4 = pd.read_csv(path_5drones_PREDICTION4)
df_5drones_PREDICTION5 = pd.read_csv(path_5drones_PREDICTION5)


mean_RMSE_5drones_UPDATE = calcola_media_rmse_per_time_step(df_5drones_UPDATE)
mean_RMSE_5drones_PREDICTION1 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION1)
mean_RMSE_5drones_PREDICTION2 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION2)
mean_RMSE_5drones_PREDICTION3 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION3)
mean_RMSE_5drones_PREDICTION4 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION4)
mean_RMSE_5drones_PREDICTION5 = calcola_media_rmse_per_time_step(df_5drones_PREDICTION5)

RMSE_multiple_plots3(mean_RMSE_5drones_UPDATE, mean_RMSE_5drones_PREDICTION1, mean_RMSE_5drones_PREDICTION2, mean_RMSE_5drones_PREDICTION3, mean_RMSE_5drones_PREDICTION4, mean_RMSE_5drones_PREDICTION5, title_plot="Evolution of the RMSE of 5 drones with different predictive strategies, standard deviation 0,5")

'''











