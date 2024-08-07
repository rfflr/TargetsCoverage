import random
from random import randrange as rg
import sys
import functionDrone as fD
import functionTarget as fT
import numpy as np
import math
import scipy.optimize
import matplotlib.pyplot as plt
import time
import pickle
import pygame
import kalmanFilter

with open('C:\PyProject\MasterThesisProject\dataset', 'rb') as file:
    dataset = pickle.load(file)

#### versione in cui è integrata anche la h_tilde ma con i target fissi


"Funzioni iniziali utili per definire il problema"


# funcG è una funzione d'informazione della posizione del target che è inversamente proporzionale all'altezza del drone
def funcG(z, zmax, G_function_parameter):
    # zmax = 50
    Gnum = math.pow(zmax - z, G_function_parameter)
    Gden = math.pow(zmax, G_function_parameter)
    G = Gnum / Gden
    return G


# funcF è una funzione d'informazione della posizione del targetche è inversamente proporzionale alla distanza dal target (proiettata sul piano)
def funcF(px, py, qx, qy, radius, F_function_parameter):
    Fnum = math.pow(radius - math.hypot(px - qx, py - qy), F_function_parameter)
    Fden = math.pow(radius, F_function_parameter)
    F = float(Fnum / Fden)
    return F


def func_h_tilde(px, py, qx, qy, radius, h_tilde_max):
    h_tilde_num = radius
    h_tilde_den = math.hypot(px - qx, py - qy)
    h_tilde = h_tilde_num / h_tilde_den
    # if h_tilde >= h_tilde_max:
    #    h_tilde = h_tilde_max
    return h_tilde


def R_functionWRONG(x_d, y_d, z_d, z_max, x_t_true, y_t_true, sigma_noise, G_function_parameter, F_function_parameter,
               radius, h_tilde_max, alpha):
    G = funcG(z_d, z_max, G_function_parameter)
    if(radius - math.hypot(x_d - x_t_true, y_d - y_t_true))>0:
        F = funcF(x_d, y_d, x_t_true, y_t_true, radius, F_function_parameter)
        h = G * F + alpha
        sigma_ij = sigma_noise / h
    else:
        h = 0.0001
        sigma_ij = 0
        #h = alpha * func_h_tilde(x_d, y_d, x_t_true, y_t_true, radius, h_tilde_max)
    R_ij = np.array([[sigma_ij**2, 0], [0, sigma_ij**2]])
    return R_ij, sigma_ij

def R_function(x_d, y_d, z_d, z_max, x_t_true, y_t_true, sigma_noise, G_function_parameter, F_function_parameter,
               radius, h_tilde_max, alpha):
    G = funcG(z_d, z_max, G_function_parameter)
    if(radius - math.hypot(x_d - x_t_true, y_d - y_t_true))>0:
        F = funcF(x_d, y_d, x_t_true, y_t_true, radius, F_function_parameter)
        h = G * F + alpha
    else:
        h = alpha * func_h_tilde(x_d, y_d, x_t_true, y_t_true, radius, h_tilde_max)
    if h == 0:
        h = 0.0001
    sigma_ij_square = (sigma_noise) / h
    #sigma_ij_square = (sigma_noise**2)/h
    R_ij = np.array([[sigma_ij_square, 0], [0, sigma_ij_square]])
    sigma_ij = math.sqrt(sigma_ij_square)
    return R_ij, sigma_ij





# funcH è la funzione obiettivo che voglio minimizzare
# la funzione prende in ingresso le variabili px, py, pz di ogni drone, e come costanti ho il theta (angolo di visione)
# per ogni drone e le posizioni qx, qy di ogni target.
# la funzione restituisce un float H che è ottunuto tramite sum_y (1/(sum_x h_ij)) ovvero la somma delle covarianze sui target,
# dove h_ij è la funzione d'informazione dell'i-esimo drone a proposito del j-esimo target
# Nota: la funzione h_ij = g * f, che è ottenuta in questo caso attraverso h_ij = funcG(...) * funcF(...)
def funcH_2d(variables, constants):
    n = int((len(variables) / 3))
    m = int((len(constants[1]) / 2))
    h = np.zeros((n, m))
    G_function_parameter = constants[3][0]
    F_function_parameter = constants[3][1]
    h_tilde_max = constants[3][2]
    alpha = constants[3][3]
    z_max = constants[3][4]
    i = 0
    while i < n:
        x_drone_i = variables[3 * i]
        y_drone_i = variables[3 * i + 1]
        z_drone_i = variables[3 * i + 2]
        theta_drone_i = constants[0][i]
        g = funcG(z_drone_i, z_max, G_function_parameter)
        radius = z_drone_i * math.tan(theta_drone_i / 2)
        j = 0
        while j < m:
            # todo distanza euclidea (non cambia niente)
            x_target_j = constants[1][2 * j]
            y_target_j = constants[1][2 * j + 1]
            if (radius - math.hypot(x_drone_i - x_target_j, y_drone_i - y_target_j)) > 0:
                f = funcF(x_drone_i, y_drone_i, x_target_j, y_target_j, radius, F_function_parameter)
                # h[i, j] = g * f
                h[i, j] = g * f + alpha
            else:
                h[i, j] = alpha * func_h_tilde(x_drone_i, y_drone_i, x_target_j, y_target_j, radius, h_tilde_max)
            j = j + 1
        i = i + 1
    # fino a qui il codice è testato e funziona
    # adesso devo valutare la seconda parte del codice, ovvero dove sommo tutti gli h_ij secondo un certo criterio,
    # tale per effettuare la minimizzazione
    # print(h)
    j = 0
    i = 0
    overall_covariance = 0
    while j < m:
        h_j = 0
        while i < n:
            h_j = h_j + h[i, j]
            i = i + 1
        i = 0
        # questo if è presente nel caso un target j-esimo non fosse identificato da nessun drone
        if h_j == 0 or h_j < 0:
            # questa sotto è una riga di codice fondamentale
            h_j = 0.0001
        overall_covariance = overall_covariance + 1 / h_j
        j = j + 1
    # print(overall_covariance)
    return overall_covariance


# funzione che mi fornisce il valore della covarianza relativa ad ogni target
def targetCovariance(variables, constants):
    n = int((len(variables) / 3))
    m = int((len(constants[1]) / 2))
    h = np.zeros((n, m))
    G_function_parameter = constants[3][0]
    F_function_parameter = constants[3][1]
    h_tilde_max = constants[3][2]
    alpha = constants[3][3]
    z_max = constants[3][4]
    i = 0
    while i < n:
        x_drone_i = variables[3 * i]
        y_drone_i = variables[3 * i + 1]
        z_drone_i = variables[3 * i + 2]
        theta_drone_i = constants[0][i]
        g = funcG(z_drone_i, z_max, G_function_parameter)
        radius = z_drone_i * math.tan(theta_drone_i / 2)
        j = 0
        while j < m:
            x_target_j = constants[1][2 * j]
            y_target_j = constants[1][2 * j + 1]
            if (radius - math.hypot(x_drone_i - x_target_j, y_drone_i - y_target_j)) > 0:
                f = funcF(x_drone_i, y_drone_i, x_target_j, y_target_j, radius, F_function_parameter)
                # h[i, j] = g * f
                h[i, j] = g * f + alpha
            else:
                h[i, j] = alpha * func_h_tilde(x_drone_i, y_drone_i, x_target_j, y_target_j, radius, h_tilde_max)
            j = j + 1
        i = i + 1
    # fino a qui il codice è testato e funziona
    # adesso devo valutare la seconda parte del codice, ovvero dove sommo tutti gli h_ij secondo un certo criterio,
    # tale per effettuare la minimizzazione
    # print(h)
    j = 0
    i = 0
    target_covariance = []
    while j < m:
        h_j = 0
        while i < n:
            h_j = h_j + h[i, j]
            i = i + 1
        i = 0
        # questo if è presente nel caso un target j-esimo non fosse identificato da nessun drone
        if h_j == 0 or h_j < 0:
            # questa sotto è una riga di codice fondamentale
            h_j = 0.0001
        j_target_covariance = 1 / h_j
        target_covariance.append(j_target_covariance)
        j = j + 1
    return target_covariance


# funzione vecchia, non più utile
def funcH_2d_psDes(variables, constants):
    n = int((len(variables) / 3))
    m = int((len(constants[1]) / 2))
    h = np.zeros((n, m))
    alpha = 0.0001
    i = 0
    while i < n:
        g = funcG(variables[3 * i + 2], constants[3][4], constants[3][0])
        radius = variables[3 * i + 2] * math.tan(constants[0][i] / 2)
        radius_true = constants[2][3 * i + 2] * math.tan(constants[0][i] / 2)
        distance_pReal_pOpt = math.hypot(constants[2][3 * i] - variables[3 * i],
                                         constants[2][3 * i + 1] - variables[3 * i + 1])
        total_radius = radius_true + radius
        j = 0
        while j < m:
            if (radius - math.hypot(variables[3 * i] - constants[1][2 * j],
                                    variables[3 * i + 1] - constants[1][2 * j + 1])) > 0:
                f = funcF(variables[3 * i], variables[3 * i + 1], constants[1][2 * j], constants[1][2 * j + 1], radius,
                          constants[3][1])
                h[i, j] = g * f + alpha
                # h[i, j] = g * f + alpha
            else:
                # h[i, j] = alpha * (radius / math.hypot(variables[3 * i] - constants[1][2 * j],
                #                                  variables[3 * i + 1] - constants[1][2 * j + 1]))
                h[i, j] = alpha * func_h_tilde(variables[3 * i], variables[3 * i + 1], constants[1][2 * j],
                                               constants[1][2 * j + 1], radius, constants[3][2])
                # h[i, j] = alpha * func_h_tilde(constants[2][3 * i], constants[2][3 * i + 1], constants[1][2 * j], constants[1][2 * j + 1], constants[2][3 * i + 2] * math.tan(constants[0][i] / 2))
            j = j + 1
        i = i + 1
    # fino a qui il codice è testato e funziona
    # adesso devo valutare la seconda parte del codice, ovvero dove sommo tutti gli h_ij secondo un certo criterio,
    # tale per effettuare la minimizzazione
    # print(h)
    j = 0
    i = 0
    H = 0
    while j < m:
        h_j = 0
        while i < n:
            h_j = h_j + h[i, j]
            i = i + 1
        i = 0
        # questo if è presente nel caso un target j-esimo non fosse identificato da nessun drone
        if h_j <= 0:
            # questa sotto è una riga di codice fondamentale
            h_j = 0.0001
        H = H + 1 / h_j
        j = j + 1
    # print(H)
    return H, h


# funzione vecchia, non più utile
def funcH_2d_psTrue(variables, constants):
    n = int((len(variables) / 3))
    m = int((len(constants[1]) / 2))
    h = np.zeros((n, m))
    alpha = 0.0001
    i = 0
    while i < n:
        g = funcG(variables[3 * i + 2], constants[3][4], constants[3][0])
        radius = variables[3 * i + 2] * math.tan(constants[0][i] / 2)
        j = 0
        while j < m:
            if (radius - math.hypot(variables[3 * i] - constants[1][2 * j],
                                    variables[3 * i + 1] - constants[1][2 * j + 1])) > 0:
                f = funcF(variables[3 * i], variables[3 * i + 1], constants[1][2 * j], constants[1][2 * j + 1], radius,
                          constants[3][1])
                h[i, j] = g * f + alpha
            else:
                # h[i, j] = 0
                # h[i, j] = alpha * (radius / math.hypot(variables[3 * i] - constants[1][2 * j],
                #                                  variables[3 * i + 1] - constants[1][2 * j + 1]))
                h[i, j] = alpha * func_h_tilde(variables[3 * i], variables[3 * i + 1], constants[1][2 * j],
                                               constants[1][2 * j + 1], radius, constants[3][2])
                # h[i, j] = alpha * func_h_tilde(constants[2][3 * i], constants[2][3 * i + 1], constants[1][2 * j], constants[1][2 * j + 1], constants[2][3 * i + 2] * math.tan(constants[0][i] / 2))
            j = j + 1
        i = i + 1
    # fino a qui il codice è testato e funziona
    # adesso devo valutare la seconda parte del codice, ovvero dove sommo tutti gli h_ij secondo un certo criterio,
    # tale per effettuare la minimizzazione
    # print(h)
    j = 0
    i = 0
    H = 0
    while j < m:
        h_j = 0
        while i < n:
            h_j = h_j + h[i, j]
            i = i + 1
        i = 0
        # questo if è presente nel caso un target j-esimo non fosse identificato da nessun drone
        if h_j <= 0:
            # questa sotto è una riga di codice fondamentale
            h_j = 0.0001
        H = H + 1 / h_j
        j = j + 1
    # print(H)
    return H, h


def funcH_Testing(variables, constants):
    # Parte 1: viene costruita la matrice d'informazione h
    n = int((len(variables) / 3))
    m = int((len(constants[1]) / 2))
    h = np.zeros((n, m))
    G_function_parameter = constants[3][0]
    F_function_parameter = constants[3][1]
    h_tilde_max = constants[3][2]
    alpha = constants[3][3]
    z_max = constants[3][4]
    i = 0
    while i < n:
        x_drone_i = variables[3 * i]
        y_drone_i = variables[3 * i + 1]
        z_drone_i = variables[3 * i + 2]
        theta_drone_i = constants[0][i]
        g = funcG(z_drone_i, z_max, G_function_parameter)
        radius = z_drone_i * math.tan(theta_drone_i / 2)
        j = 0
        while j < m:
            x_target_j = constants[1][2 * j]
            y_target_j = constants[1][2 * j + 1]
            if (radius - math.hypot(x_drone_i - x_target_j, y_drone_i - y_target_j)) > 0:
                f = funcF(x_drone_i, y_drone_i, x_target_j, y_target_j, radius, F_function_parameter)
                h[i, j] = g * f
            else:
                h[i, j] = 0
            j = j + 1
        i = i + 1
    # Parte 2: viene effettuato il calcolo della covarianza complessiva
    j = 0
    i = 0
    overall_covariance = 0
    while j < m:
        h_j = 0
        while i < n:
            h_j = h_j + h[i, j]
            i = i + 1
        i = 0
        # if è presente nel caso un target j-esimo non fosse identificato da nessun drone: per evitare termini infiniti
        if h_j == 0:
            h_j = 0.0001
        overall_covariance = overall_covariance + 1 / h_j
        j = j + 1
    return overall_covariance


def comparisonFunction(h_test, h_true, optimized_position, true_position):
    num_rows, num_columns = h_test.shape
    i = 0
    j = 0
    while j < num_columns:
        sum_column_h_test = 0
        sum_column_h_true = 0
        while i < num_rows:
            sum_column_h_test = sum_column_h_test + h_test[i, j]
            sum_column_h_true = sum_column_h_true + h_true[i, j]
            i = i + 1
        if sum_column_h_true > sum_column_h_test:
            # print("sum_column_h_test: ", sum_column_h_test, "sum_column_h_true: ", sum_column_h_true)
            return true_position
        i = 0
        j = j + 1
    return optimized_position


def fixed_target_coverage(initial_condition_drones, targets_positions, bounds_x=(-40, +40), bounds_y=(-40, +40),
                          bounds_z=(1, 50), graphics="OFF", G_function_parameter=3, F_function_parameter=1,
                          h_tilde_max=1, alpha_parameter=0.0001):
    '''
    -esempio su come vanno inserite le posizioni iniziali dei target:
            -   initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
            -   targets_positions = np.array([[-20, -20], [8, 7], [20, 20], [25, 30], [-25, +35]]);

    -   si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    -   la visualizzazione grafica è inizialmente disattivata: dato che non c'è alcuna informazione stampata a console per  adesso conviene impostare:      graphics = "ON" ;

    -   il processo dura nel complesso 300 time step.

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde.
    '''

    ''' 
        In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                    trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
    '''

    H_plots = [[], [], [], []]
    n_drones = initial_condition_drones.shape[0]
    m_targets = targets_positions.shape[0]
    drone = []
    target = []
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    for i_target in range(m_targets):
        target.append(fT.Target(targets_positions[i_target][0],
                                targets_positions[i_target][1]))  # il target viene definito per x e y

    '''variables rappresenta la variable da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]

    '''definisco i vincoli del problema di ottimizzazione (i bounds)'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 400
        y0_window = 400
        delta_window = 10
    time_step = 0
    while time_step < 300:
        ''' definisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_target = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        for i_target in range(m_targets):
            constant_target.append(target[i_target].x)
            constant_target.append(target[i_target].y)
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        """ Valutare se inserire:
        true_positions_drones = np.array((drone1.p[0], drone1.p[1], drone1.p[2], drone2.p[0], drone2.p[1], drone2.p[2]))
        H_real.append(funcH_2d(true_positions_drones, constants))
        H_true, h_true = funcH_2d_psTrue(true_positions_drones, constants)
        drone_p_before_opt = [drone1.p[0], drone1.p[1], drone1.p[2], drone2.p[0], drone2.p[1], drone2.p[2]]
        """
        # viene eseguita l'ottimizzazione
        result_optimization = scipy.optimize.minimize(funcH_2d, variables, args=constants, bounds=bounds,
                                                      method='SLSQP')
        # print(result_optimization.x)
        ''' a questo punto vado a cambiare le condizioni iniziali per l'iterazione successiva:
                come condizioni iniziali per l'ottimizzazione devo mette la soluzione ottenuta dall'ottimizzazione '''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_optimization.x.shape[0]):
            variables[i_drones_result] = result_optimization.x[i_drones_result]

        matriceTest = funcH_2d(variables, constants)

        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''

        for i_drones in range(n_drones):
            drone[i_drones].controlUpdate(np.array([result_optimization.x[3 * i_drones],
                                                    result_optimization.x[3 * i_drones + 1],
                                                    result_optimization.x[3 * i_drones + 2]]))
            drone[i_drones].p = drone[i_drones].stateUpdate()

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            random.seed(10)
            # disegno i droni
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            # disegno i target
            for i_target in range(m_targets):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target].x - 10,
                                  y0_window + delta_window * target[i_target].y - 10,
                                  20, 20))
                # pygame.display.update()
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede
        # print("iterazione finita")
        time_step += 1
    # quando si chiude il ciclo viene chiuso lo schermo
    pygame.display.quit()
    pygame.quit()
    print("fine ciclo")
    return H_plots


def moving_target_coverage(initial_condition_drones, targets_trajectories, bounds_x=(-40, +40), bounds_y=(-40, +40),
                           bounds_z=(1, 50), graphics="OFF", G_function_parameter=3, F_function_parameter=1,
                           h_tilde_max=1, alpha_parameter=0.0001):
    '''
    - esempio su come vanno inserite le posizioni iniziali dei droni:
            initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
    - esempio su come vanno inserite le traiettorie dei target:
            target_trajectories = [1, 2, 3, 5, 7]
            vedo che in questo caso inserisco solo l'indice del dataset delle traiettorie;

    - si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    - la visualizzazione grafica è inizialmente disattivata: dato che non c'è alcuna informazione stampata a console per  adesso conviene impostare:      graphics = "ON" ;

    -   il processo dura nel complesso 300 time step.

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde.
    '''

    ''' 
            In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                    H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                    H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                    H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                        trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                    H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
        '''

    H_plots = [[], [], [], []]

    n_drones = initial_condition_drones.shape[0]
    m_targets_trajectories = len(targets_trajectories)
    drone = []
    target = []
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    for i_target in range(m_targets_trajectories):
        target.append([dataset[targets_trajectories[i_target]][0][:, 0] * (1 / 10),
                       dataset[targets_trajectories[i_target]][0][:, 1] * (
                               1 / 10)])  # il target viene definito per x e y
    '''Definisco le liste H_real e H_desired che sono, rispettivamente, la covarianza complessiva reale del problema e la covarianza complessiva
            che otterrei se la configurazione dei droni fosse, in ogni istante, quella frutto dell'ottimizzazione'''
    # print(target[1][0][10])
    '''variables rappresenta l'insieme delle variabili da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]

    '''definisco i vincoli del problema di ottimizzazione (i bounds)'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 200
        y0_window = 200
        delta_window = 10

    # inizio l'iterazione per il coverage
    for i_time_step in range(target[1][0].shape[0]):
        ''' definisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_target = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        for i_target in range(m_targets_trajectories):
            constant_target.append(target[i_target][0][i_time_step])
            constant_target.append(target[i_target][1][i_time_step])
            # print(target[i_target][0][i_time_step], target[i_target][1][i_time_step])
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        # viene eseguita l'ottimizzazione
        result_optimization = scipy.optimize.minimize(funcH_2d, variables, args=constants, bounds=bounds,
                                                      method='SLSQP')

        ''' annullo il vettore variables e ci inserisco i nuovi valori, ovvero quelli frutto dell'ottimizzazione'''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_optimization.x.shape[0]):
            variables[i_drones_result] = result_optimization.x[i_drones_result]

        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''

        for i_drones in range(n_drones):
            drone[i_drones].controlUpdate(np.array([result_optimization.x[3 * i_drones],
                                                    result_optimization.x[3 * i_drones + 1],
                                                    result_optimization.x[3 * i_drones + 2]]))
            drone[i_drones].p = drone[i_drones].stateUpdate()

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            random.seed(10)
            # disegno i droni
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            # disegno i target
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target][0][i_time_step] - 10,
                                  y0_window + delta_window * target[i_target][1][i_time_step] - 10,
                                  20, 20))
                # pygame.display.update()
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede
        # print("iterazione finita")

        # quando si chiude il ciclo viene chiuso lo schermo
    return H_plots
    pygame.display.quit()
    pygame.quit()


# Calcolare il gradiente numerico usando differenze finite rispetto ad A
def numerical_gradient(func, variables, constants, h=1e-5):
    grad = np.zeros_like(variables)
    for i in range(len(variables)):
        variables_h1 = np.array(variables, dtype=float)
        variables_h2 = np.array(variables, dtype=float)
        variables_h1[i] += h
        variables_h2[i] -= h
        grad[i] = (func(variables_h1, constants) - func(variables_h2, constants)) / (2 * h)
    return grad


##### Control Update: -gradient_descent, -gradient_descent2, -adam, -rmsprop
# Implementare l'algoritmo di discesa del gradiente
def gradient_descent(func, A_init, B, learning_rate=0.01, num_iterations=100):
    A = np.array(A_init, dtype=float)
    for i in range(num_iterations):
        grad = numerical_gradient(func, A, B)
        A -= learning_rate * grad
        # print(f"Iterazione {i+1}: A = {A}, grad = {grad}")
    return A


# Implementare l'algoritmo di discesa del gradiente con decadimento del tasso di apprendimento e momentum
def gradient_descent2(func, A_init, B, learning_rate=0.1, num_iterations=1000, decay_rate=0.99, momentum=0.9):
    A = np.array(A_init, dtype=float)
    velocity = np.zeros_like(A)
    for i in range(num_iterations):
        grad = numerical_gradient(func, A, B)
        velocity = momentum * velocity - learning_rate * grad
        A += velocity
        learning_rate *= decay_rate
        # if i % 100 == 0:  # Stampa ogni 100 iterazioni
        #    print(f"Iterazione {i+1}: A = {A}, grad = {grad}, learning_rate = {learning_rate}")
    return A


# Implementare l'algoritmo di Adam
def adam(func, A_init, B, learning_rate=0.001, num_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    A = np.array(A_init, dtype=float)
    m = np.zeros_like(A)  # Primo momento (media dei gradienti)
    v = np.zeros_like(A)  # Secondo momento (media dei quadrati dei gradienti)
    t = 0  # Contatore delle iterazioni

    for i in range(num_iterations):
        t += 1
        grad = numerical_gradient(func, A, B)

        # Aggiornare il primo momento
        m = beta1 * m + (1 - beta1) * grad

        # Aggiornare il secondo momento
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Correzione del bias
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Aggiornamento dei parametri
        A -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return A


# Implementare l'algoritmo RMSProp con limiti
def rmsprop(func, A_init, B, learning_rate=0.01, num_iterations=1000, decay_rate=0.9, epsilon=1e-8, lower_bound=None,
            upper_bound=None):
    A = np.array(A_init, dtype=float)
    Eg2 = np.zeros_like(A)  # Media mobile esponenziale dei quadrati dei gradienti

    for i in range(num_iterations):
        grad = numerical_gradient(func, A, B)

        # Aggiornare la media mobile esponenziale dei quadrati dei gradienti
        Eg2 = decay_rate * Eg2 + (1 - decay_rate) * (grad ** 2)

        # Aggiornare i parametri
        A -= learning_rate * grad / (np.sqrt(Eg2) + epsilon)

        # Applicare il clipping ai parametri
        if lower_bound is not None:
            A = np.maximum(A, lower_bound)
        if upper_bound is not None:
            A = np.minimum(A, upper_bound)

        # Stampa ogni 100 iterazioni
        #if i % 100 == 0:
        #    print(f"Iterazione {i + 1}: A = {A}, grad = {grad}")

    return A


def fixed_target_coverage_gradient(initial_condition_drones, targets_positions, bounds_x=(-40, +40),
                                   bounds_y=(-40, +40),
                                   bounds_z=(1, 50), graphics="OFF", G_function_parameter=3, F_function_parameter=1,
                                   h_tilde_max=1, alpha_parameter=0.0001, learning_rate_parameter=0.1,
                                   gradient_descent_method="gradient_descent", total_time_step=700):
    '''
    -esempio su come vanno inserite le posizioni iniziali dei target:
            -   initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
            -   targets_positions = np.array([[-20, -20], [8, 7], [20, 20], [25, 30], [-25, +35]]);

    -   si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    -   la visualizzazione grafica è inizialmente disattivata, per attivarla impostare:      graphics = "ON" ;

    -   il processo dura nel complesso 700 time step. Può essere modificato variando total_time_step

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde.
    -il control update è eseguito con un metodo di discesa del gradiente. Complessivamente i metodi utilizzabili sono 4. Si può modificarlo col parametro gradient_descent_method = "...":
            -   "gradient_descent" -> metodo del gradiente di discesa impostato automaticamente, si può variare solo il learning_rate;
            -   "gradient_descent2" -> simile a gradient_descent con l'aggiunta del decadimento del tasso di apprendimento ed un momentum;
            -   "rmsprop" -> Root Mean Square Propagation, metodo con tasso di apprendimento adattivo;
            -   "adam" -> Adaptive Moment Estimation, estensione di RMSProp che tiene conto della media mobile dei momenti del primo e del secondo ordine del gradiente;
    '''

    ''' 
        In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                    trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
    '''

    H_plots = [[], [], [], []]
    n_drones = initial_condition_drones.shape[0]
    m_targets = targets_positions.shape[0]
    drone = []
    target = []
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    for i_target in range(m_targets):
        target.append(fT.Target(targets_positions[i_target][0],
                                targets_positions[i_target][1]))  # il target viene definito per x e y

    '''variables rappresenta la variable da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]

    '''definisco i vincoli del problema di ottimizzazione (i bounds)'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 400
        y0_window = 400
        delta_window = 10
    time_step = 0
    while time_step < total_time_step:
        ''' definisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_target = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        for i_target in range(m_targets):
            constant_target.append(target[i_target].x)
            constant_target.append(target[i_target].y)
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        """ Valutare se inserire:
        true_positions_drones = np.array((drone1.p[0], drone1.p[1], drone1.p[2], drone2.p[0], drone2.p[1], drone2.p[2]))
        H_real.append(funcH_2d(true_positions_drones, constants))
        H_true, h_true = funcH_2d_psTrue(true_positions_drones, constants)
        drone_p_before_opt = [drone1.p[0], drone1.p[1], drone1.p[2], drone2.p[0], drone2.p[1], drone2.p[2]]
        """
        # viene eseguita la discesa del gradiente, si può scegliere tra più metodi
        if gradient_descent_method == "gradient_descent":
            result_gradient_descent = gradient_descent(funcH_2d, variables, constants, num_iterations=1,
                                                       learning_rate=learning_rate_parameter)
        if gradient_descent_method == "gradient_descent2":
            result_gradient_descent = gradient_descent2(funcH_2d, variables, constants, num_iterations=1,
                                                        learning_rate=learning_rate_parameter)
        if gradient_descent_method == "adam":
            result_gradient_descent = adam(funcH_2d, variables, constants, num_iterations=1,
                                           learning_rate=learning_rate_parameter)
        if gradient_descent_method == "rmsprop":
            result_gradient_descent = rmsprop(funcH_2d, variables, constants, num_iterations=1,
                                              learning_rate=learning_rate_parameter)

        ''' a questo punto vado a cambiare le condizioni iniziali per l'iterazione successiva:
                come condizioni iniziali per l'ottimizzazione devo mette la soluzione ottenuta dall'ottimizzazione '''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_gradient_descent.shape[0]):
            variables[i_drones_result] = result_gradient_descent[i_drones_result]

        matriceTest = funcH_2d(variables, constants)

        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''
        # devo modificare questa fase in modo che vari direttamente la posizione del drone
        for i_drones in range(n_drones):
            drone[i_drones].p[0] = result_gradient_descent[3 * i_drones]
            drone[i_drones].p[1] = result_gradient_descent[3 * i_drones + 1]
            drone[i_drones].p[2] = result_gradient_descent[3 * i_drones + 2]

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            random.seed(10)
            # disegno i droni
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            # disegno i target
            for i_target in range(m_targets):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target].x - 10,
                                  y0_window + delta_window * target[i_target].y - 10,
                                  20, 20))
                # pygame.display.update()
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede
        # print("iterazione finita")
        time_step += 1
    # quando si chiude il ciclo viene chiuso lo schermo
    pygame.display.quit()
    pygame.quit()
    print("fine ciclo")
    return H_plots


def moving_target_coverage_gradient(initial_condition_drones, targets_trajectories, bounds_x=(-40, +40),
                                    bounds_y=(-40, +40),
                                    bounds_z=(1, 50), graphics="OFF", G_function_parameter=3, F_function_parameter=1,
                                    h_tilde_max=1, alpha_parameter=0.0001, learning_rate_parameter=0.1,
                                    gradient_descent_method="gradient_descent"):
    '''
    - esempio su come vanno inserite le posizioni iniziali dei droni:
            initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
    - esempio su come vanno inserite le traiettorie dei target:
            target_trajectories = [1, 2, 3, 5, 7]
            vedo che in questo caso inserisco solo l'indice del dataset delle traiettorie;

    - si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    - la visualizzazione grafica è inizialmente disattivata: dato che non c'è alcuna informazione stampata a console per  adesso conviene impostare:      graphics = "ON" ;

    -   il processo dura nel complesso 350 time step.

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde;
    -il control update è eseguito con un metodo di discesa del gradiente. Complessivamente i metodi utilizzabili sono 4. Si può modificarlo col parametro gradient_descent_method = "...":
            -   "gradient_descent" -> metodo del gradiente di discesa impostato automaticamente, si può variare solo il learning_rate;
            -   "gradient_descent2" -> simile a gradient_descent con l'aggiunta del decadimento del tasso di apprendimento ed un momentum;
            -   "rmsprop" -> Root Mean Square Propagation, metodo con tasso di apprendimento adattivo;
            -   "adam" -> Adaptive Moment Estimation, estensione di RMSProp che tiene conto della media mobile dei momenti del primo e del secondo ordine del gradiente;
    '''

    ''' 
            In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                    H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                    H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                    H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                        trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                    H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
        '''

    H_plots = [[], [], [], []]

    n_drones = initial_condition_drones.shape[0]
    m_targets_trajectories = len(targets_trajectories)
    drone = []
    target = []
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    for i_target in range(m_targets_trajectories):
        target.append([dataset[targets_trajectories[i_target]][0][:, 0] * (1 / 10),
                       dataset[targets_trajectories[i_target]][0][:, 1] * (
                               1 / 10)])  # il target viene definito per x e y
    '''Definisco le liste H_real e H_desired che sono, rispettivamente, la covarianza complessiva reale del problema e la covarianza complessiva
            che otterrei se la configurazione dei droni fosse, in ogni istante, quella frutto dell'ottimizzazione'''
    # print(target[1][0][10])
    '''variables rappresenta l'insieme delle variabili da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]

    '''definisco i vincoli del problema di ottimizzazione (i bounds)'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 400
        y0_window = 400
        delta_window = 10

    # inizio l'iterazione per il coverage
    for i_time_step in range(target[1][0].shape[0]):
        ''' definisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_target = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        for i_target in range(m_targets_trajectories):
            constant_target.append(target[i_target][0][i_time_step])
            constant_target.append(target[i_target][1][i_time_step])
            # print(target[i_target][0][i_time_step], target[i_target][1][i_time_step])
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        # viene eseguita la discesa del gradiente, si può scegliere tra più metodi
        if gradient_descent_method == "gradient_descent":
            result_gradient_descent = gradient_descent(funcH_2d, variables, constants, num_iterations=1,
                                                       learning_rate=learning_rate_parameter)
        if gradient_descent_method == "gradient_descent2":
            result_gradient_descent = gradient_descent2(funcH_2d, variables, constants, num_iterations=1,
                                                        learning_rate=learning_rate_parameter)
        if gradient_descent_method == "adam":
            result_gradient_descent = adam(funcH_2d, variables, constants, num_iterations=1,
                                           learning_rate=learning_rate_parameter)
        if gradient_descent_method == "rmsprop":
            result_gradient_descent = rmsprop(funcH_2d, variables, constants, num_iterations=1,
                                              learning_rate=learning_rate_parameter)

        ''' annullo il vettore variables e ci inserisco i nuovi valori, ovvero quelli frutto dell'ottimizzazione'''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_gradient_descent.shape[0]):
            variables[i_drones_result] = result_gradient_descent[i_drones_result]

        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''

        for i_drones in range(n_drones):
            drone[i_drones].p[0] = result_gradient_descent[3 * i_drones]
            drone[i_drones].p[1] = result_gradient_descent[3 * i_drones + 1]
            drone[i_drones].p[2] = result_gradient_descent[3 * i_drones + 2]

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            random.seed(10)
            # disegno i droni
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            # disegno i target
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target][0][i_time_step] - 10,
                                  y0_window + delta_window * target[i_target][1][i_time_step] - 10,
                                  20, 20))
                # pygame.display.update()
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede
        # print("iterazione finita")
        # quando si chiude il ciclo viene chiuso lo schermo
    return H_plots
    pygame.display.quit()
    pygame.quit()


"con il kalman filter"


def moving_target_coverage_gradient_KF(initial_condition_drones, targets_trajectories, bounds_x=(-40, +40),
                                       bounds_y=(-40, +40),
                                       bounds_z=(1, 50), graphics="OFF", G_function_parameter=3, F_function_parameter=1,
                                       h_tilde_max=1, alpha_parameter=0.0001, learning_rate_parameter=0.1,
                                       gradient_descent_method="gradient_descent", KF="OFF", sigma_noise=1.5):
    '''
    - esempio su come vanno inserite le posizioni iniziali dei droni:
            initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
    - esempio su come vanno inserite le traiettorie dei target:
            target_trajectories = [1, 2, 3, 5, 7]
            vedo che in questo caso inserisco solo l'indice del dataset delle traiettorie;

    - si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    - la visualizzazione grafica è inizialmente disattivata: dato che non c'è alcuna informazione stampata a console per  adesso conviene impostare:      graphics = "ON" ;
    - i droni sono rappresentati ognuno con colori differenti, la posizione reale dei target è rappresentata in rosso, la stima corretta delle posizioni dei target è rappresentata in verde (la stima viene eseguita su una misura  disturbata da un rumore gaussiano);

    -   il processo dura nel complesso 350 time step.

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde;
    -il control update è eseguito con un metodo di discesa del gradiente. Complessivamente i metodi utilizzabili sono 4. Si può modificarlo col parametro gradient_descent_method = "...":
            -   "gradient_descent" -> metodo del gradiente di discesa impostato automaticamente, si può variare solo il learning_rate;
            -   "gradient_descent2" -> simile a gradient_descent con l'aggiunta del decadimento del tasso di apprendimento ed un momentum;
            -   "rmsprop" -> Root Mean Square Propagation, metodo con tasso di apprendimento adattivo;
            -   "adam" -> Adaptive Moment Estimation, estensione di RMSProp che tiene conto della media mobile dei momenti del primo e del secondo ordine del gradiente;
    -KF: applicazione del filtro di kalman:
            -   "OFF" -> il valore passato ai droni è la posizione reale dei target;
            -   "NOISY" -> il valore dei target passato ai droni è il valore reale disturbato solo da un rumore;
            -   "PREDICT" -> ai droni viene passata la predizione della posizione dei target ottenuta attraverso il filtro di Kalman;
            -   "UPDATE" -> ai droni viene passata la predizione CORRETTA con l'aggiornamento della misura della posizione dei target, ottenuta attraverso il filtro di Kalman;
    '''

    ''' 
            In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                    H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                    H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                    H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                        trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                    H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
        '''

    H_plots = [[], [], [], []]

    n_drones = initial_condition_drones.shape[0]
    m_targets_trajectories = len(targets_trajectories)

    '''Definisco le matrici dei filtri'''
    # Intervallo di tempo
    dt = 1
    # Matrice di transizione dello stato per il movimento in 2D # x_k+1 = x_k + x'_k * dt
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # Matrice di osservazione (posizione soltanto)
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    # Covarianza del rumore di processo
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # Measurement noise covariance based on noise characteristics
    sigma = sigma_noise
    R = np.array([[sigma ** 2, 0], [0, sigma ** 2]])
    # da questa covarianza io vado a calcolarmi il rumore
    # esempio: misura_rumorosa = misura_pulita + np.random.normal(0, sigma, misura_pulita.shape))

    drone = []
    target = []
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    for i_target in range(m_targets_trajectories):
        target.append([dataset[targets_trajectories[i_target]][0][:, 0] * (1 / 10),
                       dataset[targets_trajectories[i_target]][0][:, 1] * (
                                   1 / 10)])  # il target viene definito per x e y

    # prima di calcolare effettivamente il filtro di kalman per ogni target, devo calcolare la P0
    # devo quindi definire variabili e costanti iniziali
    '''variables rappresenta l'insieme delle variabili da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]
    constant_theta = []
    constant_target = []
    constant_drone_position = []
    constant_parameter = []
    for i_drones in range(n_drones):
        constant_theta.append(drone[i_drones].theta)
        constant_drone_position.append(drone[i_drones].p[0])
        constant_drone_position.append(drone[i_drones].p[1])
        constant_drone_position.append(drone[i_drones].p[2])
    for i_target in range(m_targets_trajectories):
        constant_target.append(target[i_target][0][0])
        constant_target.append(target[i_target][1][0])
    # definisco gli ulteriori parametri
    constant_parameter.append(G_function_parameter)
    constant_parameter.append(F_function_parameter)
    constant_parameter.append(h_tilde_max)
    constant_parameter.append(alpha_parameter)
    constant_parameter.append(bounds_z[1])
    # creo la matrice di costanti complessiva
    constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]
    targets_covariances = targetCovariance(variables, constants)
    # successivamente mi calcolo attraverso una funzione la covarianza di ogni target

    '''INIZIALIZZAZIONE FILTRO DI KALMAN'''
    K_filter = []  # matrice in cui saranno contenuti le m classi dei filtri di Kalman
    for i_target in range(m_targets_trajectories):
        # Stato iniziale (posizione e velocità iniziali)
        x0 = np.array([target[i_target][0][0], target[i_target][1][0], 0, 0]).reshape((4, 1))
        # usando la funzione targetCovariance() abbiamo generato una lista di covarianze, una per target. Al j-esimo target viene assegnato
        sigma_p = targets_covariances[i_target]
        sigma_v = 100
        P0 = np.array([[sigma_p, 0, 0, 0], [0, sigma_p, 0, 0], [0, 0, sigma_v, 0], [0, 0, 0, sigma_v]])
        # in questa maniera inizializzo tanti filtri quanti i target
        K_filter.append(kalmanFilter.KalmanFilter2D(A, C, Q, R, x0, P0))

    '''definisco i vincoli del problema di ottimizzazione (i bounds)'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 200
        y0_window = 200
        delta_window = 10

    '''COSTRUISCO LE MATRICI IN CUI SALVERO' I RISULTATI DEL FILTRO DI KALMAN'''
    predicted_value = []

    for i in range(m_targets_trajectories):
        predicted_value.append([])

    updated_value = []

    for i in range(m_targets_trajectories):
        updated_value.append([])

    # INIZIO ITERAZIONE PER COVERAGE
    for i_time_step in range(target[1][0].shape[0]):
        '''FASE IN CUI VIENE AGGIORNATO IL FILTRO DI KALMAN (L'HO METTO QUI DATO CHE PER ADESSO LAVORA INDIPENDENTEMENTE DAL RESTO)'''
        constant_target = []
        for i_target in range(m_targets_trajectories):
            x_measure = target[i_target][0][i_time_step]
            y_measure = target[i_target][1][i_time_step]
            measure = np.array([x_measure, y_measure])
            noisy_measure = measure + np.random.normal(0, sigma, measure.shape)
            noisy_measure = noisy_measure.reshape([2, 1])
            if KF == "NOISY":
                constant_target.append(noisy_measure[0])
                constant_target.append(noisy_measure[1])

            # ESEGUO LA PREDIZIONE
            K_filter[i_target].predict()
            # fase in cui viene salvato il risultato della predizione

            x_predicted = K_filter[i_target].x[0]
            y_predicted = K_filter[i_target].x[1]
            if KF == "PREDICT":
                constant_target.append(x_predicted)
                constant_target.append(y_predicted)
            xy_predicted = [x_predicted, y_predicted]
            predicted_value[i_target].append(xy_predicted)

            # ESEGUO LA CORREZIONE
            K_filter[i_target].update(noisy_measure)
            # fase in cui viene salvato il valore della correzione
            x_updated = K_filter[i_target].x[0]
            y_updated = K_filter[i_target].x[1]
            if KF == "UPDATE":
                constant_target.append(x_updated)
                constant_target.append(y_updated)
            xy_updated = [x_updated, y_updated]
            updated_value[i_target].append(xy_updated)

            # TODO primo test: ogni 10 operazioni si usa la covarianza del problema

        '''CONSTANTS'''
        ''' definisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        if KF == "OFF":
            for i_target in range(m_targets_trajectories):
                constant_target.append(target[i_target][0][i_time_step])
                constant_target.append(target[i_target][1][i_time_step])

            # print(target[i_target][0][i_time_step], target[i_target][1][i_time_step])
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        '''DISCESA DEL GRADIENTE'''
        # viene eseguita la discesa del gradiente, si può scegliere tra più metodi
        if gradient_descent_method == "gradient_descent":
            result_gradient_descent = gradient_descent(funcH_2d, variables, constants, num_iterations=1,
                                                       learning_rate=learning_rate_parameter)
        if gradient_descent_method == "gradient_descent2":
            result_gradient_descent = gradient_descent2(funcH_2d, variables, constants, num_iterations=1,
                                                        learning_rate=learning_rate_parameter)
        if gradient_descent_method == "adam":
            result_gradient_descent = adam(funcH_2d, variables, constants, num_iterations=1,
                                           learning_rate=learning_rate_parameter)
        if gradient_descent_method == "rmsprop":
            result_gradient_descent = rmsprop(funcH_2d, variables, constants, num_iterations=1,
                                              learning_rate=learning_rate_parameter)

        '''RIDEFINIZIONE VARIABLES'''
        ''' annullo il vettore variables e ci inserisco i nuovi valori, ovvero quelli frutto dell'ottimizzazione'''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_gradient_descent.shape[0]):
            variables[i_drones_result] = result_gradient_descent[i_drones_result]

        '''COSTRUZIONE DEI PLOT'''
        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        '''AGGIORNAMENTO POSIZIONE DRONI'''
        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''
        for i_drones in range(n_drones):
            drone[i_drones].p[0] = result_gradient_descent[3 * i_drones]
            drone[i_drones].p[1] = result_gradient_descent[3 * i_drones + 1]
            drone[i_drones].p[2] = result_gradient_descent[3 * i_drones + 2]

        '''In questa zona metto il filtro di kalman'''

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            random.seed(10)
            # disegno i droni
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            # disegno i target
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target][0][i_time_step] - 10,
                                  y0_window + delta_window * target[i_target][1][i_time_step] - 10,
                                  20, 20))
                # pygame.display.update()
            # line_color = (255, 255, 255)
            # pygame.draw.line(win, line_color, (400, 0), (400, 800))
            # pygame.draw.line(win, line_color, (0, 400), (800, 400))
            # win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            # pygame.display.update()
            # time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede

            '''DISEGNO I TARGET STIMATI'''
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (0, 255, 0),
                                 (x0_window + delta_window * K_filter[i_target].x[0] - 5,
                                  y0_window + delta_window * K_filter[i_target].x[1] - 5,
                                  10, 10))

            # pygame.display.update()
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)

        # print("iterazione finita")
        # quando si chiude il ciclo viene chiuso lo schermo
    return H_plots
    pygame.display.quit()
    pygame.quit()


''' IMPLEMENTAZIONE FINALE DELL' ALGORITMO CON FILTRO DI KALMAN '''


def moving_target_coverage_gradient_KF_multipleUpdate(initial_condition_drones, targets_trajectories,
                                                      bounds_x=(-40, +40),
                                                      bounds_y=(-40, +40), bounds_z=(1, 50), graphics="OFF",
                                                      G_function_parameter=3,
                                                      F_function_parameter=1, h_tilde_max=1, alpha_parameter=0.0001,
                                                      learning_rate_parameter=0.1,
                                                      gradient_descent_method="gradient_descent", KF="OFF",
                                                      sigma_noise=1.5, sigma_w=0.1):
    '''
    - esempio su come vanno inserite le posizioni iniziali dei droni:
            initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
    - esempio su come vanno inserite le traiettorie dei target:
            target_trajectories = [1, 2, 3, 5, 7]
            vedo che in questo caso inserisco solo l'indice del dataset delle traiettorie;

    - si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    - la visualizzazione grafica è inizialmente disattivata: dato che non c'è alcuna informazione stampata a console per  adesso conviene impostare:      graphics = "ON" ;
    - i droni sono rappresentati ognuno con colori differenti, la posizione reale dei target è rappresentata in rosso, la stima corretta delle posizioni dei target è rappresentata in verde (la stima viene eseguita su una misura  disturbata da un rumore gaussiano);

    -   il processo dura nel complesso 350 time step.

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde;
    -il control update è eseguito con un metodo di discesa del gradiente. Complessivamente i metodi utilizzabili sono 4. Si può modificarlo col parametro gradient_descent_method = "...":
            -   "gradient_descent" -> metodo del gradiente di discesa impostato automaticamente, si può variare solo il learning_rate;
            -   "gradient_descent2" -> simile a gradient_descent con l'aggiunta del decadimento del tasso di apprendimento ed un momentum;
            -   "rmsprop" -> Root Mean Square Propagation, metodo con tasso di apprendimento adattivo;
            -   "adam" -> Adaptive Moment Estimation, estensione di RMSProp che tiene conto della media mobile dei momenti del primo e del secondo ordine del gradiente;
    -KF: applicazione del filtro di kalman:
            -   "OFF" -> il valore passato ai droni è la posizione reale dei target;
            -   "NOISY" -> il valore dei target passato ai droni è il valore reale disturbato solo da un rumore;
            -   "PREDICT" -> ai droni viene passata la predizione della posizione dei target ottenuta attraverso il filtro di Kalman;
            -   "UPDATE" -> ai droni viene passata la predizione CORRETTA con l'aggiornamento della misura della posizione dei target, ottenuta attraverso il filtro di Kalman;
    '''

    ''' 
            In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                    H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                    H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                    H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                        trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                    H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
        '''

    H_plots = [[], [], [], []]

    n_drones = initial_condition_drones.shape[0]
    m_targets_trajectories = len(targets_trajectories)

    '''Definisco le matrici dei filtri'''
    # Intervallo di tempo
    dt = 1
    # Matrice di transizione dello stato per il movimento in 2D # x_k+1 = x_k + x'_k * dt
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # Matrice di osservazione (posizione soltanto)
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    # Covarianza del rumore di processo
    Q = sigma_w * np.array([[(dt ^ 3) / 3, 0, (dt ^ 2) / 2, 0],
                            [0, (dt ^ 3) / 3, 0, (dt ^ 2) / 2],
                            [(dt ^ 2) / 2, 0, dt, 0],
                            [0, (dt ^ 2) / 2, 0, dt]])
    # Measurement noise covariance based on noise characteristics
    sigma = sigma_noise
    R = np.array([[sigma ** 2, 0], [0, sigma ** 2]])
    # da questa covarianza io vado a calcolarmi il rumore
    # esempio: misura_rumorosa = misura_pulita + np.random.normal(0, sigma, misura_pulita.shape))

    drone = []
    target = []
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    for i_target in range(m_targets_trajectories):
        target.append([dataset[targets_trajectories[i_target]][0][:, 0] * (1 / 10),
                       dataset[targets_trajectories[i_target]][0][:, 1] * (
                                   1 / 10)])  # il target viene definito per x e y

    # prima di calcolare effettivamente il filtro di kalman per ogni target, devo calcolare la P0
    # devo quindi definire variabili e costanti iniziali
    '''variables rappresenta l'insieme delle variabili da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]
    constant_theta = []
    constant_target = []
    constant_drone_position = []
    constant_parameter = []
    for i_drones in range(n_drones):
        constant_theta.append(drone[i_drones].theta)
        constant_drone_position.append(drone[i_drones].p[0])
        constant_drone_position.append(drone[i_drones].p[1])
        constant_drone_position.append(drone[i_drones].p[2])
    for i_target in range(m_targets_trajectories):
        constant_target.append(target[i_target][0][0] + np.random.normal(0, sigma))
        constant_target.append(target[i_target][1][0]+ np.random.normal(0, sigma))
    # definisco gli ulteriori parametri
    constant_parameter.append(G_function_parameter)
    constant_parameter.append(F_function_parameter)
    constant_parameter.append(h_tilde_max)
    constant_parameter.append(alpha_parameter)
    constant_parameter.append(bounds_z[1])
    # creo la matrice di costanti complessiva
    constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]
    targets_covariances = targetCovariance(variables, constants)
    # successivamente mi calcolo attraverso una funzione la covarianza di ogni target

    '''INIZIALIZZAZIONE FILTRO DI KALMAN'''
    K_filter = []  # matrice in cui saranno contenuti le m classi dei filtri di Kalman
    for i_target in range(m_targets_trajectories):
        # Stato iniziale (posizione e velocità iniziali)
        x0 = np.array([target[i_target][0][0], target[i_target][1][0], 0, 0]).reshape((4, 1))
        # usando la funzione targetCovariance() abbiamo generato una lista di covarianze, una per target. Al j-esimo target viene assegnato
        sigma_p = targets_covariances[i_target]
        sigma_v = 100
        P0 = np.array([[sigma_p, 0, 0, 0], [0, sigma_p, 0, 0], [0, 0, sigma_v, 0], [0, 0, 0, sigma_v]])
        # in questa maniera inizializzo tanti filtri quanti i target
        # l'inizializzazione la eseguo solo con la R e sigma base, l'update multiplo con multiple covarianze sull'errore di misura la eseguo più avanti
        K_filter.append(kalmanFilter.KalmanFilter2D(A, C, Q, R, x0, P0))

    '''definisco i vincoli del problema di ottimizzazione (i bounds)'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 200
        y0_window = 200
        delta_window = 10

    '''COSTRUISCO LE MATRICI IN CUI SALVERO' I RISULTATI DEL FILTRO DI KALMAN'''
    predicted_value = []

    for i in range(m_targets_trajectories):
        predicted_value.append([])

    updated_value = []

    for i in range(m_targets_trajectories):
        updated_value.append([])

    # INIZIO ITERAZIONE PER COVERAGE
    for i_time_step in range(target[1][0].shape[0]):
        ''' COSTRUZIONE DELLA MATRICE R E SIGMA_MATRIX '''
        constant_target = []
        R = []
        sigma_matrix = []
        for i_target in range(m_targets_trajectories):
            x_measure = target[i_target][0][i_time_step]
            y_measure = target[i_target][1][i_time_step]
            R_drone = []
            sigma_drone = []
            for i_drone in range(n_drones):
                x_d = variables[3 * i_drone]
                y_d = variables[3 * i_drone + 1]
                z_d = variables[3 * i_drone + 2]
                theta_d = constants[0][i_drone]
                radius = z_d * math.tan(theta_d/2)
                z_max = constants[3][4]
                R_ij, sigma_ij = R_function(x_d, y_d, z_d, z_max, x_measure, y_measure, sigma, G_function_parameter,
                                            F_function_parameter, radius, h_tilde_max, alpha_parameter)
                R_drone.append(R_ij)
                sigma_drone.append(sigma_ij)
            R.append(R_drone)
            sigma_matrix.append(sigma_drone)

        ''' COSTRUZIONE DELL'OSSERVAZIONE, PREDIZIONE E FILTRAGGIO '''
        for i_target in range(m_targets_trajectories):
            x_measure = target[i_target][0][i_time_step]
            y_measure = target[i_target][1][i_time_step]
            measure = np.array([x_measure, y_measure])
            # PREDIZIONE DEL FILTRO DI KALMAN
            K_filter[i_target].predict()
            # ESEGUO L'UPDATE SEQUENZIALE
            for i_drone in range(n_drones):
                sigma_covarianza = sigma_matrix[i_target][i_drone]
                R_covarianza = R[i_target][i_drone]
                print("Drone ", i_drone, ", Target ", i_target, ", R ", R_covarianza, ", sigma ", sigma_covarianza)
                #if sigma_matrix[i_target][i_drone] > 300:
                #    continue
                K_filter[i_target].R = R[i_target][i_drone]
                observation = measure + np.random.normal(0, sigma_matrix[i_target][i_drone], measure.shape)
                observation = observation.reshape([2, 1])
                print("Posizione vera target ", measure, ", misura rumorosa del target passata al drone", observation, ", Stima della posizione del target prima della correzione ", K_filter[i_target].x[0], K_filter[i_target].x[1])
                # ESEGUO LA CORREZIONE
                # questo if serve per evitare che una misura troppo rumorosa influenzi negativamente la stima. in pratica se nessun drone vede il target ci affidiamo solo alla predizione (è come se non avessimo alcuna osservazione)
                # solo se il drone vede il target e quindi è in grado di fornirci un'osservazione relativamente "poco rumorosa" viene effettuata la correzione
                if sigma_covarianza < 20:
                    K_filter[i_target].update(observation)
                    print("Stima della posizione del target dopo la correzione ", K_filter[i_target].x[0], K_filter[i_target].x[1])
            x_updated = K_filter[i_target].x[0]
            y_updated = K_filter[i_target].x[1]
            if KF == "UPDATE":
                constant_target.append(x_updated)
                constant_target.append(y_updated)
            xy_updated = [x_updated, y_updated]
            updated_value[i_target].append(xy_updated)

        '''
        for i_target in range(m_targets_trajectories):
            x_measure = target[i_target][0][i_time_step]
            y_measure = target[i_target][1][i_time_step]
            measure = np.array([x_measure, y_measure])
            noisy_measure = measure + np.random.normal(0, sigma, measure.shape)
            noisy_measure = noisy_measure.reshape([2, 1])
            if KF == "NOISY":
                constant_target.append(noisy_measure[0])
                constant_target.append(noisy_measure[1])

            # QUI INSERISCO LA COSTRUZIONE DELLA MATRICE DI COVARIANZA

        #TODO QUI INSERISCO UN'ALTRA ITERAZIONE FOR I_TARGET IN RANGE(M_TARGET)
            # ESEGUO LA PREDIZIONE
            K_filter[i_target].predict()
            # fase in cui viene salvato il risultato della predizione

            x_predicted = K_filter[i_target].x[0]
            y_predicted = K_filter[i_target].x[1]
            if KF == "PREDICT":
                constant_target.append(x_predicted)
                constant_target.append(y_predicted)
            xy_predicted = [x_predicted, y_predicted]
            predicted_value[i_target].append(xy_predicted)

            # ESEGUO LA CORREZIONE
            # todo QUI INSERISCO GLI UPDATE MULTIPLI
            K_filter[i_target].update(noisy_measure)
            # fase in cui viene salvato il valore della correzione
            x_updated = K_filter[i_target].x[0]
            y_updated = K_filter[i_target].x[1]
            if KF == "UPDATE":
                constant_target.append(x_updated)
                constant_target.append(y_updated)
            xy_updated = [x_updated, y_updated]
            updated_value[i_target].append(xy_updated)

            # TODO primo test: ogni 10 operazioni si usa la covarianza del problema

        '''

        '''CONSTANTS'''
        ''' definisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        if KF == "OFF":
            for i_target in range(m_targets_trajectories):
                constant_target.append(target[i_target][0][i_time_step])
                constant_target.append(target[i_target][1][i_time_step])

            # print(target[i_target][0][i_time_step], target[i_target][1][i_time_step])
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        '''DISCESA DEL GRADIENTE'''
        # viene eseguita la discesa del gradiente, si può scegliere tra più metodi
        if gradient_descent_method == "gradient_descent":
            result_gradient_descent = gradient_descent(funcH_2d, variables, constants, num_iterations=1,
                                                       learning_rate=learning_rate_parameter)
        if gradient_descent_method == "gradient_descent2":
            result_gradient_descent = gradient_descent2(funcH_2d, variables, constants, num_iterations=1,
                                                        learning_rate=learning_rate_parameter)
        if gradient_descent_method == "adam":
            result_gradient_descent = adam(funcH_2d, variables, constants, num_iterations=1,
                                           learning_rate=learning_rate_parameter)
        if gradient_descent_method == "rmsprop":
            result_gradient_descent = rmsprop(funcH_2d, variables, constants, num_iterations=1,
                                              learning_rate=learning_rate_parameter)

        '''RIDEFINIZIONE VARIABLES'''
        ''' annullo il vettore variables e ci inserisco i nuovi valori, ovvero quelli frutto dell'ottimizzazione'''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_gradient_descent.shape[0]):
            variables[i_drones_result] = result_gradient_descent[i_drones_result]

        '''COSTRUZIONE DEI PLOT'''
        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        '''AGGIORNAMENTO POSIZIONE DRONI'''
        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''
        for i_drones in range(n_drones):
            drone[i_drones].p[0] = result_gradient_descent[3 * i_drones]
            drone[i_drones].p[1] = result_gradient_descent[3 * i_drones + 1]
            drone[i_drones].p[2] = result_gradient_descent[3 * i_drones + 2]

        '''In questa zona metto il filtro di kalman'''

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            random.seed(10)
            # disegno i droni
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            # disegno i target
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target][0][i_time_step] - 10,
                                  y0_window + delta_window * target[i_target][1][i_time_step] - 10,
                                  20, 20))
                # pygame.display.update()
            # line_color = (255, 255, 255)
            # pygame.draw.line(win, line_color, (400, 0), (400, 800))
            # pygame.draw.line(win, line_color, (0, 400), (800, 400))
            # win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            # pygame.display.update()
            # time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede

            '''DISEGNO I TARGET STIMATI'''
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (0, 255, 0),
                                 (x0_window + delta_window * K_filter[i_target].x[0] - 5,
                                  y0_window + delta_window * K_filter[i_target].x[1] - 5,
                                  10, 10))

            # pygame.display.update()
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)

        # print("iterazione finita")
        # quando si chiude il ciclo viene chiuso lo schermo
    return H_plots
    pygame.display.quit()
    pygame.quit()




def moving_target_coverage_gradient_KF_multipleUpdate_prediction(initial_condition_drones, targets_trajectories,
                                                      bounds_x=(-40, +40),
                                                      bounds_y=(-40, +40), bounds_z=(1, 50), graphics="OFF",
                                                      G_function_parameter=3,
                                                      F_function_parameter=1, h_tilde_max=1, alpha_parameter=0.0001,
                                                      learning_rate_parameter=0.1,
                                                      gradient_descent_method="gradient_descent", KF="OFF",
                                                      sigma_noise=1.5, sigma_w=0.1, forward_prediction_steps=5):
    '''
    - esempio su come vanno inserite le posizioni iniziali dei droni:
            initial_condition_drones = np.array([[-4, -6, 40], [-4, 0, 40], [6, 10, 40], [16, 35, 40]]);
    - esempio su come vanno inserite le traiettorie dei target:
            target_trajectories = [1, 2, 3, 5, 7]
            vedo che in questo caso inserisco solo l'indice del dataset delle traiettorie;

    - si può settare i vincoli per l'ottimizzazione tramite i bounds: bounds_x, bounds_y, bounds_z: una volta settati questi vincoli sono uguali per tutti i droni;

    - la visualizzazione grafica è inizialmente disattivata: dato che non c'è alcuna informazione stampata a console per  adesso conviene impostare:      graphics = "ON" ;
    - i droni sono rappresentati ognuno con colori differenti, la posizione reale dei target è rappresentata in rosso, la stima corretta delle posizioni dei target è rappresentata in verde (la stima viene eseguita su una misura  disturbata da un rumore gaussiano);

    -   il processo dura nel complesso 350 time step.

    -L'informazione viene calcolata attraverso una funzione h = g * f, dove:
            - g = ([z_max - z]^G_function_parameter)/([z_max]^G_function_parameter)
            - f = ([r - |p_q|]^F_function_parameter)/([r]^F_function_parameter), dove p = posizione sul piano x,y del drone, e q = posizione sul piano x,y del target
        -   G_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene riducendo l'altezza del drone;
        -   F_function_parameter è un parametro che influenza il coverage -> tende a priorizzare l'informazione che si ottiene quando il centro della telecamera del drone si trova sovrapposto al target;

    -   h_tilde_max ->  limite superiore al valore che può assumere h_tilde;
    -   alpha ->  parametro per scalare il valore assunto da h_tilde;
    -il control update è eseguito con un metodo di discesa del gradiente. Complessivamente i metodi utilizzabili sono 4. Si può modificarlo col parametro gradient_descent_method = "...":
            -   "gradient_descent" -> metodo del gradiente di discesa impostato automaticamente, si può variare solo il learning_rate;
            -   "gradient_descent2" -> simile a gradient_descent con l'aggiunta del decadimento del tasso di apprendimento ed un momentum;
            -   "rmsprop" -> Root Mean Square Propagation, metodo con tasso di apprendimento adattivo;
            -   "adam" -> Adaptive Moment Estimation, estensione di RMSProp che tiene conto della media mobile dei momenti del primo e del secondo ordine del gradiente;
    -KF: applicazione del filtro di kalman:
            -   "OFF" -> il valore passato ai droni è la posizione reale dei target;
            -   "NOISY" -> il valore dei target passato ai droni è il valore reale disturbato solo da un rumore;
            -   "PREDICT" -> ai droni viene passata la predizione della posizione dei target ottenuta attraverso il filtro di Kalman;
            -   "UPDATE" -> ai droni viene passata la predizione CORRETTA con l'aggiornamento della misura della posizione dei target, ottenuta attraverso il filtro di Kalman;
    '''

    ''' 
            In H_plots verrano inseriti i valori delle covarianze complessive, rispettivamente:
                    H_plots[0] = H_real, ovvero la covarianza complessiva reale, senza tenere conto della h_tilde;
                    H_plots[1] = H_real_tilde, la covarianza complessica reale in cui si tiene conto anche della h_tilde;
                    H_plots[2] = H_opt, la covarianza complessiva ottima, ovvero quella che otterrei se i droni si 
                        trovassero nella configurazione ottima frutto dell'ottimizzazione, senza tener conto di h_tilde;
                    H_plots[3] = H_opt_tilde, la covarianza complessiva ottima (come la precedente), ma tendo conto di h_tilde.
        '''

    H_plots = [[], [], [], []]
    ''' n_drones = n° di droni presenti, m_targets_trajectories = n° di target presenti'''
    n_drones = initial_condition_drones.shape[0]
    m_targets_trajectories = len(targets_trajectories)

    '''Vengono definite le matrici dei filtri di Kalman: ogni drone avrà rispettivamente il suo filtro di Kalman'''
    # Intervallo di tempo
    dt = 1
    # Matrice di transizione dello stato per il movimento in 2D # x_k+1 = x_k + x'_k * dt
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # Matrice di osservazione (posizione soltanto)
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    # Covarianza del rumore di processo
    Q = sigma_w * np.array([[(dt ^ 3) / 3, 0, (dt ^ 2) / 2, 0],
                            [0, (dt ^ 3) / 3, 0, (dt ^ 2) / 2],
                            [(dt ^ 2) / 2, 0, dt, 0],
                            [0, (dt ^ 2) / 2, 0, dt]])
    # Covarianza del rumore di misura basata sulle caratteristiche del rumore; i valori della matrice
    # inizializzate tutte con lo stesso valore standard
    sigma = sigma_noise
    R = np.array([[sigma ** 2, 0], [0, sigma ** 2]])
    # da questa deviazione standard sigma vado a calcolarmi il rumore
    # esempio: misura_rumorosa = misura_pulita + np.random.normal(0, sigma, misura_pulita.shape))

    '''Generazione droni e target'''
    # Creo una lista vuota in cui aggiungo i droni (class) che genererò
    drone = []
    # Creo una lista vuota in cui aggiungo i dataset per le traiettorie dei target
    target = []
    # Generazione ricorsiva dei droni
    for i_drones in range(n_drones):
        drone.append(fD.Pursuerer(initial_condition_drones[i_drones], [0, 0, 0]))
    # Generazione ricorsiva dei target
    for i_target in range(m_targets_trajectories):
        target.append([dataset[targets_trajectories[i_target]][0][:, 0] * (1 / 10),
                       dataset[targets_trajectories[i_target]][0][:, 1] * (1 / 10)]) #il target viene definito per x e y

    # prima di calcolare effettivamente il filtro di kalman per ogni target, devo calcolare la P0
    # devo quindi definire variabili e costanti iniziali
    '''variables rappresenta l'insieme delle variabili da ottimizzare e, allo stesso tempo, le condizioni iniziali del problema'''
    # inizializzo la matrice variables, matrice di dimensione (3N x 1), dove N = n° droni
    variables = np.zeros(initial_condition_drones.shape[0] * 3)

    # assegno le condizioni iniziali della posizione dei droni all'interno della matrice variables
    for i_drones in range(n_drones):
        variables[3 * i_drones] = initial_condition_drones[i_drones][0]
        variables[3 * i_drones + 1] = initial_condition_drones[i_drones][1]
        variables[3 * i_drones + 2] = initial_condition_drones[i_drones][2]

    ''' constants è la matrice che passo alla funzione func_H2d ed è costituita di sottomatrici:
        - constant_theta -> contiene i valori di theta, ovvero l'angolo di visione della telecamera dei droni;
        - constant_target -> contiene le posizioni dei target (dato che stiamo lavorando con un dataset ad ogni istante di tempo dell'iterazione la posizione del target viene aggiornata: constant_target viene ridefinita ogni volta);
        - constant_drone_position -> contiene la posizione attuale dei droni (anche questa matrice viene ridefinita ad ogni iterazione);
        - constant_parameter -> contiene una serie di parametri utili all'ottimizzazione:
                - G_function_parameter;
                - F_function_parameter;
                - h_tilde_max;
                - alpha_parameter;
                - bounds_z[1] -> è il limite superiore in altezza dei droni.
        '''
    # inizializzo le matrici di costanti
    constant_theta = []
    constant_target = []
    constant_drone_position = []
    constant_parameter = []
    # inserisco i valori di theta e della posizione dei droni all'interno della rispettiva matrice
    for i_drones in range(n_drones):
        constant_theta.append(drone[i_drones].theta)
        constant_drone_position.append(drone[i_drones].p[0])
        constant_drone_position.append(drone[i_drones].p[1])
        constant_drone_position.append(drone[i_drones].p[2])
    # inserisco la posizione dei target all'interno della rispettiva matrice (alla posizione viene aggiunto un rumore)
    for i_target in range(m_targets_trajectories):
        constant_target.append(target[i_target][0][0] + np.random.normal(0, sigma))
        constant_target.append(target[i_target][1][0]+ np.random.normal(0, sigma))
    # definisco gli ulteriori parametri
    constant_parameter.append(G_function_parameter)
    constant_parameter.append(F_function_parameter)
    constant_parameter.append(h_tilde_max)
    constant_parameter.append(alpha_parameter)
    constant_parameter.append(bounds_z[1])
    # creo la matrice di costanti complessiva constants
    constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]
    ''' 
        All'istante iniziale, in base alla posizione relativa tra i droni ed i target ho una certa informazione su ogni
        target. la funzione targetCovariance(.,.) mi fornisce la covarianza (ovvero l'inverso dell'informazione) 
        sull'informazione della posizione dei target.
    '''
    targets_covariances = targetCovariance(variables, constants)

    '''INIZIALIZZAZIONE FILTRO DI KALMAN'''
    K_filter = []  # matrice in cui saranno contenuti le classi dei filtri di Kalman (dimensione = n° target)
    for i_target in range(m_targets_trajectories):
        # Stato iniziale -> x, y, v_x, v_y -> rispettivamente posizione x e y e velocità lungo x e y
        x0 = np.array([target[i_target][0][0], target[i_target][1][0], 0, 0]).reshape((4, 1))
        # sigma_p è la covarianza iniziale sulla posizione: utilizzo il valore calcolato attraverso la la funzione targetCovariance(.,.)
        sigma_p = targets_covariances[i_target]
        # sigma_v è la covarianza iniziale sulla velocità: utilizzo un valore standard
        sigma_v = 100
        # P0 è la matrice di covarianza iniziale della stima
        P0 = np.array([[sigma_p, 0, 0, 0], [0, sigma_p, 0, 0], [0, 0, sigma_v, 0], [0, 0, 0, sigma_v]])
        # Scrivere "K_filter.append(kalmanFilter.KalmanFilter2D(A, C, Q, R, x0, P0))" mi permette di avere tanti filtri quanti target.
        # L'inizializzazione dei filtro viene eseguita utilizzando come matrice di covarianza sul rumore di misura la R ed i sigma standard.
        # La correzione multipla dove utilizzo le covarianze sull'errore di misura di ogni drone la eseguo più avanti.
        K_filter.append(kalmanFilter.KalmanFilter2D(A, C, Q, R, x0, P0))

    ''' Definisco i vincoli del problema di ottimizzazione (i bounds). Successivamente converto la lista in una tupla.'''
    bounds = []
    for i_drones in range(n_drones):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_z)
    bounds = tuple(bounds)

    ''' Prima parte grafica'''
    if graphics == "ON":
        "Inizializzo la parte grafica"
        # pygame.init()
        pygame.display.init()
        win = pygame.display.set_mode((800, 800))
        # win = pygame.display.set_mode((800, 800), pygame.FULLSCREEN)
        pygame.display.set_caption("Minimizzazione posizione droni")
        win.fill((0, 0, 0))
        pygame.time.delay(50)

        time.sleep(0.5)

        # parametri per settare il centro del grafico:
        x0_window = 200
        y0_window = 200
        delta_window = 10

    '''COSTRUISCO LE MATRICI IN CUI SALVERO' I RISULTATI DEL FILTRO DI KALMAN'''
    predicted_value = []

    for i in range(m_targets_trajectories):
        predicted_value.append([])

    updated_value = []

    for i in range(m_targets_trajectories):
        updated_value.append([])

    # INIZIO ITERAZIONE PER COVERAGE
    for i_time_step in range(target[1][0].shape[0]):
        ''' COSTRUZIONE DELLA MATRICE R E SIGMA_MATRIX '''
        # La matrice constant_target, come già detto sopra, viene ricalcolata ad ogni iterazione
        constant_target = []
        # R è una matrice che contiene la covarianza di tutte le combinazioni drone-target
        R = []
        # sigma_matrix è una matrice che contiene la deviazione standard di tutte le combinazioni drone-target
        sigma_matrix = []
        ''' Questo ciclo FOR serve per costruirci le matrici R e sigma_matrix'''
        for i_target in range(m_targets_trajectories):
            x_measure = target[i_target][0][i_time_step]
            y_measure = target[i_target][1][i_time_step]
            R_drone = []
            sigma_drone = []
            for i_drone in range(n_drones):
                x_d = variables[3 * i_drone]
                y_d = variables[3 * i_drone + 1]
                z_d = variables[3 * i_drone + 2]
                theta_d = constants[0][i_drone]
                radius = z_d * math.tan(theta_d/2)
                z_max = constants[3][4]
                ''' La funzione R_function() ci serve per calcolare rispettivamente covarianza e deviazione standard
                    della combinazione drone i-esimo e target j-esimo.'''
                R_ij, sigma_ij = R_function(x_d, y_d, z_d, z_max, x_measure, y_measure, sigma, G_function_parameter,
                                            F_function_parameter, radius, h_tilde_max, alpha_parameter)
                R_drone.append(R_ij)
                sigma_drone.append(sigma_ij)
            R.append(R_drone)
            sigma_matrix.append(sigma_drone)

        ''' COSTRUZIONE DELL'OSSERVAZIONE, APPLICAZIONE FILTRO DI KALMAN (PREDIZIONE E CORREZIONE), PREDIZIONE IN AVANTI DI P PASSI '''
        ''' Il successivo ciclo FOR ci serve per l'applicazione del filtro di Kalman e relativa predizione di una misura della posizione rumorosa'''
        # p = n° di passi in avanti della predizione della posizione del target
        p = forward_prediction_steps
        # p_targets_predictions[i_target][i_preditions][coordinate] è una matrice che contiene TUTTE le predizioni
        # sulla posizione dei target, per tutti i passi in avanti della predizione (i_preditions sono i passi in avanti della predizione)
        p_targets_predictions = []
        for i_target in range(m_targets_trajectories):
            x_measure = target[i_target][0][i_time_step]
            y_measure = target[i_target][1][i_time_step]
            measure = np.array([x_measure, y_measure])
            '''PREDIZIONE DEL FILTRO DI KALMAN'''
            K_filter[i_target].predict()
            '''ESEGUO LA CORREZIONE SEQUENZIALE'''
            for i_drone in range(n_drones):
                sigma_i_drone_j_target = sigma_matrix[i_target][i_drone]
                R_i_drone_j_target = R[i_target][i_drone]
                #print("Drone ", i_drone, ", Target ", i_target, ", R ", R_i_drone_j_target, ", sigma ", sigma_i_drone_j_target)
                ''' per la correzione sequenziale aggiorno il valore della covarianza del rumore di misura del target relativa al drone i-esimo'''
                K_filter[i_target].R = R[i_target][i_drone]
                ''' observation è l'osservazione sul target del drone i-esimo, disturbata dal rumore di misura'''
                observation = measure + np.random.normal(0, sigma_matrix[i_target][i_drone], measure.shape)
                observation = observation.reshape([2, 1])
                #print("Posizione vera target ", measure, ", misura rumorosa del target passata al drone", observation, ", Stima della posizione del target prima della correzione ", K_filter[i_target].x[0], K_filter[i_target].x[1])
                '''ESEGUO LA CORREZIONE'''
                # questo if serve per evitare che una misura troppo rumorosa influenzi negativamente la stima. in pratica se nessun drone vede il target ci affidiamo solo alla predizione (è come se non avessimo alcuna osservazione)
                # solo se il drone vede il target e quindi è in grado di fornirci un'osservazione relativamente "poco rumorosa" viene effettuata la correzione
                if sigma_i_drone_j_target < 200:
                    K_filter[i_target].update(observation)
                    #print("Stima della posizione del target dopo la correzione ", K_filter[i_target].x[0], K_filter[i_target].x[1])
            ''' x_updated e y_updated sono i valori stimati della posizione dell' i_esimo target dopo che la stima 
                della posizione è stata corretta con tutte le osservazioni della posizione disponibili'''
            x_updated = K_filter[i_target].x[0]
            y_updated = K_filter[i_target].x[1]
            ''' aggiorno constant_target con la stima della posizione. L'algoritmo del gradiente di discesa riceverà in
                ingresso il valore stimato della posizione del target e non la posizione reale del target'''
            if KF == "UPDATE":
                constant_target.append(x_updated)
                constant_target.append(y_updated)
            # aggiorno la lista delle posizioni stimate dei target
            xy_updated = [x_updated, y_updated]
            updated_value[i_target].append(xy_updated)      # lista delle correzioni della posizione

            '''VIENE ESEGUITA LA PREDIZIONE IN AVANTI CON EVENTUALE AGGIORNAMENTO DELLA POSIZIONE PER LA RICERCA DELL'OTTIMO'''
            targets_predictions = []
            ''' Eseguo p predizioni in avanti per ogni target'''
            xy_prediction = K_filter[i_target].x
            for i_prediction in range(p):
                xy_prediction = K_filter[i_target].A @ xy_prediction
                targets_predictions.append(xy_prediction)
            p_targets_predictions.append(targets_predictions)
            ''' Se è impostata questa modalità fornisco in ingresso all'algoritmo di ottimizzazione il valore 
                della predizione in avanti di p passi'''
            if KF == "PREDICTION":
                constant_target.append(xy_prediction[0])
                constant_target.append(xy_prediction[1])

        '''CONSTANTS'''
        ''' ridefinisco la costante da inserire nell'algoritmo di ottimizzazione '''
        constant_theta = []
        constant_drone_position = []
        constant_parameter = []
        for i_drones in range(n_drones):
            constant_theta.append(drone[i_drones].theta)
            constant_drone_position.append(drone[i_drones].p[0])
            constant_drone_position.append(drone[i_drones].p[1])
            constant_drone_position.append(drone[i_drones].p[2])
        ''' Se è impostata questa modalità fornisco in ingresso all'algoritmo di ottimizzazione il valore reale del target'''
        if KF == "OFF":
            for i_target in range(m_targets_trajectories):
                constant_target.append(target[i_target][0][i_time_step])
                constant_target.append(target[i_target][1][i_time_step])

            # print(target[i_target][0][i_time_step], target[i_target][1][i_time_step])
        # definisco gli ulteriori parametri
        constant_parameter.append(G_function_parameter)
        constant_parameter.append(F_function_parameter)
        constant_parameter.append(h_tilde_max)
        constant_parameter.append(alpha_parameter)
        constant_parameter.append(bounds_z[1])
        # creo la matrice di costanti complessiva
        constants = [constant_theta, constant_target, constant_drone_position, constant_parameter]

        '''DISCESA DEL GRADIENTE'''
        # viene eseguita la discesa del gradiente, si può scegliere tra più metodi. Il metodo migliore risulta essere rmsprop
        if gradient_descent_method == "gradient_descent":
            result_gradient_descent = gradient_descent(funcH_2d, variables, constants, num_iterations=1,
                                                       learning_rate=learning_rate_parameter)
        if gradient_descent_method == "gradient_descent2":
            result_gradient_descent = gradient_descent2(funcH_2d, variables, constants, num_iterations=1,
                                                        learning_rate=learning_rate_parameter)
        if gradient_descent_method == "adam":
            result_gradient_descent = adam(funcH_2d, variables, constants, num_iterations=1,
                                           learning_rate=learning_rate_parameter)
        if gradient_descent_method == "rmsprop":
            result_gradient_descent = rmsprop(funcH_2d, variables, constants, num_iterations=1,
                                              learning_rate=learning_rate_parameter)

        '''RIDEFINIZIONE VARIABLES'''
        ''' Annullo il vettore variables e ci inserisco i nuovi valori, ovvero quelli frutto dell'ottimizzazione.
            In pratica, un passo del gradiente di discesa corrisponde alla nuova posizione del drone;
            successivamente i target si sposteranno e verrà eseguito un nuovo gradiente di discesa il cui risultato
            sarà la nuova posizione del drone'''
        variables = np.zeros(initial_condition_drones.shape[0] * 3)

        for i_drones_result in range(result_gradient_descent.shape[0]):
            variables[i_drones_result] = result_gradient_descent[i_drones_result]

        '''COSTRUZIONE DEI PLOT'''
        ''' vengono aggiunti i termini di covarianza complessiva alla matrice H '''
        H_plots[0].append(funcH_Testing(constants[2], constants))
        H_plots[1].append(funcH_2d(constants[2], constants))
        H_plots[2].append(funcH_Testing(variables, constants))
        H_plots[3].append(funcH_2d(variables, constants))

        '''AGGIORNAMENTO POSIZIONE DRONI'''
        ''' inizia la fase di aggiornamento in cui il drone aggiorna la sua posizione nella direzione desiderata dall'ottimizzazione '''
        for i_drones in range(n_drones):
            drone[i_drones].p[0] = result_gradient_descent[3 * i_drones]
            drone[i_drones].p[1] = result_gradient_descent[3 * i_drones + 1]
            drone[i_drones].p[2] = result_gradient_descent[3 * i_drones + 2]

        '''In questa zona metto il filtro di kalman'''

        ''' seconda parte grafica '''
        # la funzione aggiorna riempie lo schermo di nero, serve per eliminare la traccia delle posizioni precedenti
        if graphics == "ON":
            win.fill((0, 0, 0))
            #la funzione random.seed(10) serve per avere ad ogni iterazione la stessa combinazione di colori per i droni
            random.seed(10)
            '''DISEGNO I DRONI'''
            for i_drones in range(n_drones):
                pygame.draw.circle(win, color=(rg(0, 255), rg(0, 255), rg(0, 255)),
                                   center=((x0_window + delta_window * drone[i_drones].p[0]),
                                           (x0_window + delta_window * drone[i_drones].p[1])),
                                   radius=(drone[i_drones].visibilityCone() * 10))
                # pygame.display.update()

            '''DISEGNO I TARGET NELLE LORO POSIZIONI REALI (COLORE ROSSO)'''
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (255, 0, 0),
                                 (x0_window + delta_window * target[i_target][0][i_time_step] - 10,
                                  y0_window + delta_window * target[i_target][1][i_time_step] - 10,
                                  20, 20))
                # pygame.display.update()
            # line_color = (255, 255, 255)
            # pygame.draw.line(win, line_color, (400, 0), (400, 800))
            # pygame.draw.line(win, line_color, (0, 400), (800, 400))
            # win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            # pygame.display.update()
            # time.sleep(0.1)
            # la funzione precedente è funzionale a livello grafico, permette di avere il tempo di vedere graficamente cosa succede

            '''DISEGNO I TARGET NELLE LORO POSIZIONI STIMATE E CORRETTE (COLORE VERDE)'''
            for i_target in range(m_targets_trajectories):
                pygame.draw.rect(win, (0, 255, 0),
                                 (x0_window + delta_window * K_filter[i_target].x[0] - 5,
                                  y0_window + delta_window * K_filter[i_target].x[1] - 5,
                                  10, 10))

            '''DISEGNO LE PREDIZIONI DELLE POSIZIONI DEI TARGET (COLORE CHE TENDE DAL NERO AL BLU: PIù è BLU PIù SONO '''
            for i_target in range(m_targets_trajectories):
                for prediction in range(p):
                    pygame.draw.rect(win, (0, 0, 0 + 255*(prediction/p)),
                                     (x0_window + delta_window * p_targets_predictions[i_target][prediction][0] - 5,
                                      y0_window + delta_window * p_targets_predictions[i_target][prediction][1] - 5,
                                      10, 10))

            # pygame.display.update()
            '''Creo una linea centrale e raddrizzo l'immagine'''
            line_color = (255, 255, 255)
            pygame.draw.line(win, line_color, (400, 0), (400, 800))
            pygame.draw.line(win, line_color, (0, 400), (800, 400))
            win.blit(pygame.transform.flip(win, 0, 1), (0, 0))
            pygame.display.update()
            time.sleep(0.1)

        # print("iterazione finita")
        # quando si chiude il ciclo viene chiuso lo schermo
    return H_plots
    pygame.display.quit()
    pygame.quit()








def H_plots(H_real, H_desired):
    H_real.pop(0)
    H_desired.pop(0)
    plt.plot(H_real, color='green', label="H real: {}".format(np.mean(H_real)))
    plt.plot(H_desired, color='red', label="H desired: {}".format(np.mean(H_desired)))
    plt.legend()
    if min(H_real) <= min(H_desired):
        bottom = min(H_real) - 10
    else:
        bottom = min(H_desired) - 10
    if max(H_real) >= max(H_desired):
        upper = max(H_real) + 10
    else:
        upper = max(H_desired) + 10
    plt.axis((0, 300, bottom, upper))
    # plt.yscale('log')
    plt.show()

    '''
    # stesso grafico ma in scala logaritmica
    plt.plot(H_real, color='green', label="H real: {}".format(np.mean(H_real)))
    plt.plot(H_desired, color='red', label="H desired: {}".format(np.mean(H_desired)))
    plt.legend()
    plt.yscale('log', base=10)
    plt.show()'''
