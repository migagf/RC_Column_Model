import numpy as np
import csv
from numpy import pi
import pandas as pd
import scipy as sp


def load_file(filename):
    out = []
    with open(filename, 'r') as file:
        data = csv.reader(file)
        for line in data:
            if len(line) == 1:
                out.append(float(line[0]))
            else:
                for col in line:
                    out.append(float(col))
    return np.array(out)


def save_results(response, filename='results.out'):
    # save the results
    with open(filename, 'w') as f:
        for i in range(0, len(response)):
            # value = str(MTheta[0][i])
            value = "{:e}".format(response[i])
            if i < len(response) - 1:
                f.write(''.join([value, '\n']))
            else:
                f.write(''.join([value]))


def compute_error(model, data):
    #print(len(data))
    err = np.sum(np.abs(model - data)) / len(data)
    # print(err)

    return err


def run_model(modelparams, theta_all, interpolate=False):
    # Interpolate theta_all using 10_000 points
    
    if interpolate:
        xvec = np.arange(0, len(theta_all))
        f = sp.interpolate.interp1d(xvec, theta_all)
        xint = np.linspace(0, xvec[-1], 10000)
        theta_all = f(xint)

    # First, load all the parameters from modelparams
    eta_1 = modelparams[0]
    eta_2 = 1 - eta_1
    Ko = modelparams[1]
    My0 = modelparams[2]

    # Pinching Parameters
    sig = modelparams[3]
    lam = modelparams[4]
    mu_p = modelparams[5]
    sig_p = modelparams[6]
    Rsmax = modelparams[7]

    N = np.round(modelparams[8], 0)
    alpha = modelparams[9]

    alpha_1 = modelparams[10]
    alpha_2 = modelparams[11]
    betam1 = modelparams[12]

    ## Run Model
    M_el = 0
    M_st = 0
    My = 1.0
    Msig = sig * My
    Mbar = lam * My

    M_hist = []
    H_hist = []

    M_hist.append(M_el)
    Kel = alpha * Ko

    Rk = 1.0
    Rk_min = Rk
    H = 0
    H_hist.append(H)

    thetam_pos = 0
    thetam_neg = 0
    maxtheta = 0
    s = 0.001
    Rs = 0

    thetamax_pos = [0, 0]

    thetamax_neg = [0, 0]
    momenmax_pos = [0, 0]
    momenmax_neg = [0, 0]

    # Start in Regime 0 (not unloading-reloading branch)
    regime_vec = []
    regime = 0

    regime_vec.append(regime)

    for i in range(1, len(theta_all)):
        theta_i = theta_all[i - 1]
        theta_ip1 = theta_all[i]

        dtheta = theta_ip1 - theta_i

        # Define regime (unload-reload or not)
        if M_hist[-1] * dtheta < 0 and regime == 0:  # if changing from loading to unloading
            regime = 1

            # if unloading happens, store the moment and rotation at the previous point (pivot)
            if M_hist[-1] > 0:
                thetamax_pos = [theta_i, theta_i]
                momenmax_pos = [M_hist[-1], M_hist[-1]]
            else:
                thetamax_neg = [theta_i, theta_i]
                momenmax_neg = [M_hist[-1], M_hist[-1]]

        if regime == 1 and M_hist[-2] > 0 and M_hist[-1] > 0 and dtheta > 0 and theta_ip1 > 0:
            # reloading, store the values
            regime = 2
            thetamax_pos[0] = theta_i
            momenmax_pos[0] = M_hist[-1]
            if thetamax_pos[1] - thetamax_pos[0] != 0:
                K2 = (momenmax_pos[1] - momenmax_pos[0]) / (thetamax_pos[1] - thetamax_pos[0])
            else:
                K2 = Ko

        elif regime == 1 and M_hist[-1] < 0 and dtheta < 0 and theta_ip1 < 0:
            regime = 2
            thetamax_neg[0] = theta_i
            momenmax_neg[0] = M_hist[-1]
            if thetamax_neg[1] - thetamax_neg[0] != 0:
                K2 = (momenmax_neg[1] - momenmax_neg[0]) / (thetamax_neg[1] - thetamax_neg[0])
            else:
                K2 = Ko

        if regime == 0 or regime == 1:
            z_ini = 1.0
            z_trial = 0.001

            counter = 0

            while abs(z_ini - z_trial) > 1.0e-8 and counter < 100:
                z_ini = z_trial

                # Function:
                A = 1 - (eta_1 * np.sign(z_trial * dtheta) + eta_2) * np.abs(z_trial / My) ** N
                Kh = (Rk - alpha) * Ko * A

                s = np.max([Rs * (thetam_pos - thetam_neg), 0.001])

                B = np.exp(-0.5 * ((z_trial - Mbar * np.sign(dtheta)) / (Msig)) ** 2)
                if B < 10 ** -20:
                    B = 10 ** -20

                Ks = ((1 / np.sqrt(2 * np.pi)) * (s / Msig) * B) ** (-1)

                Kr = (Kh * Ks) / (Kh + Ks)
                f = Kr * dtheta - z_trial + M_st

                # And it's derivative:
                if np.isnan(z_trial ** (N - 1)):
                    A_z = 0
                else:
                    A_z = - (eta_1 * np.sign(z_trial * dtheta) + eta_2) * N * z_trial ** (N - 1) * (
                                (np.sign(z_trial) / np.abs(My)) ** N)

                B_z = - (z_trial - Mbar * np.sign(dtheta)) * B / (Msig ** 2)
                Ks_z = - (np.sqrt(2 * pi) * Msig / s) * (B ** -2) * B_z
                Kh_z = (Rk - alpha) * Ko * A_z
                Kr_z = (Kh_z * Ks ** 2 + Ks_z * Kh ** 2) / (Kh + Ks) ** 2

                f_z = -1.0 + Kr_z * dtheta

                z_trial = z_trial - f / f_z
                counter += 1

            # Get tangent stiffness with the converged z value
            # Kh = (Rk - alpha) * Ko * (1 - (eta_1 * np.sign(z_trial * dtheta) + eta_2) * np.abs(z_trial / My) ** N)

            # Compute tangent stiffness

            K = Kel + (Kh * Ks) / (Kh + Ks)  # This one is the total stiffness of the parallel/series spring system

        else:  # if regime == 2
            K = K2

        M_cur = K * dtheta + M_hist[-1]

        # How to go back to regime 0?
        if regime == 2:
            if M_cur > 0 and dtheta > 0 and theta_ip1 > thetamax_pos[1]:  # if on Q1 and moment exceed pivot, then go back to regime 0
                regime = 0

            elif M_cur > 0 and dtheta < 0 and theta_ip1 < thetamax_pos[0]:
                regime = 0
                momenmax_pos[0] = M_hist[-1]

            elif M_cur < 0 and dtheta < 0 and theta_ip1 < thetamax_neg[1]:
                regime = 0

            elif M_cur < 0 and dtheta > 0 and theta_ip1 > thetamax_neg[0]:
                regime = 0
                momenmax_neg[0] = M_hist[-1]

        if M_cur * theta_ip1 < 0:
            regime = 0

        regime_vec.append(regime)
        M_hist.append(M_cur)
        M_el += Kel * dtheta
        M_st = M_cur - M_el

        # Stiffness Degradation
        Rk_trial = (np.abs(M_cur) + alpha_1 * My) / (Ko * np.abs(theta_i) + alpha_1 * My)
        Rk_min = min([Rk_trial, Rk_min])
        Rk = Rk_trial + (1 - alpha_2) * (Rk_min - Rk_trial)

        # Strength Degradation
        dH = M_cur * dtheta * (1 - (Kel + Rk * Kh) / (Rk * Ko))

        H += dH
        H_hist.append(H)
        My = My0 / (1 + betam1 * H)

        thetam_pos = max([thetam_pos, theta_i])
        thetam_neg = min([thetam_neg, theta_i])

        Msig = sig * My
        Mbar = lam * My

        if np.abs(theta_i) - maxtheta > 0:
            dRs = (1 / (np.sqrt(2 * pi) * sig_p)) * np.exp(-0.5 * ((maxtheta - mu_p) / (sig_p)) ** 2) * (
                        np.abs(theta_i) - maxtheta)
            Rs += dRs * Rsmax

            maxtheta = np.abs(theta_i)

    return M_hist, H_hist, theta_all