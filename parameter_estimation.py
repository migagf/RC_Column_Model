import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from ph_model.Functions import *

from scipy.optimize import differential_evolution
import time


def get_residual(ModelParams, ThetaIn, MThetaOut):
    '''

    '''
    # Run the model
    Mtheta, EH, Theta = run_model(ModelParams, ThetaIn)
    residual = compute_error(Mtheta, MThetaOut)

    return residual


if __name__ == '__main__':
    
    #ModelParams = np.array([eta_1, Ko, My0, sig, lam, mu_p, 
    #                        sig_p, Rsmax, N, alpha, alpha_1, alpha_2, betam1])
    
    # Check files in normalized_hysteresis folder
    data = pd.read_csv('normalized_hysteresis/test_254.csv')
    ThetaIn = np.array(data['disp'])
    MThetaOut = np.array(data['force'])
    
    parameters = [1.0, 1.0, 1.0, 0.5, 0.5, 
                  1.0, 1.0, 0.5, 1.0, 0.01, 
                  2.0, 0.1, 0.01]
    bounds = [(0.5, 5.0), (0.95, 1.05), (0.95, 1.05), (0.02, 0.95), (0.02, 0.95), 
              (0.0, 5.0), (0.01, 10), (0.01, 1.0), (1.0, 10), (0.0, 0.1), 
              (0.5, 10), (0.0, 1.0), (0.0, 0.05)]
    
    M_hist, H_hist, theta_all = run_model(parameters, ThetaIn)
    plt.plot(theta_all, M_hist)
    plt.show()

    #start_time = time.time()
    #optimum = differential_evolution(get_residual, args=(ThetaIn, MThetaOut), bounds=bounds, maxiter=50, popsize=20, disp=True, workers=20)
    #end_time = time.time()
    #execution_time = end_time - start_time

    #print("Execution time:", execution_time, "seconds")
    #print(optimum.x)

    # Run the model with optimum parameters
    #MthetaOpt, EHOpt, ThetaOpt = run_model(optimum.x, ThetaIn)
    #residualOpt = compute_error(MthetaOpt, MThetaOut)

    # Print the residual with optimum parameters
    #print("Residual with optimum parameters:", residualOpt)

    # Plot the response
    #plt.plot(ThetaOpt, MthetaOpt)
    #plt.plot(ThetaIn, MThetaOut, 'k--')
    #plt.xlabel('Displacement')
    #plt.ylabel('Force')
    #plt.title('Response with Optimum Parameters')
    #plt.show()
    
    #if plots:
    #    #f2 = sp.interpolate.interp1d(xvec, MthetaObj) # interpolation of objective vector
    #    #MthetaObj_int = f2(xint)
    #    save_results(MthetaObj, 'MThetaOut.out')
    #else:
    #    save_results(MthetaOut_int)
    
    
    pass