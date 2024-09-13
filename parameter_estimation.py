import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from ph_model.Functions import *

from scipy.optimize import differential_evolution
import time

# This code runs a parameter estimation using moment-rotation data...
# Uses a differential evolution algorithm to find the optimum parameters

def get_residual(ModelParams, ThetaIn, MThetaOut):
    '''
    Compute the residual between the model and the experimental data
    
    '''
    # Run the model
    Mtheta, EH, Theta = run_model(ModelParams, ThetaIn)
    # Now, need to run this with the rest of the column (add elastic portion)

    residual = compute_error(Mtheta, MThetaOut)

    return residual


if __name__ == '__main__':
    
    #ModelParams = np.array([eta_1, Ko, My0, sig, lam, mu_p, 
    #                        sig_p, Rsmax, N, alpha, alpha_1, alpha_2, betam1])
    
    # Check files in normalized_hysteresis folder
    data = pd.read_csv('normalized_hysteresis/test_267.csv')
    ThetaIn = np.array(data['disp'])
    MThetaOut = np.array(data['force'])
    
    plt.plot(ThetaIn, MThetaOut)
    plt.show()

    parameters = [1.0, 1.0, 1.0, 0.5, 0.5, 
                  1.0, 1.0, 0.5, 1.0, 0.01, 
                  2.0, 0.1, 0.01]
    
    bounds = [(0.5, 5.0), (0.95, 1.05), (0.95, 1.05), (0.02, 0.95), (0.02, 0.95), 
              (0.0, 5.0), (0.01, 10), (0.01, 1.0), (1.0, 10), (0.0, 0.1), 
              (0.5, 10), (0.01, 2.0), (0.0, 0.05)]
    
    #M_hist, H_hist, theta_all = run_model(parameters, ThetaIn)
    #plt.plot(theta_all, M_hist)
    #plt.show()

    # Run the optimization and time it
    start_time = time.time()
    # optimum = differential_evolution(get_residual, args=(ThetaIn, MThetaOut), bounds=bounds, maxiter=10, popsize=50, disp=True, workers=10, polish=False)
    end_time = time.time()

    execution_time = end_time - start_time

    print("Execution time:", execution_time, "seconds")
    #print(optimum.x)
    optimumx = [8.54062516e-01, 9.60280603e-01, 9.76523884e-01, 2.34825174e-01,
                2.14672046e-01, 2.06251329e+00, 1.48512456e+00, 9.01229695e-01,
                8.05707272e+00, 4.86674713e-03, 1.85995448e+00, 6.21563532e-01,
                4.07576188e-02]
    
    # Run the model with optimum parameters
    MthetaOpt, EHOpt, ThetaOpt = run_model(optimumx, ThetaIn)

    # Plot the response
    plt.plot(ThetaOpt, MthetaOpt)
    plt.plot(ThetaIn, MThetaOut, 'k--')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title('Response with Optimum Parameters')
    plt.show()
    
    #if plots:
    #    #f2 = sp.interpolate.interp1d(xvec, MthetaObj) # interpolation of objective vector
    #    #MthetaObj_int = f2(xint)
    #    save_results(MthetaObj, 'MThetaOut.out')
    #else:
    #    save_results(MthetaOut_int)
    
    
    pass