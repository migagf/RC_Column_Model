# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from ModelParameters import *
from Functions import *
import json

plots = False

try:
    f = open('scInput.json')
    data = json.load(f)
    npts = data["EDP"][0]['length']
    
except Exception as err:
    print('could not find json input file... trying out file', err)
    MthetaObj = load_file(r'MThetaOut.csv')
    npts = len(MthetaObj)
    
if plots:
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    MthetaObj = load_file(r'MThetaOut.csv')
    npts = len(MthetaObj)
    
    # interpolate MTheta
    #xvec = np.arange(0, len(MthetaObj))
    #f = sp.interpolate.interp1d(xvec, MthetaObj)
    #xint = np.linspace(0, xvec[-1], 10000)
    #MthetaObj = f(xint)
    

eta_2 = 1 - eta_1

Ko = 1.0
My0 = 1.0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    ModelParams = np.array([eta_1, Ko, My0, sig, lam, mu_p, 
                            sig_p, Rsmax, N, alpha, alpha_1, alpha_2, betam1])
    
    # Load curve
    ThetaIn = load_file(r'ThetaIn.csv')
    Mtheta, EH, Theta = run_model(ModelParams, ThetaIn)  # Theta and Mtheta have 10_000 values each
    
    xvec = np.linspace(0, 1, len(Mtheta))   # long vector
    
    f0 = sp.interpolate.interp1d(xvec, Theta)  # interpolation of theta vector
    f1 = sp.interpolate.interp1d(xvec, Mtheta) # interpolation of response vector
    
    xint = np.linspace(0, 1, npts)  # short vector
    
    ThetaIn_int = f0(xint)
    MthetaOut_int = f1(xint)
    
    if plots:
        #f2 = sp.interpolate.interp1d(xvec, MthetaObj) # interpolation of objective vector
        #MthetaObj_int = f2(xint)
        save_results(MthetaObj, 'MThetaOut.out')
    else:
        save_results(MthetaOut_int)
    
    if plots:
        plt.figure(dpi = 1200)
        plt.plot(ThetaIn_int, MthetaOut_int, 'r')
        plt.plot(ThetaIn_int, MthetaObj, 'k--')
        plt.grid()
        plt.show()
    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
