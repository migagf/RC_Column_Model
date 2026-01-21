import numpy as np
import matplotlib.pyplot as plt

from column_model.material_models import *

# Use latex interface for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


if __name__ == "__main__":

    # Example BW parameters
    eta1 = 1.0
    k0 = 1.0
    sy0 = 1.0
    sig = 0.1
    lam = 0.1

    n = 1.0
    alpha = 0.01
    alpha1 = 1.0
    alpha2 = 0.1
    betam1 = 0.0

    mup = [1.0, 1.0, 3.0, 1.0]
    sigp = [1.0, 1.0, 1.0, 2.0]
    rsmax = [0.6, 0.9, 0.9, 0.9]

    # Create list of BW parameters
    params_1 = [eta1, k0, sy0, sig, lam, mup[0], sigp[0], rsmax[0], n, alpha, alpha1, alpha2, betam1]
    params_2 = [eta1, k0, sy0, sig, lam, mup[1], sigp[1], rsmax[1], n, alpha, alpha1, alpha2, betam1]
    params_3 = [eta1, k0, sy0, sig, lam, mup[2], sigp[2], rsmax[2], n, alpha, alpha1, alpha2, betam1]
    params_4 = [eta1, k0, sy0, sig, lam, mup[3], sigp[3], rsmax[3], n, alpha, alpha1, alpha2, betam1]

    # Create Plastic Hinge
    ph_1 = deg_bw_material_original(params_1)
    ph_2 = deg_bw_material_mod(params_2)
    ph_3 = deg_bw_material_mod(params_3)
    ph_4 = deg_bw_material_mod(params_4)

    # Define some strain history
    maxStrains = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    strains = createCyclicStrain(maxStrains=maxStrains, dStrain=0.0001)

    # Get stress response
    strain_ph1, stress_ph1 = material_test(ph_1, strains[0])
    strain_ph2, stress_ph2 = material_test(ph_2, strains[0])
    strain_ph3, stress_ph3 = material_test(ph_3, strains[0])
    strain_ph4, stress_ph4 = material_test(ph_4, strains[0])

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    axs[0, 0].plot(strain_ph1, stress_ph1, color='k')
    axs[0, 0].set_title('(a) Original BW \n ($R_s=0.6$)')
    axs[0, 0].set_xlabel('$\\theta/\\theta_y$')
    axs[0, 0].set_ylabel('$M/M_y$')
    axs[0, 0].grid(True)

    axs[0, 1].plot(strain_ph2, stress_ph2, color='k')
    axs[0, 1].set_title('(b) Proposed BW \n ($R_s=0.9$, $\mu_p=1.0$, $\sigma_p=1.0$)')
    axs[0, 1].set_xlabel('$\\theta/\\theta_y$')
    axs[0, 1].set_ylabel('$M/M_y$')
    axs[0, 1].grid(True)

    axs[1, 0].plot(strain_ph3, stress_ph3, color='k')
    axs[1, 0].set_title('(c) Proposed BW \n ($R_s=0.9$, $\mu_p=3.0$, $\sigma_p=1.0$)')
    axs[1, 0].set_xlabel('$\\theta/\\theta_y$')
    axs[1, 0].set_ylabel('$M/M_y$')
    axs[1, 0].grid(True)

    axs[1, 1].plot(strain_ph4, stress_ph4, color='k')
    axs[1, 1].set_title('(d) Proposed BW \n ($R_s=0.9$, $\mu_p=1.0$, $\sigma_p=2.0$)')
    axs[1, 1].set_xlabel('$\\theta/\\theta_y$')
    axs[1, 1].set_ylabel('$M/M_y$')
    axs[1, 1].grid(True)

    # Set axis limits for all subplots
    for ax in axs.flat:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig('material_model_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.show()
