import openseespy.opensees as ops
import matplotlib.pyplot as plt
import opsvis as opsv  # for visualization
import numpy as np
import pandas as pd
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sections import *
from units import *
from utilities import *
from modelBuilder import *

# Add latex fonts for plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

g = 386.4

def process_ground_motion(selGM, outer_sf, gm_all):
    """Process a single ground motion analysis"""
    # -----
    # Process Ground Motion and plot
    # -----
    print(f'Processing Ground Motion {selGM} / {gm_all[-1]}')

    # Create output folder for selected GM
    out_folder = f'opensees_out\\gm{selGM}\\sf_{str(outer_sf)}'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Open Ground Motion Data
    cd = os.getcwd()
    gmTable = pd.read_csv("gms//site0.csv")
    gmCode = str(selGM).zfill(2)
    sf = gmTable["factor"][selGM] * outer_sf

    print(gmTable["TH_file"][selGM])

    with open(cd+"//gms//"+gmTable["TH_file"][selGM]+".json") as file:
        gm_data = json.load(file)
    
    gmx = np.array(gm_data["data_x"]) * g * sf
    gmy = np.array(gm_data["data_y"]) * g * sf
    dt = float(gm_data["dT"])
    gmx = gmx[0:-1:1]
    gmy = gmy[0:-1:1]
    t = np.arange(0, dt * len(gmx), dt)

    # Create subplots with 3:1 width ratio (3/4 left, 1/4 right)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3), dpi=200, gridspec_kw={'width_ratios': [3, 1]})

    # Left subplot: time histories
    axs[0].plot(t, gmx/g, 'r', linewidth=1.0, label='Direction X')
    axs[0].plot(t, gmy/g, 'k', linewidth=1.0, label='Direction Y')
    axs[0].grid()
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('$\\ddot{u}_g$ (g)')
    axs[0].set_xlim([0, 10 * np.round(max(t) / 10 + 1)])
    pga = np.round(np.max(np.abs([gmx, gmy])) / g, 2)
    ylim_val = np.ceil(10 * pga) / 10
    axs[0].set_ylim([-ylim_val, ylim_val])
    axs[0].set_title('Ground Motion ' + gmTable["TH_file"][selGM])
    axs[0].legend()

    # Right subplot: gmx vs gmy
    axs[1].plot(gmx/g, gmy/g, color='k', linewidth=1.0)
    axs[1].set_xlabel('$\ddot{u}_{gx}$ (g)')
    axs[1].set_ylabel('$\ddot{u}_{gy}$ (g)')
    axs[1].set_title('Direction X vs Direction Y')
    axs[1].grid()

    plt.tight_layout()
    # Save to the corresponding out_gmx folder
    plt.savefig(os.path.join(out_folder, f'GM_{gmCode}_plot.png'))
    plt.close()
    print('Ground motion plots saved')

    # -----
    # Create structural model and do modal analysis
    # ----- 

    # Load excel file with model geometry
    nodes_df = pd.read_excel("model_geometry.xlsx", sheet_name="nodeInfo")
    elements_df = pd.read_excel("model_geometry.xlsx", sheet_name="elements")

    # Generate Model
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)  # frame 3D

    getSections(plot=False)
    buildModel(nodes_df, elements_df)
    print('Structural model created')

    # Perform an eigenvalue analysis
    numEigen = 10
    eigenValues = ops.eigen(numEigen)
    # print("eigen values at start of transient:",eigenValues)
    T = 2 * np.pi / np.sqrt(np.array(eigenValues))
    print('Periods of identified modes:', T)

    # Uncomment to show mode shape
    opsv.plot_mode_shape(1, fig_wi_he=(200., 200.), sfac=-1000.0)
    # opsv.plot_model(node_labels=1, element_labels=1, local_axes=1, fig_wi_he=(300., 300.))
    ax = plt.gca()
    ax.view_init(elev=45, azim=45)  # tilt up/down, rotate around
    ax.grid(False)
    plt.savefig(os.path.join(out_folder, f'ModeShape_mode_gm{gmCode}.png'))
    #plt.show()
    print('Mode shape plot saved')

    # -----
    # Add Damping
    # -----
    zeta = 0.02
    omega1 = 2 * np.pi / T[0]
    omega2 = 2 * np.pi / T[1]

    alphaM = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
    betaK = 2 * zeta / (omega1 + omega2)

    # Add Rayleigh Damping
    ops.rayleigh(alphaM, 0.0, betaK, 0.0)
    print(f'Rayleigh Damping added: alphaM={alphaM}, betaK={betaK}')

    # Add self-weight load to nodes
    axialLoad = 320.0 * kip
    topColNodes = nodes_df[nodes_df["desc"] == "top of column"]["nodeID"].tolist()

    ops.timeSeries('Linear', 10)
    ops.pattern('Plain', 10, 10)
    for nodeID in topColNodes:
        ops.load(nodeID, 0.0, 0.0, -axialLoad, 0.0, 0.0, 0.0)

    ops.wipeAnalysis() # Wipe the eigenvalue analysis

    # Apply gravity loads
    ops.system("BandGeneral")
    ops.numberer('RCM')
    ops.constraints('Transformation')

    ops.test('NormDispIncr', 1.0e-8,  10)
    ops.algorithm("Newton")
    ops.integrator('LoadControl', 0.1)
    ops.analysis('Static')

    print('Applying gravity loads...')
    ok = ops.analyze(10)
    if ok != 0:
        print("Gravity Load Analysis Failed to Converge")
    else:
        print("Gravity Load Analysis Converged Successfully")
    
    # fix gravity loads for dynamic analysis
    ops.loadConst('-time', 0.0)
    print('Gravity loads fixed for dynamic analysis.')

    ops.wipeAnalysis()  # Wipe the gravity load analysis

    # -----
    # Create recorders
    # -----

    # Recorders for node displacements (top)
    nodeTags = []
    for nodeID in nodes_df[nodes_df["desc"] == "top of beam"]["nodeID"]:
        nodeTags.append(nodeID)

    ops.recorder('EnvelopeNode', '-file', os.path.join(out_folder, 'node_disp_x.out'), '-node', *nodeTags, '-dof', 1, 'disp')
    ops.recorder('EnvelopeNode', '-file', os.path.join(out_folder, 'node_disp_y.out'), '-node', *nodeTags, '-dof', 2, 'disp')
    
    # Recorders for accelerations
    ops.recorder('EnvelopeNode', '-file', os.path.join(out_folder, 'node_acc_x.out'), '-node', *nodeTags, '-dof', 1, 'accel')
    ops.recorder('EnvelopeNode', '-file', os.path.join(out_folder, 'node_acc_y.out'), '-node', *nodeTags, '-dof', 2, 'accel')

    # nodeTags = []
    # for nodeID in nodes_df[nodes_df["desc"] == "base"]["nodeID"]:
    #    nodeTags.append(nodeID)

    # ops.recorder('Node', '-file', os.path.join(out_folder, 'base_disp_x_'+'.out'), '-node', *nodeTags, '-dof', 1, 'disp')
    # ops.recorder('Node', '-file', os.path.join(out_folder, 'base_disp_y_'+'.out'), '-node', *nodeTags, '-dof', 2, 'disp')
    
    print('Node displacement recorders created')

    # Recorders for element forces
    eleTags = []
    for eleID in elements_df[elements_df["type"] == "pierCol"]["eleTag"]:
        eleTags.append(eleID)

    # Create one recorder per element
    for eleTag in eleTags:
        ops.recorder('EnvelopeElement', '-file', os.path.join(out_folder, f'ele_forces_{eleTag}.out'), '-ele', eleTag, 'force')

    print('Element force recorders created')

    # -----
    # Run dynamic analysis
    # -----

    # timeSeries('Path', tag, '-dt', dt=0.0, '-values', *values, '-time', *time, '-filePath', filePath='', '-fileTime', fileTime='', '-factor', factor=1.0, '-startTime', startTime=0.0, '-useLast', '-prependZero')
    ops.timeSeries('Path', 1, '-dt', dt, '-values', *gmx)
    ops.timeSeries('Path', 2, '-dt', dt, '-values', *gmy)

    # pattern('UniformExcitation', patternTag, dir, '-disp', dispSeriesTag, '-vel', velSeriesTag, '-accel', accelSeriesTag, '-vel0', vel0, '-fact', fact)
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)
    ops.pattern('UniformExcitation', 2, 2, '-accel', 2)

    ops.system("BandGeneral")
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-10,  10, 0)
    ops.algorithm("Newton")
    ops.numberer('RCM')
    ops.integrator('Newmark',  0.5,  0.25)

    ops.analysis('Transient')

    # set some variables for dynamic analysis
    tFinal = len(gmx)*dt
    tCurrent = ops.getTime()
    ok = 0

    # time = [tCurrent]
    # u_top = np.zeros((len(gmx), 65))  # array to store response values (one column per pier)

    # Perform the transient analysis
    step = 0

    # u1 = [0.0]

    print('Starting dynamic analysis...')
    while ok == 0 and tCurrent < tFinal:
        step += 1
        ok = ops.analyze(1, dt)
        
        # if the analysis fails try initial tangent iteration
        if ok != 0:
            print("regular newton failed .. lets try an initial stiffness for this step")
            ops.test('NormDispIncr', 1.0e-10,  100, 1)
            ops.algorithm('ModifiedNewton', '-initial')
            ok = ops.analyze(1, dt)
            
            if ok == 0:
                print("that worked .. back to regular newton")

                ops.test('NormDispIncr', 1.0e-10,  10, 0)
                ops.algorithm('Newton')
        
        tCurrent = ops.getTime()
        
        # u1.append(ops.nodeDisp(110, 1))  # just to make sure we have the last step recorded
        # time.append(tCurrent)

    print('Dynamic analysis completed.')
    if ok != 0:
        print("Dynamic Analysis Failed to Converge")
    else:
        print("Dynamic Analysis Converged Successfully")
    
    print(f'Ground Motion {selGM} processing completed.\n')
    return selGM


if __name__ == "__main__":
    
    sf_vec = [0.5, 1.0, 2.0, 3.0]
    # Define scale factor for ground motions

    for outer_sf in sf_vec:
        print(f'Processing sf {outer_sf}')

        # List of ground motion codes
        gm_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # list of ground motion indices to process

        # Parallel processing of ground motions
        with ProcessPoolExecutor(max_workers=10) as executor:
            # Submit all ground motions for processing
            futures = {executor.submit(process_ground_motion, selGM, outer_sf, gm_all): selGM for selGM in gm_all}
            
            # Wait for all to complete
            for future in as_completed(futures):
                selGM = futures[future]
                try:
                    result = future.result()
                    print(f'Completed GM {result}')
                except Exception as exc:
                    print(f'Ground motion {selGM} generated an exception: {exc}')

        print(f'Finished processing ground motions for sf {outer_sf}')

    print('All ground motions processed.')
