from structural_model.elements import *
from structural_model.material_models import *
from structural_model.structure_model import *

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import os

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from structural_utilities import createModel

def run_analysis(model, gm_array, dt):

    # CRITICAL: Reset model to initial state before each analysis
    model.resetState()
    
    p_load = gm_array * 1.0
    
    # Pre-allocate arrays for better performance
    n_steps = len(p_load)
    u1_hist = np.zeros(n_steps)
    v1_hist = np.zeros(n_steps)
    a1_hist = np.zeros(n_steps)
    base_shear_hist = np.zeros(n_steps)
    
    solver = TH_Solver(dt=dt)
    start_time = time.time()

    # Precompute mass matrix multiplication factor
    mass_factor = model.mass @ np.concatenate(([1.0], np.zeros(model.mass.shape[0] - 1)))
    
    for i, p in enumerate(p_load):
        P = - p * mass_factor
        solver.step_increment(model, P)
        
        # Read state AFTER each step (don't cache cstate reference)
        u1_hist[i] = model.cstate["un"][0]
        v1_hist[i] = model.cstate["vn"][0]
        a1_hist[i] = model.cstate["an"][0]
        
        # Prefer model API if available, fallback to internal restoring force
        try:
            base_shear = float(model.get_base_shear())
        except Exception:
            try:
                base_shear = float(model.cstate.get('pr', [np.nan])[0])
            except Exception:
                base_shear = np.nan
        base_shear_hist[i] = base_shear

    end_time = time.time()
    
    # Calculate peak values using numpy for better performance
    peak_acc = np.max(np.abs(a1_hist))
    peak_vel = np.max(np.abs(v1_hist))
    peak_disp = np.max(np.abs(u1_hist))
    pga = np.max(np.abs(gm_array))

    return peak_acc, peak_vel, peak_disp, pga, base_shear_hist, u1_hist, end_time - start_time


def get_gm(selGM):
    # This function gets an unscaled ground motion (scaled only to represent scenario)

    # Open Ground Motion Data
    cd = os.getcwd()
    gmTable = pd.read_csv("gms//site0.csv")

    gmCode = str(selGM).zfill(2)

    g = 386

    sf = gmTable["factor"][selGM]

    print(gmTable["TH_file"][selGM])

    with open(cd+"//gms//"+gmTable["TH_file"][selGM]+".json") as file:
        gm_data = json.load(file)
    
    gmx = np.array(gm_data["data_x"]) * g * sf
    gmy = np.array(gm_data["data_y"]) * g * sf
    dt = float(gm_data["dT"])

    # print("dt =", dt)

    gmx = gmx[0:-1:1]
    gmy = gmy[0:-1:1]

    t = np.arange(0, dt * len(gmx), dt)

    return gmx, gmy, dt, t


def process_single_gm(selGM, mlk_models, sf):
    """Process a single ground motion for all models"""
    pierDisp_x = []
    pierAcc_x = []
    
    gm_x, gm_y, dt, t = get_gm(selGM)
    
    for row in mlk_models.itertuples():
        row_dict = row._asdict()
        model = createModel(row_dict)

        model.eig(show=False)

        model.resetState()
        peak_acc, peak_vel, peak_disp, pga, base_shear_hist_x, ux_hist, elapsed_time = run_analysis(model, sf*gm_x, dt)
        print(f"GM {selGM} X - Peak Acc: {peak_acc:.3f} in/s², Peak Vel: {peak_vel:.3f} in/s, Peak Disp: {peak_disp:.3f} in, PGA: {pga:.3f} in/s², Time: {elapsed_time:.2f} s")
        pierDisp_x.append(peak_disp)
        pierAcc_x.append(peak_acc)

    return selGM, pierDisp_x, pierAcc_x



if __name__ == "__main__":

    # Load model inventory
    model_inventory = pd.read_csv("red_line_models.csv")

    # Filter only MLK model (structure_id == as_06)
    mlk_models = model_inventory[model_inventory["structure_id"] == "as_06"]
    
    print(f"Total MLK models to simulate: {len(mlk_models)}")

    sf_list = [2.0]
    allGMs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for sf in sf_list:
        
        allPierDisp_x = {}
        allPierAcc_x = {}

        out_folder = f"surrogate_out\\sf_{str(sf)}"

        # Parallel processing of ground motions
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(process_single_gm, selGM, mlk_models, sf): selGM for selGM in allGMs}
            
            for future in as_completed(futures):
                selGM = futures[future]
                try:
                    gm_id, disp_x, acc_x = future.result()
                    allPierDisp_x[gm_id] = disp_x
                    allPierAcc_x[gm_id] = acc_x
                    print(f'Completed GM {gm_id}')
                except Exception as exc:
                    print(f'Ground motion {selGM} generated an exception: {exc}')
    
        # Save results to CSV (sort by GM index to maintain order)
        disp_x_df = pd.DataFrame([allPierDisp_x[gm] for gm in allGMs], index=allGMs)
        disp_x_df.to_csv(out_folder+"mlk_pier_disp_x.csv", index_label="GM_Index")
        acc_x_df = pd.DataFrame([allPierAcc_x[gm] for gm in allGMs], index=allGMs)
        acc_x_df.to_csv(out_folder+"mlk_pier_acc_x.csv", index_label="GM_Index")
        
        print(f'Finished processing all ground motions for sf {sf}')

    # Finished all simulations
    print("All simulations completed.")
