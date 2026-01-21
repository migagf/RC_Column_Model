"""
Structural Analysis Utilities for BART Regional Earthquake Simulations

This module contains utility functions for structural modeling, spatial analysis,
and system resource management that support the regional simulation workflow.

Functions are organized into categories:
- Structural Properties: Mass calculations
- Model Creation: Building structural models from inventory
- Spatial Analysis: Distance calculations, nearest neighbors
- System Resources: Memory monitoring, configuration optimization
- Cache Management: Model caching utilities
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import psutil

# Import structural model classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structural_model.structure_model import InelasticFrame
from structural_model.material_models import DegradingBoucWen_Fix


# ============================================================================
# Structural Properties
# ============================================================================

@lru_cache(maxsize=128)
def get_mass(height, girder_length=70*12):
    """
    Calculate pier mass properties for dynamic analysis
    
    Parameters
    ----------
    height : float
        Column height in inches
    girder_length : float
        Girder span length in inches (default: 70 ft = 840 in)
    
    Returns
    -------
    list
        [lateral_mass, small_rot_mass, large_rot_mass] in kips/(in/s²)
        
    Notes
    -----
    Mass calculation includes:
    - 2 girders (box sections)
    - 1 rigid cap beam
    - Column(s)
    
    Assumes:
    - Concrete unit weight: 150 pcf
    - Girder: 5'x4' box with 8" wall thickness
    - Cap beam: 5" x 24' x 6' (height x width x thickness)
    """
    # Get mass
    conc_gamma = 150 / (12 ** 3 * 1000)  # kips/(in3) (kips per cubic inch)

    # Calculations for one girder
    girder_t, girder_w, girder_h = 8, 12*5, 12*4  # (in)
    girder_a = girder_w * girder_h - (girder_w - 2 * girder_t) * (girder_h - 2 * girder_t)      # section area of girder (in2)
    girder_m = girder_a * girder_length * conc_gamma                                            # kips (weight of one girder)
    r_girder = np.sqrt(8 ** 2 + (5 / 2) ** 2) * 12                                              # in (distance from girder cm to girder cm)

    ix_girder = (1/12 * (girder_w ** 2 + girder_h ** 2)
                 - 1/12 * ((girder_w - 2 * girder_t) ** 2
                 + (girder_h - 2 * girder_t) ** 2) + r_girder ** 2) * girder_m

    # Calculations for rigid beam under the girders
    beam_h = 5 * 12       # inch
    beam_w = 3 * 8 * 12   # inch
    beam_t = 6 * 12       # inch

    beam_a = beam_h * beam_w  # area (in2)
    beam_m = beam_a * beam_t * conc_gamma # kips

    ix_beam = 1 / 12 * (beam_h ** 2 + beam_w ** 2) * beam_m

    # Calculations for the column
    col_d = beam_t
    col_a = np.pi * (col_d / 2) ** 2  # in2

    col_m = col_a * height * conc_gamma

    # Values for the entire pier (2 girders, beam, column)
    lat_mass = beam_m + 2 * girder_m + col_m
    rot_mass = ix_beam + 2 * ix_girder

    mass = [lat_mass/386, 0.1*rot_mass/386, rot_mass/386]  # kips/(in/s^2)
    
    return mass


# ============================================================================
# Model Creation
# ============================================================================

def createModel(pier_data):
    """
    Create structural model from pier inventory data
    
    Parameters
    ----------
    pier_data : dict or pd.Series
        Pier data containing:
        - NumberOfColumns: int
        - ColumnData: list of column dictionaries with parameters
        - GirderLength: float (optional, in feet)
    
    Returns
    -------
    InelasticFrame
        Configured structural model ready for time-history analysis
        
    Notes
    -----
    The model uses:
    - Bouc-Wen hysteretic hinges at column ends
    - P-delta geometry
    - 2% Rayleigh damping
    - Imperial units (kips, inches)
    """
    n_columns = pier_data['NumberOfColumns']
    columns = eval(pier_data['ColumnData']) if isinstance(pier_data['ColumnData'], str) else pier_data['ColumnData']
    hingedata = []

    for ii in range(n_columns):
        column_dict = dict(columns[ii])

        stiffness = column_dict['column_parameters']['strength_stiffness']['stiffness']   # kips-in/rad
        strength = column_dict['column_parameters']['strength_stiffness']['strength']     # kips-in

        bw_parameters = column_dict['model_parameters']

        # bw_parameters has the following parameters in order:
        # 0.    1.    2.   3.  4.  5.  6.   7.    8.    9.     10.    11.    12. 13.     14.
        # gamma kappa eta1 sig lam mup sigp rsmax alpha alpha1 alpha2 betam1 n   kappa_k res_min
        ph_params = [
            bw_parameters[2], # eta1
            stiffness*bw_parameters[0], # k0
            strength*bw_parameters[1],  # sy0
            bw_parameters[3], # sig
            bw_parameters[4], # lam
            bw_parameters[5], # mup
            bw_parameters[6], # sigp
            bw_parameters[7], # rsmax
            bw_parameters[12], # n
            bw_parameters[8], # alpha
            bw_parameters[9], # alpha1
            bw_parameters[10], # alpha2
            bw_parameters[11] # betam1
        ]

        ph_data = DegradingBoucWen_Fix(ph_params)

        # If n_columns > 1, we need to add hinge data twice (top and bottom of column)
        if n_columns > 1:
            hingedata.append(ph_data)
            hingedata.append(ph_data)
        else:
            hingedata.append(ph_data)

    # Let's bring all parameters to SI units
    E = column_dict['column_parameters']['strength_stiffness']['E'] # psi
    E = E * 0.5  # Stiffness reduction factor for cracked concrete
    I = column_dict['column_parameters']['strength_stiffness']['I']
    h = column_dict['height']
    
    # Get girder length from inventory (convert from feet to inches if needed)
    girder_length_ft = pier_data.get('GirderLength', 70)  # Default 70 feet
    # Ensure girder_length is a plain float (convert from numpy types if needed)
    girder_length_ft = float(girder_length_ft) if girder_length_ft is not None else 70.0
    girder_length_in = girder_length_ft * 12  # Convert to inches

    massdata = get_mass(float(h), float(girder_length_in)) # kips/(in/s^2)
    # If n_columns > 1, need to add the last two values again, as many times as n_columns - 1
    if n_columns > 1:
        massdata = massdata + [massdata[-2], massdata[-1]] * (n_columns - 1)
    else:
        massdata = massdata
    
    # Approximate mass
    print(f'Creating structure model with {n_columns} columns, E={E}, I={I}, h={h}, girder_length={girder_length_ft} ft')
    damping = [0.02]
    weight = massdata[0] * 386.4 # kips

    print(len(hingedata), n_columns)
    # Create structure model
    
    structure_model = InelasticFrame(n_columns, E, I, h, hingedata, massdata, damping, weight, geometry="p-delta", units="imperial")
    print('Structure model created.')

    return structure_model


# ============================================================================
# Spatial Analysis
# ============================================================================

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two 2D points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_nearest_neighbors(inventory_file, event_grid_file, output_path, n_neighbors=4, line_name=''):
    """
    Find nearest ground motion grid points for each pier location
    
    Parameters
    ----------
    inventory_file : str
        Path to pier inventory CSV with Latitude, Longitude columns
    event_grid_file : str
        Path to ground motion grid CSV with Latitude, Longitude, Site columns
    output_path : str
        Directory to save NearestNeighbors.csv
    n_neighbors : int
        Number of nearest neighbors to find (default: 4)
    line_name : str
        Line name to include in output filename (e.g., 'red', 'yellow')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Station_id and Neighbor_1...Neighbor_N columns
        
    Notes
    -----
    Uses simple Euclidean distance on lat/lon coordinates.
    For more accurate distances, consider using geodesic calculations.
    """
    # Read the inventory and event grid files
    inventory = pd.read_csv(inventory_file)
    event_grid = pd.read_csv(event_grid_file)

    # Extract the pier locations
    pier_locations = inventory[['Latitude', 'Longitude']].values
    grid_locations = event_grid[['Latitude', 'Longitude']].values

    # Create a list to store the nearest neighbors
    nearest_neighbors = []

    # For each pier location, find the nearest neighbors
    for idx, pier_loc in enumerate(pier_locations):
        distances = [euclidean_distance(pier_loc, grid_loc) for grid_loc in grid_locations]
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_sites = event_grid.iloc[nearest_indices]['GP_file'].values
        
        # Store the pier ID and its nearest neighbors
        neighbor_data = {'Station_id': inventory.iloc[idx]['id']}
        for i, site in enumerate(nearest_sites, 1):
            neighbor_data[f'Neighbor_{i}'] = site
        
        nearest_neighbors.append(neighbor_data)

    # Convert to DataFrame
    nearest_neighbors_df = pd.DataFrame(nearest_neighbors)

    # Save to CSV with line-specific filename
    if line_name:
        output_file = os.path.join(output_path, f'NearestNeighbors_{line_name}.csv')
    else:
        output_file = os.path.join(output_path, 'NearestNeighbors.csv')
    nearest_neighbors_df.to_csv(output_file, index=False)
    print(f"Nearest neighbors saved to {output_file}")

    return nearest_neighbors_df


# ============================================================================
# System Resources
# ============================================================================

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def monitor_system_resources():
    """
    Monitor current system resource utilization
    
    Returns
    -------
    dict
        cpu_percent: float
        memory_percent: float
        memory_mb: float - current process memory usage in MB
        available_memory_gb: float
    """
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_mb': get_memory_usage(),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3)
    }


def get_optimal_configuration():
    """
    Determine optimal parallel processing configuration based on system resources
    
    Returns
    -------
    dict
        batch_size: int - Recommended pier batch size
        max_workers: int - Recommended number of parallel workers
        use_processes: bool - Whether to use processes vs threads
        system_info: dict - CPU and memory information
        
    Notes
    -----
    Guidelines:
    - Uses 75% of available CPU cores
    - Limits workers based on available memory
    - Switches to threading if memory constrained
    - Adjusts batch size based on memory per worker
    """
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    
    # Base configuration
    max_workers = max(1, int(cpu_count * 0.75))  # Use 75% of CPUs
    
    # Adjust based on memory (assume ~500MB per worker for processes)
    memory_gb = memory.total / (1024**3)
    max_workers_by_memory = int(memory_gb / 0.5)
    
    if max_workers_by_memory < max_workers:
        print(f"Memory constraint: reducing workers from {max_workers} to {max_workers_by_memory}")
        max_workers = max(1, max_workers_by_memory)
    
    # Batch size: 2-8 piers per batch, scaled with worker count
    batch_size = min(8, max(2, max_workers // 2))
    
    # Use threading if very memory constrained
    use_processes = memory_gb > 4
    
    return {
        'batch_size': batch_size,
        'max_workers': max_workers,
        'use_processes': use_processes,
        'system_info': {
            'cpu_count': cpu_count,
            'total_memory_gb': memory_gb,
            'available_memory_gb': memory.available / (1024**3)
        }
    }


# ============================================================================
# Cache Management (Deprecated - use BartInventory instead)
# ============================================================================

def load_or_build_cached_models(inventory_path, cache_dir='./cached_models', verbose=True):
    """
    Load cached pier models or build them if cache doesn't exist
    
    DEPRECATED: Use BartInventory class instead for better cache management
    
    Parameters
    ----------
    inventory_path : str
        Path to pier inventory CSV
    cache_dir : str
        Directory containing or to create cache files
    verbose : bool
        Print progress messages
    
    Returns
    -------
    tuple
        (cached_models dict, periods DataFrame)
    """
    import warnings
    warnings.warn(
        "load_or_build_cached_models() is deprecated. Use BartInventory class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    pkl_path = os.path.join(cache_dir, 'pier_models.pkl')
    periods_path = os.path.join(cache_dir, 'pier_periods.csv')
    
    if os.path.exists(pkl_path) and os.path.exists(periods_path):
        if verbose:
            print(f"Loading cached models from {cache_dir}...")
        
        with open(pkl_path, 'rb') as f:
            cached_models = pickle.load(f)
        
        periods_df = pd.read_csv(periods_path)
        
        if verbose:
            print(f"Loaded {len(cached_models)} cached models")
        
        return cached_models, periods_df
    else:
        if verbose:
            print(f"Cache not found in {cache_dir}. Building models...")
        
        from build_pier_models import build_all_pier_models
        
        result = build_all_pier_models(
            inventory_path=inventory_path,
            output_dir=cache_dir,
            verbose=verbose
        )
        
        # Load the newly created cache
        with open(pkl_path, 'rb') as f:
            cached_models = pickle.load(f)
        
        periods_df = pd.read_csv(periods_path)
        
        return cached_models, periods_df
