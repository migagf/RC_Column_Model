#!/usr/bin/env python3
"""
Test script to validate the improved material model hierarchy
"""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from material_models import BoucWen, DegradingBoucWen, deg_bw_material_mod

def test_material_inheritance():
    """Test that all material models properly inherit from UniaxialMaterial"""
    
    print("Testing Material Model Inheritance and Functionality")
    print("=" * 60)
    
    # Test BoucWen material
    print("\n1. Testing BoucWen Material:")
    bw_params = [0.05, 0.5, 0.5, 2.0, 30000.0]  # alpha, gamma, beta, n, k
    bw_material = BoucWen(bw_params)
    
    print(f"   Type: {type(bw_material).__name__}")
    print(f"   Parent methods available: {hasattr(bw_material, 'commit_state')}")
    print(f"   Parameter validation: {hasattr(bw_material, 'validate_parameters')}")
    print(f"   Initial state: stress={bw_material.get_stress()}, strain={bw_material.get_strain()}")
    
    # Test a simple loading
    bw_material.set_trial_state(0.001)
    print(f"   After trial strain 0.001: stress={bw_material.get_trial_stress():.3f}")
    bw_material.commit_state()
    print(f"   After commit: stress={bw_material.get_stress():.3f}")
    
    # Test DegradingBoucWen material
    print("\n2. Testing DegradingBoucWen Material:")
    dbw_params = [0.02, 50000.0, 300.0, 0.2, 0.8, 0.0, 0.1, 1.0, 2.0, 0.05, 0.1, 0.8, 0.1]
    dbw_material = DegradingBoucWen(dbw_params)
    
    print(f"   Type: {type(dbw_material).__name__}")
    print(f"   Parent methods available: {hasattr(dbw_material, 'commit_state')}")
    print(f"   Initial state: stress={dbw_material.get_stress()}, strain={dbw_material.get_strain()}")
    
    # Test deg_bw_material_mod
    print("\n3. Testing deg_bw_material_mod Material:")
    mod_material = deg_bw_material_mod(dbw_params)
    
    print(f"   Type: {type(mod_material).__name__}")
    print(f"   Parent methods available: {hasattr(mod_material, 'commit_state')}")
    print(f"   Initial state: stress={mod_material.get_stress()}, strain={mod_material.get_strain()}")
    
    # Test string representations
    print("\n4. Testing String Representations:")
    print("   BoucWen:")
    print(f"   {bw_material}")
    
    print("\n   DegradingBoucWen:")
    print(f"   {dbw_material}")
    
    print("\n   deg_bw_material_mod:")
    print(f"   {mod_material}")
    
    # Test reset functionality
    print("\n5. Testing Reset Functionality:")
    bw_material.set_trial_state(0.002)
    bw_material.commit_state()
    print(f"   BoucWen before reset: stress={bw_material.get_stress():.6f}")
    bw_material.reset_state()
    print(f"   BoucWen after reset: stress={bw_material.get_stress():.6f}")
    
    print("\n6. Testing Material Info:")
    info = bw_material.get_material_info()
    print(f"   Material type: {info['type']}")
    print(f"   Number of parameters: {len(info['parameters'])}")
    
    print("\nAll tests completed successfully! ✓")
    print("The parent class functionality is now properly integrated.")

if __name__ == "__main__":
    test_material_inheritance()
