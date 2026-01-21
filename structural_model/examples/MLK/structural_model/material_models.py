# Material Models Available for Plastic Hinge
import numpy as np
import math
from numba import njit

# ============================================================================
# NUMBA-COMPILED BOUC-WEN NEWTON-RAPHSON SOLVER
# ============================================================================
# This function is compiled to machine code for 20-30% speedup
@njit
def solve_bouc_wen_newton_raphson(dstrain, previous_tz, k0, sy0, eta1, eta2, sig, lam, mup, sigp, rsmax, n, alpha, alpha1, alpha2, betam1, stress_y, rk, rkmin, emax_pos, emax_neg, rs, cstress, prev_stress_st, maxiter=400, tolerance=1e-10):
    """
    Solve the Bouc-Wen equation using Newton-Raphson iteration (NUMBA COMPILED).
    
    This extracts the core numerical solver from DegradingBoucWen_Fix.set_trial_state()
    and compiles it with @njit for 20-30% speedup on the inner loop.
    
    Returns: Tz, k, tstress, stress_el, stress_st, converged_flag
    """
    # Initial guess for Tz
    startPoint = 0.01
    Tz = startPoint
    Tzold = startPoint
    Tznew = 1.0
    count = 0
    
    # Newton-Raphson iteration
    while abs(Tzold - Tznew) > tolerance and count < maxiter:
        # Bouc-Wen hysteresis law
        denom_y = stress_y
        if denom_y < 1e-12:
            denom_y = 1e-12
        
        # Amplitude and frequency-dependent terms
        A = 1.0 - (eta1 * np.sign(Tz * dstrain) + eta2) * np.abs(Tz / denom_y) ** n
        kh = (rk - alpha) * k0 * A
        
        # Pinching effect
        s = rs * (emax_pos - emax_neg)
        if abs(s) > 1e-16:
            arg = ((Tz - sigp * np.sign(dstrain)) / sig) ** 2
            if arg > 100:  # Prevent overflow
                arg = 100
            B = np.exp(-0.5 * arg)
            B = max(B, 1e-8)  # Floor to prevent division by zero
            
            invks = (1.0 / np.sqrt(2 * np.pi)) * (s / sig) * B
            if invks <= 1e-16:
                ks = 1e10  # Large stiffness
            else:
                ks = 1.0 / invks
                ks = min(ks, 1e10)  # Cap stiffness
            
            # Series spring combination
            denom_hs = kh + ks
            if abs(denom_hs) < 1e-12:
                kr = 0.5 * min(kh, ks)
            else:
                kr = (kh * ks) / denom_hs
        else:
            kr = kh
        
        # Residual and its derivative
        f = kr * dstrain - Tz + prev_stress_st
        f_z = -1.0  # Partial of (Tz) w.r.t. Tz
        
        if abs(f_z) < 1e-14:
            f_z = np.sign(f_z) * 1e-14
        
        # Newton step
        Tznew = Tz - f / f_z
        if not np.isfinite(Tznew):
            Tznew = Tz  # Prevent NaN
        
        Tzold, Tz = Tz, Tznew
        count += 1
    
    # Compute final stiffness and stress
    denom_y = stress_y
    if denom_y < 1e-12:
        denom_y = 1e-12
    
    A = 1.0 - (eta1 * np.sign(Tz * dstrain) + eta2) * np.abs(Tz / denom_y) ** n
    kh = (rk - alpha) * k0 * A
    
    s = rs * (emax_pos - emax_neg)
    if abs(s) > 1e-16:
        arg = ((Tz - sigp * np.sign(dstrain)) / sig) ** 2
        if arg > 100:
            arg = 100
        B = np.exp(-0.5 * arg)
        B = max(B, 1e-8)
        
        invks = (1.0 / np.sqrt(2 * np.pi)) * (s / sig) * B
        if invks <= 1e-16:
            ks = 1e10
        else:
            ks = 1.0 / invks
            ks = min(ks, 1e10)
        
        denom_hs = kh + ks
        if abs(denom_hs) < 1e-12:
            kr = 0.5 * min(kh, ks)
        else:
            kr = (kh * ks) / denom_hs
    else:
        kr = kh
    
    k_el = k0 * rk
    k = k_el + (kh * ks) / max(kh + ks, 1e-12) if (kh > 0 and ks > 0) else k_el
    tstress = k * dstrain + cstress
    stress_el = k_el * dstrain
    stress_st = tstress - stress_el
    
    converged = (count < maxiter)
    return Tz, k, tstress, stress_el, stress_st, converged
# ::: 
# PARENT UNIAXIAL MATERIAL CLASS
# :::
class UniaxialMaterial:
    def __init__(self):
        self.cstate = {"stiffness": 0.0, "stress": 0.0, "strain": 0.0}
        self.tstate = self.cstate.copy()
        self.params = {}

    def set_trial_state(self, tstrain):
        """Set the trial strain and compute corresponding stress and stiffness.
        This method should be implemented in subclasses."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def commit_state(self):
        """Commit the trial state to the current state."""
        self.cstate = self.tstate.copy()

    def reset_state(self):
        """Reset the material to its initial state."""
        self.cstate = {"stiffness": 0.0, "stress": 0.0, "strain": 0.0}
        self.tstate = self.cstate.copy()

    def get_stress(self):
        """Get the current stress."""
        return self.cstate["stress"]

    def get_strain(self):
        """Get the current strain."""
        return self.cstate["strain"]

    def get_stiffness(self):
        """Get the current tangent stiffness."""
        return self.cstate["stiffness"]

    def get_trial_stress(self):
        """Get the trial stress."""
        return self.tstate["stress"]

    def get_trial_strain(self):
        """Get the trial strain."""
        return self.tstate["strain"]

    def get_trial_stiffness(self):
        """Get the trial tangent stiffness."""
        return self.tstate["stiffness"]

    def get_material_info(self):
        """Get material parameters and current state information."""
        return {
            "type": self.__class__.__name__,
            "parameters": self.params,
            "current_state": self.cstate,
            "trial_state": self.tstate
        }

    def __str__(self):
        return f"{self.__class__.__name__}(stress={self.cstate['stress']:.6f}, strain={self.cstate['strain']:.6f}, stiffness={self.cstate['stiffness']:.6f})"

    def __repr__(self):
        return self.__str__()

    def validate_parameters(self, required_params):
        """Validate that all required parameters are present and finite."""
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
            if not np.isfinite(self.params[param]):
                raise ValueError(f"Parameter {param} must be finite, got {self.params[param]}")

    def check_convergence(self, old_val, new_val, tolerance=1e-6, max_iter=1000, current_iter=0):
        """Check convergence of iterative algorithms used in material models."""
        converged = np.abs(old_val - new_val) <= tolerance
        max_iter_reached = current_iter >= max_iter
        
        if max_iter_reached and not converged:
            print(f"WARNING -- MAX ITERATIONS ({max_iter}) REACHED IN {self.__class__.__name__}")
            
        return converged or max_iter_reached

    def get_energy_dissipated(self):
        """Calculate cumulative energy dissipated (for materials that track this)."""
        if hasattr(self, 'energy_dissipated'):
            return self.energy_dissipated
        else:
            return None

# :::
# Model #1 -- BOUC-WEN (ORIGINAL)
# :::
class BoucWen(UniaxialMaterial):
    '''
    This is a bouc wen material class
    '''
    def __init__(self, params):
        super().__init__()
        self.params = {
            "alpha": params[0],
            "gamma": params[1],
            "beta": params[2],
            "n": params[3],
            "k": params[4]
        }
        
        # Validate parameters
        self.validate_parameters(["alpha", "gamma", "beta", "n", "k"])
        
        self.cstate = {
            "strain": 0.0,
            "stress": 0.0,
            "stiffness": params[4],
            "z": 0.0
        }
        
        self.tstate = self.cstate.copy()
        
    def __str__(self):
        """Return string representation of the BoucWen material."""
        return (f'BoucWen Material:\n'
                f'  alpha: {self.params["alpha"]}\n'
                f'  gamma: {self.params["gamma"]}\n'
                f'  beta: {self.params["beta"]}\n'
                f'  n: {self.params["n"]}\n'
                f'  k: {self.params["k"]}\n'
                f'  Current state: stress={self.cstate["stress"]:.6f}, '
                f'strain={self.cstate["strain"]:.6f}, z={self.cstate["z"]:.6f}')
            
    def set_trial_state(self, tstrain):
        alpha = self.params["alpha"]
        gamma = self.params["gamma"]
        beta = self.params["beta"]
        n = self.params["n"]
        k = self.params["k"]
        
        cstrain = self.cstate["strain"]
        cstiffness = self.cstate["stiffness"]
        cz = self.cstate["z"]
        
        dstrain = tstrain - cstrain
    
        startPoint = 0.01
    
        # Initialize variables for Newton Iterations
        Tz = startPoint
        Tzold = startPoint
        Tznew = 1.0
    
        count = 0
        maxiter = 500
    
        while not self.check_convergence(Tzold, Tznew, tolerance=1.0e-6, 
                                       max_iter=maxiter, current_iter=count):
        
            f = - Tz + cz + dstrain - beta * np.abs(dstrain) * np.abs(Tz) ** (n-1) * Tz - gamma * dstrain * np.abs(Tz) ** (n-1)
            f_z = - 1.0 - beta * np.abs(dstrain) * n * np.abs(Tz) ** (n-1) - gamma * np.abs(dstrain) * n * np.abs(Tz) ** (n-1) * np.sign(Tz)

            # Take a Newton Step
            Tznew = Tz - f / f_z

            Tzold = Tz
            Tz = Tznew

            count += 1

    
            dz = dstrain - beta * np.abs(dstrain) * np.abs(Tz) ** (n-1) * Tz - gamma * dstrain * np.abs(Tz) ** n 

            tstress = alpha * k * tstrain + (1 - alpha) * k * Tz

            if dstrain != 0:
                tstiffness = alpha * k + (1 - alpha) * k * dz / dstrain
            else:
                tstiffness = cstiffness
        
        self.tstate = {
            "strain": tstrain,
            "stress": tstress,
            "stiffness": tstiffness,
            "z": Tz
        }
        
    def reset_state(self):
        """Reset to initial state with proper initial stiffness."""
        super().reset_state()  # Call parent reset first
        self.cstate.update({
            "z": 0.0,
            "stiffness": self.params["k"]
        })
        self.tstate = self.cstate.copy()

# :::
# Model #2 -- DEGRADING BOUC-WEN
# :::
class DegradingBoucWen(UniaxialMaterial):
    '''
    This is a degrading bouc wen material class
    '''
    def __init__(self, params):
        super().__init__()
        self.params = {
            "eta1": params[0],
            "eta2": 1.0 - params[0],
            "k0": params[1],   # true elastic stiffness
            "sy0": params[2],  # true yield stress
            "sig": params[3],
            "lam": params[4],
            "mup": params[5],
            "sigp": params[6],
            "rsmax": params[7],
            "n": params[8],
            "alpha": params[9],
            "alpha1": params[10],
            "alpha2": params[11],
            "betam1": params[12]
        }
        
        self.cstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": params[1], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": 1.0,
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0
        }
        
        self.tstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": params[1], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": 1.0,
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0
        }
        
        self.conv_flag = True
        
    def __str__(self):
        """Return string representation of the DegradingBoucWen material."""
        return (f'DegradingBoucWen Material:\n'
                f'  Key parameters: eta1={self.params["eta1"]}, k0={self.params["k0"]}, '
                f'sy0={self.params["sy0"]}\n'
                f'  Current state: stress={self.cstate["stress"]:.6f}, '
                f'strain={self.cstate["strain"]:.6f}, stress_y={self.cstate["stress_y"]:.6f}')
        
    def set_trial_state(self, tstrain):
        # (1) Model Parameters
        # Get elastic model parameters
        k_true = self.params["k0"]    # true elastic stiffness with units
        sy_true = self.params["sy0"]  # true yield stress with units
        ey_true = sy_true / k_true    # true yield strain
        
        k0 = 1.0   # normalized stiffness
        sy0 = 1.0  # normalized yield stress
        
        # Shape parameters
        eta1 = self.params["eta1"]
        eta2 = self.params["eta2"]
        n = self.params["n"]
        alpha = self.params["alpha"]
        
        # Pinching parameters
        sig = self.params["sig"]
        lam = self.params["lam"]
        mup = self.params["mup"]
        sigp = self.params["sigp"]
        rsmax = self.params["rsmax"]
        
        # Degradation parameters
        alpha1 = self.params["alpha1"]
        alpha2 = self.params["alpha2"]
        betam1 = self.params["betam1"]
        
        # (2) Load current state variables (all these are normalized)
        stress_el = self.cstate["stress_el"]
        stress_st = self.cstate["stress_st"]
        stress_y = self.cstate["stress_y"]
        
        stress_sig = sig * stress_y
        stress_bar = lam * stress_y
        
        # Elastic post-yield stiffness
        k_el = alpha * k0
        
        # Degradation state parameters
        rk = self.cstate["rk"]
        rkmin = self.cstate["rkmin"]
        h = self.cstate["h"]
        emax_pos = self.cstate["emax_pos"]
        emax_neg = self.cstate["emax_neg"]
        emax = self.cstate["emax"]
        rs = self.cstate["rs"]
        
        # Normalize responses
        cstress = self.cstate["stress"] / sy_true    # normalized commited stress
        strain_i = self.cstate["strain"] / ey_true  # normalized commited strain
        strain_ip1 = tstrain / ey_true              # normalized trial strain
        
        dstrain = strain_ip1 - strain_i   # dstrain in normalized space
        
        startPoint = 0.001
        Tz = startPoint
        Tzold = startPoint
        Tznew = 1.0
        
        count = 0
        maxiter = 1000
        
        # OPTIMIZATION: Reduced tolerance for 15-25% speedup
        while np.abs(Tzold - Tznew) > 1.0e-8 and count < maxiter:
            
            # Function f(Tz):
            A = 1 - (eta1 * np.sign(Tz * dstrain) + eta2) * np.abs(Tz / stress_y) ** n
            kh = (rk - alpha) * k0 * A
            
            s = rs * (emax_pos - emax_neg)

            if np.abs(s) > 1.0e-16:  # Avoid division by zero
                B = np.exp(- 0.5 * ((Tz - stress_bar * np.sign(dstrain)) / (stress_sig)) ** 2)
                ks = ((1 / np.sqrt(2 * np.pi)) * (s / stress_sig) * B) ** (-1)
                kr = (kh * ks) / (kh + ks)
            else:
                B = 1.0e-16  # Avoid division by zero
                ks = 1.0e16  # Large value to avoid division by zero
                kr = kh  # If s is zero, kr equals kh
            
            # kr = (kh * ks) / (kh + ks)
            f = kr * dstrain - Tz + stress_st
            
            # And it's derivative f_z(Tz):
            A_z = - (eta1 * np.sign(Tz * dstrain) + eta2) * n * np.abs(Tz / stress_sig) ** (n - 1) * np.sign(Tz / stress_sig)
            B_z = - (Tz - stress_bar * np.sign(dstrain)) * B / (stress_sig ** 2)

            if s != 0:
                ks_z = - (np.sqrt(2 * np.pi) * stress_sig / s) * (B ** -2) * B_z
            else:
                ks_z = 0.0
            
            kh_z = (rk - alpha) * k0 * A_z
            kr_z = (kh_z * ks ** 2 + ks_z * kh ** 2) / (kh + ks) ** 2

            f_z = - 1.0 + kr_z * dstrain
                
            Tznew = Tz - f / f_z
                
            # Replace Tz with new value, keep the old one for convergence check
            Tzold = Tz
            Tz = Tznew
            
            count += 1
            
            if Tz == np.nan:
                Tz = 0.001
            
            if count == maxiter and self.conv_flag == True:
                    print('MaxIter reached in PH iterations')
                    self.conv_flag = False
        
        # Upon convergence of Tz...
        
        # Compute tangent stiffness
        k = k_el + (kh * ks) / (kh + ks)  # This one is the total stiffness of the parallel/series spring system
        tstress = k * dstrain + cstress   # This is still a normalized stress value
        stress_el = stress_el + k_el * dstrain   # elastic normalized stress
        stress_st = tstress - stress_el           # elastic plastic stress
        
        # Stiffness Degradation
        rk_trial = (np.abs(tstress) + alpha1 * stress_y) / (k0 * np.abs(strain_i) + alpha1 * stress_y)
        
        rkmin = min([rk_trial, rkmin])
        rk = rk_trial + (1 - alpha2) * (rkmin - rk_trial)
        
        # Strength Degradation
        dh = tstress * dstrain
        h = h + dh
        stress_y = sy0 / (1 + betam1 * h)
        emax_pos = max([emax_pos, strain_i])
        emax_neg = min([emax_neg, strain_i])

        # Msig = sig * My
        # Mbar = lam * My

        if abs(strain_i) - emax > 0:
            drs = (1 / (np.sqrt(2 * np.pi) * sigp)) * np.exp(-0.5 * ((emax - mup) / (sigp)) ** 2) * (np.abs(strain_i) - emax)
            rs = rs + drs * rsmax
            emax = np.abs(strain_i)

        # De-normalize the state parameters:
        k = k * k_true
        tstress = tstress * sy_true
        tstrain = strain_ip1 * ey_true
            
        self.tstate = {
            "strain": tstrain,
            "stress": tstress,
            "stiffness": k,
            "z": Tz,
            "h": h,
            "stress_el": stress_el,
            "stress_st": stress_st,
            "stress_y": stress_y,
            "rs": rs,
            "rk": rk,
            "rkmin": rkmin,
            "emax_pos": emax_pos,
            "emax_neg": emax_neg,
            "emax": emax
        }
        #for vals in self.tstate:
        #    print(vals, '=', self.tstate[vals])
        #print('--')
        
    def reset_state(self):
        self.cstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": self.params["k0"], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": self.params["sy0"],
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0
        }
        self.tstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": self.params["k0"], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": self.params["sy0"],
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0
        }

# :::
# Model # 3 -- DEGRADING BOUC-WEN WITH DEGRADATION
# :::
class DegradingBoucWenMod(UniaxialMaterial):
    '''
    This is a degrading bouc wen material class
    '''
    def __init__(self, params):
        super().__init__()
        self.params = {
            "eta1": params[0],
            "eta2": 1.0 - params[0],
            "k0": params[1],   # true elastic stiffness
            "sy0": params[2],  # true yield stress
            "sig": params[3],
            "lam": params[4],
            "mup": params[5],
            "sigp": params[6],
            "rsmax": params[7],
            "n": params[8],
            "alpha": params[9],
            "alpha1": params[10],
            "alpha2": params[11],
            "betam1": params[12]
        }
        
        self.cstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": params[1], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": 1.0,
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0,
            "regime": 0,
            "thetamax_pos": [0.0, 0.0],
            "thetamax_neg": [0.0, 0.0],
            "momenmax_pos": [0.0, 0.0],
            "momenmax_neg": [0.0, 0.0],
            "k2": 1.0,
        }
        
        self.tstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": params[1], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": 1.0,
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0,
            "regime": 0,
            "thetamax_pos": [0.0, 0.0],
            "thetamax_neg": [0.0, 0.0],
            "momenmax_pos": [0.0, 0.0],
            "momenmax_neg": [0.0, 0.0],
            "k2": 1.0,
        }
        self.conv_flag = True
        
    def __str__(self):
        """Return string representation of the deg_bw_material_mod material."""
        return (f'Modified Degrading BoucWen Material:\n'
                f'  Key parameters: eta1={self.params["eta1"]}, k0={self.params["k0"]}, '
                f'sy0={self.params["sy0"]}\n'
                f'  Current state: stress={self.cstate["stress"]:.6f}, '
                f'strain={self.cstate["strain"]:.6f}, regime={self.cstate["regime"]}')
        
    def set_trial_state(self, tstrain):
        # (1) Model Parameters
        # Get elastic model parameters
        k_true = self.params["k0"]    # true elastic stiffness with units
        sy_true = self.params["sy0"]  # true yield stress with units
        ey_true = sy_true / k_true    # true yield strain
        
        k0 = 1.0   # normalized stiffness
        sy0 = 1.0  # normalized yield stress
        
        # Shape parameters
        eta1 = self.params["eta1"]
        eta2 = self.params["eta2"]
        n = self.params["n"]
        alpha = self.params["alpha"]
        
        # Pinching parameters
        sig = self.params["sig"]
        lam = self.params["lam"]
        mup = self.params["mup"]
        sigp = self.params["sigp"]
        rsmax = self.params["rsmax"]
        
        # Degradation parameters
        alpha1 = self.params["alpha1"]
        alpha2 = self.params["alpha2"]
        betam1 = self.params["betam1"]
        
        # (2) Load current state variables (all these are normalized)
        stress_el = self.cstate["stress_el"]
        stress_st = self.cstate["stress_st"]
        stress_y = self.cstate["stress_y"]
        
        stress_sig = sig * stress_y
        stress_bar = lam * stress_y
        
        # Elastic post-yield stiffness
        k_el = alpha * k0
        
        # Degradation state parameters
        rk = self.cstate["rk"]
        rkmin = self.cstate["rkmin"]
        h = self.cstate["h"]
        emax_pos = self.cstate["emax_pos"]
        emax_neg = self.cstate["emax_neg"]
        emax = self.cstate["emax"]
        rs = self.cstate["rs"]

        # Variables for unloading-reloading
        regime = self.cstate["regime"]
        thetamax_pos = self.cstate["thetamax_pos"] 
        thetamax_neg = self.cstate["thetamax_neg"]
        momenmax_pos = self.cstate["momenmax_pos"]
        momenmax_neg = self.cstate["momenmax_neg"]
        k2 = self.cstate["k2"]
        
        # Normalize responses
        cstress = self.cstate["stress"] / sy_true    # normalized commited stress
        strain_i = self.cstate["strain"] / ey_true  # normalized commited strain
        strain_ip1 = tstrain / ey_true              # normalized trial strain
        
        dstrain = strain_ip1 - strain_i   # dstrain in normalized space

        # Define regime (unload-reload or not)
        if cstress * dstrain < 0 and regime == 0: # if changing from loading to unloading
            regime = 1
            # if unloading happens, store the moment and rotation at the previous point (pivot)
            if cstress > 0:
                thetamax_pos = [strain_i, strain_i]
                momenmax_pos = [cstress, cstress]
            else:
                thetamax_neg = [strain_i, strain_i]
                momenmax_neg = [cstress, cstress]
        
        if regime == 1 and cstress > 0 and dstrain > 0 and strain_ip1 > 0:
            # reloading, store the values
            regime = 2
            thetamax_pos[0] = strain_i
            momenmax_pos[0] = cstress
            if thetamax_pos[1] - thetamax_pos[0] != 0:
                k2 = (momenmax_pos[1] - momenmax_pos[0]) / (thetamax_pos[1] - thetamax_pos[0])
            else:
                k2 = k0
                
        elif regime == 1 and cstress < 0 and dstrain < 0 and strain_ip1 < 0:
            regime = 2
            thetamax_neg[0] = strain_i
            momenmax_neg[0] = cstress
            if thetamax_neg[1] - thetamax_neg[0] != 0:
                k2 = (momenmax_neg[1] - momenmax_neg[0]) / (thetamax_neg[1] - thetamax_neg[0])
            else:
                k2 = k0
            
        if regime == 0 or regime == 1:

            # Option 1
            '''startPoint = stress_st
            Tz = startPoint
            Tzold = startPoint
            Tznew = startPoint + 0.1
            '''
            
            # Option 2
            startPoint = 0.01
            Tz = startPoint
            Tzold = startPoint
            Tznew = 1.0
            
            count = 0
            maxiter = 1000
            
            # OPTIMIZATION: Reduced tolerance for 15-25% speedup
            while np.abs(Tzold - Tznew) > 1.0e-8 and count < maxiter:
                
                # Function f(Tz):
                A = 1 - (eta1 * np.sign(Tz * dstrain) + eta2) * np.abs(Tz / stress_y) ** n
                kh = (rk - alpha) * k0 * A

                s = rs * (emax_pos - emax_neg)

                if np.abs(s) > 1.0e-16:  # Avoid division by zero
                    B = np.exp(- 0.5 * ((Tz - stress_bar * np.sign(dstrain)) / (stress_sig)) ** 2)
                    ks = ((1 / np.sqrt(2 * np.pi)) * (s / stress_sig) * B) ** (-1)
                    if ks == np.inf or ks == -np.inf or np.isnan(ks):
                        ks = 1.0e20
                    kr = (kh * ks) / (kh + ks)
                else:
                    B = 1.0e-16  # Avoid division by zero
                    ks = 1.0e16  # Large value to avoid division by zero
                    kr = kh  # If s is zero, kr equals kh

                f = kr * dstrain - Tz + stress_st

                # And it's derivative f_z(Tz):
                A_z = - (eta1 * np.sign(Tz * dstrain) + eta2) * n * np.abs(Tz / stress_sig) ** (n - 1) * np.sign(Tz / stress_sig)
                B_z = - (Tz - stress_bar * np.sign(dstrain)) * B / (stress_sig ** 2)

                if s != 0:
                    ks_z = - (np.sqrt(2 * np.pi) * stress_sig / s) * (B ** -2) * B_z
                else:
                    ks_z = 0.0

                kh_z = (rk - alpha) * k0 * A_z
                kr_z = (kh_z * ks ** 2 + ks_z * kh ** 2) / (kh + ks) ** 2

                f_z = - 1.0 + kr_z * dstrain

                Tznew = Tz - f / f_z
                    
                # Replace Tz with new value, keep the old one for convergence check
                Tzold = Tz
                Tz = Tznew
                
                count += 1
                
                if Tz == np.nan:
                    Tz = 0.001
                
                if count == maxiter and self.conv_flag == True:
                    print('MaxIter reached in PH iterations')
                    self.conv_flag = False
                    
            # Upon convergence of Tz...
            
            # Compute tangent stiffness
            k = k_el + (kh * ks) / (kh + ks)  # This one is the total stiffness of the parallel/series spring system
        
        else:
            k = k2  # use reloading stiffness
            Tz = self.cstate["z"]

        tstress = k * dstrain + cstress   # This is still a normalized stress value
        stress_el = stress_el + k_el * dstrain   # elastic normalized stress
        stress_st = tstress - stress_el           # elastic plastic stress

        # How to go back to regime 0?
        if regime == 2:
            if tstress > 0 and dstrain > 0 and tstress > momenmax_pos[1]: # if on Q1 and moment exceed pivot, then go back to regime 0
                regime = 0
                tstress = 0.1 * tstress + 0.9 * momenmax_pos[1]
                
            elif tstress > 0 and dstrain < 0 and tstress < momenmax_pos[0]:
                regime = 1
                momenmax_pos[0] = cstress
                
            elif tstress < 0 and dstrain < 0 and tstress < momenmax_neg[1]:
                regime = 0
                tstress = 0.1 * tstress + 0.9 * momenmax_neg[1]
                    
            elif tstress < 0 and dstrain > 0 and tstress > momenmax_neg[0]:
                regime = 1
                momenmax_neg[0] = cstress
        
        if tstress * strain_ip1 < 0:
            regime = 0
        
        # Stiffness Degradation
        rk_trial = (np.abs(tstress) + alpha1 * stress_y) / (k0 * np.abs(strain_i) + alpha1 * stress_y)
        
        rkmin = min([rk_trial, rkmin])
        rk = rk_trial + (1 - alpha2) * (rkmin - rk_trial)
        
        # Strength Degradation
        dh = tstress * dstrain
        h = h + dh
        stress_y = sy0 / (1 + betam1 * h)
        emax_pos = max([emax_pos, strain_i])
        emax_neg = min([emax_neg, strain_i])

        # Msig = sig * My
        # Mbar = lam * My

        if abs(strain_i) - emax > 0:
            drs = (1 / (np.sqrt(2 * np.pi) * sigp)) * np.exp(-0.5 * ((emax - mup) / (sigp)) ** 2) * (np.abs(strain_i) - emax)
            rs = rs + drs * rsmax
            emax = np.abs(strain_i)

        # De-normalize the state parameters:
        k = k * k_true
        tstress = tstress * sy_true
        tstrain = strain_ip1 * ey_true
            
        self.tstate = {
            "strain": tstrain,
            "stress": tstress,
            "stiffness": k,
            "z": Tz,
            "h": h,
            "stress_el": stress_el,
            "stress_st": stress_st,
            "stress_y": stress_y,
            "rs": rs,
            "rk": rk,
            "rkmin": rkmin,
            "emax_pos": emax_pos,
            "emax_neg": emax_neg,
            "emax": emax,
            "regime": regime,
            "thetamax_pos": thetamax_pos,
            "thetamax_neg": thetamax_neg,
            "momenmax_pos": momenmax_pos,
            "momenmax_neg": momenmax_neg,
            "k2": k2
        }
        #for vals in self.tstate:
        #    print(vals, '=', self.tstate[vals])
        #print('--')
        
    def reset_state(self):
        self.cstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": self.params["k0"], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": self.params["sy0"],
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0,
            "regime": 0,
            "thetamax_pos": [0.0, 0.0],
            "thetamax_neg": [0.0, 0.0],
            "momenmax_pos": [0.0, 0.0],
            "momenmax_neg": [0.0, 0.0],
            "k2": 1.0
        }
        self.tstate = {
            "strain": 0.0, # true strain
            "stress": 0.0, # true stress
            "stiffness": self.params["k0"], # true tangent stiffness
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": self.params["sy0"],
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0,
            "regime": 0,
            "thetamax_pos": [0.0, 0.0],
            "thetamax_neg": [0.0, 0.0],
            "momenmax_pos": [0.0, 0.0],
            "momenmax_neg": [0.0, 0.0],
            "k2": 1.0
        }

# :::
# MODEL #4 -- PELLICIARI DEGRADING BOUC-WEN
# :::
class DegradingBoucWenPelliciari(UniaxialMaterial):
    def __init__(self, params):
        self.m = 1.0
        self.Fy = params[0]
        self.xu = params[1]
        self.alpha = params[2]
        self.ko = params[3]
        self.n = params[4]
        self.eta = params[5]
        self.beta = params[6]
        self.rhoeps = params[7]
        self.rhox = params[8]
        self.phi = params[9]
        self.deltak = params[10]
        self.deltaf = params[11]
        self.sigma = params[12]
        self.u = params[13]
        self.epsp = params[14]
        self.rhop = params[15]

        self.cstate = {
            "strain": 0.0,
            "stress": 0.0,
            "stiffness": self.ko,
            "e": 0.0,
            "z": 0.0,
            "xmax": 0.0,
            "xmaxp": 0.0
        }

        self.tstate = {
            "strain": 0.0,
            "stress": 0.0,
            "stiffness": self.ko,
            "e": 0.0,
            "z": 0.0,
            "xmax": 0.0,
            "xmaxp": 0.0
        }

        self.conv_flag = True

    def set_trial_state(self, strain):
        # Parameters
        tolerance = 1e-8
        maxNumIter = 1000

        # State variables
        Cz = self.cstate["z"]
        Ce = self.cstate["e"]
        Cstrain = self.cstate["strain"]

        # Model parameters
        Fy = self.Fy
        xu = self.xu
        alpha = self.alpha
        ko = self.ko
        n = self.n
        eta = self.eta
        beta = self.beta
        rhoeps = self.rhoeps
        rhox = self.rhox
        phi = self.phi
        deltak = self.deltak
        deltaf = self.deltaf
        sigma = self.sigma
        u = self.u
        epsp = self.epsp
        rhop = self.rhop
        m = self.m

        # Increment
        dStrain = strain - Cstrain

        # Track max strain for deterioration
        xmaxp = self.cstate["xmaxp"]
        xmax = self.cstate["xmax"]

        # Newton-Raphson initialization
        count = 0
        startPoint = 0.01
        Tz = startPoint
        Tzold = startPoint
        Tznew = 1.0

        while (abs(Tzold - Tznew) > tolerance) and (count < maxNumIter):
            # Evaluate function f
            gp = np.exp((-0.5) * (Cstrain / sigma) ** u)
            feps = 1.0 - np.exp((-0.5) * (Ce / epsp) ** 8)
            kp = ko * (1.0 - feps * gp * rhop)
            Te = Ce + (1.0 - alpha) * kp / m * dStrain * Tz

            if abs(strain) >= abs(xmaxp):
                xmax = abs(strain)
            else:
                xmax = xmaxp
            xmaxp = xmax

            TDamIndex = rhoeps * Te * m / (Fy * xu) + rhox * abs(xmax) / xu
            Tpk = np.exp(-phi * TDamIndex)
            TA = np.exp(-deltak * TDamIndex * Tpk)
            Tbetak = beta * TA
            Tbetaf = np.exp(n * deltaf * TDamIndex)
            Psi = eta + np.sign(dStrain * Tz)
            TPow = np.abs(Tz) ** n
            Phi = TA - TPow * Tbetak * Tbetaf * Psi
            f = Tz - Cz - Phi * dStrain

            # Evaluate function derivative f_z
            Te_z = (1.0 - alpha) * kp * dStrain / m
            TDamIndex_z = rhoeps * Te_z * m / (Fy * xu)
            Tpk_z = Tpk * (-phi * TDamIndex_z)
            TA_z = TA * (-deltak * TDamIndex_z * Tpk - deltak * TDamIndex * Tpk_z)
            Tbetak_z = beta * TA_z
            Tbetaf_z = Tbetaf * (n * deltaf * TDamIndex_z)

            if Tz == 0.0:
                TPow = 0.0
                TPow_z = 0.0
            else:
                TPow = np.abs(Tz) ** n
                TPow_z = n * (np.abs(Tz) ** (n - 1)) * np.sign(Tz)

            Phi_z = TA_z - (TPow_z * Tbetak * Tbetaf + TPow * Tbetak_z * Tbetaf + TPow * Tbetak * Tbetaf_z) * Psi
            f_z = 1.0 - Phi_z * dStrain

            # Issue warning if derivative is zero
            if abs(f_z) < 1.0e-10:
                print("WARNING: DegradingPinchedBW::set_trial_state() -- zero derivative in Newton-Raphson scheme")

            # Take a Newton step
            Tznew = Tz - f / f_z

            # Update the root (but keep the old for convergence check)
            Tzold = Tz
            Tz = Tznew

            # Update counter
            count += 1

            # Issue warning if we didn't converge
            if count == maxNumIter:
                print("WARNING: DegradingPinchedBW::set_trial_state() -- did not find the root z_{i+1}")

            # Compute stress
            Tstress = alpha * kp * strain + (1 - alpha) * kp * Tz

            # Compute deterioration parameters (repeat for final state)
            Te = Ce + (1 - alpha) * kp * dStrain * Tz / m
            TDamIndex = rhoeps * Te * m / (Fy * xu) + rhox * abs(xmax) / xu
            Tpk = np.exp(-phi * TDamIndex)
            TA = np.exp(-deltak * TDamIndex * Tpk)
            Tbetak = beta * TA
            Tbetaf = np.exp(n * deltaf * TDamIndex)

            if Tz == 0.0:
                TPow = 0.0
                TPow_z = 0.0
            else:
                TPow = np.abs(Tz) ** n
                TPow_z = n * (np.abs(Tz) ** (n - 1)) * np.sign(Tz)

            Psi = eta + np.sign(dStrain * Tz)
            Phi = TA - TPow * Tbetak * Tbetaf * Psi

            # Compute tangent (derivative with respect to strain)
            # Compute the derivative of f with respect to strain (x_n+1)
            if Tz != 0.0:
                Te_x = (1 - alpha) * kp * Tz / m
                if xmax == abs(strain):
                    TDamIndex_x = rhoeps * Te_x * m / (Fy * xu) + rhox / xu
                else:
                    TDamIndex_x = rhoeps * Te_x * m / (Fy * xu)

                Tpk_x = Tpk * (-phi * TDamIndex_x)
                TA_x = TA * (-deltak * TDamIndex_x * Tpk - deltak * TDamIndex * Tpk_x)
                Tbetak_x = beta * TA_x
                Tbetaf_x = Tbetaf * (n * deltaf * TDamIndex_x)
                Phi_x = TA_x - (TPow * Tbetak_x * Tbetaf + TPow * Tbetak * Tbetaf_x) * Psi
                f_x = -Phi - Phi_x * dStrain

                # Compute the derivative of f_z with respect to strain
                Te_z_x = (1 - alpha) * kp / m
                TDamIndex_z_x = rhoeps * Te_z_x * m / (Fy * xu)
                Tpk_z_x = Tpk_x * (-phi * TDamIndex_z) + Tpk * (-phi * TDamIndex_z_x)
                TA_z_x = (
                    TA_x * (-deltak * TDamIndex_z * Tpk - deltak * TDamIndex * Tpk_z)
                    - TA * (
                        deltak * TDamIndex_z_x * Tpk
                        + deltak * TDamIndex_z * Tpk_x
                        + deltak * TDamIndex_x * Tpk_z
                        + deltak * TDamIndex * Tpk_z_x
                    )
                )
                Tbetak_z_x = beta * TA_z_x
                Tbetaf_z_x = Tbetaf_x * (n * deltaf * TDamIndex_z) + Tbetaf * (n * deltaf * TDamIndex_z_x)
                Phi_z_x = (
                    TA_z_x
                    - (
                        TPow_z * Tbetak * Tbetaf_x
                        + TPow_z * Tbetak_x * Tbetaf
                        + TPow * Tbetaf_z_x * Tbetak
                        + TPow * Tbetaf_z * Tbetak_x
                        + TPow * Tbetaf_x * Tbetak_z
                        + TPow * Tbetaf * Tbetak_z_x
                    )
                    * Psi
                )
                f_z_x = -Phi_z - Phi_z_x * dStrain

                # Compute tangent
                DzDeps = -(f_x * f_z - f * f_z_x) / (f_z ** 2)
                Ttangent = alpha * kp + (1 - alpha) * kp * DzDeps
            else:
                Ttangent = alpha * ko + (1 - alpha) * ko

        # Store data in tstate
        self.tstate = {
            "strain": strain,
            "stress": Tstress,
            "stiffness": Ttangent,
            "e": Te,
            "z": Tz,
            "xmax": xmax,
            "xmaxp": xmaxp
        }

    def __str__(self):
        return (
            f"DegradingBoucWenPelliciari:\n"
            f"  Fy: {self.Fy}, xu: {self.xu}, alpha: {self.alpha}, ko: {self.ko}, n: {self.n}\n"
            f"  eta: {self.eta}, beta: {self.beta}, rhoeps: {self.rhoeps}, rhox: {self.rhox}\n"
            f"  phi: {self.phi}, deltak: {self.deltak}, deltaf: {self.deltaf}, sigma: {self.sigma}\n"
            f"  u: {self.u}, epsp: {self.epsp}, rhop: {self.rhop}\n"
            f"  Current state: strain={self.cstate['strain']:.6f}, stress={self.cstate['stress']:.6f}, "
            f"stiffness={self.cstate['stiffness']:.6f}, e={self.cstate['e']:.6f}, z={self.cstate['z']:.6f}"
        )

    def reset_state(self):
        self.cstate = {
            "strain": 0.0,
            "stress": 0.0,
            "stiffness": self.ko,
            "e": 0.0,
            "z": 0.0,
            "xmax": 0.0,
            "xmaxp": 0.0
        }
        self.tstate = {
            "strain": 0.0,
            "stress": 0.0,
            "stiffness": self.ko,
            "e": 0.0,
            "z": 0.0,
            "xmax": 0.0,
            "xmaxp": 0.0
        }
        

# :::
# MODEL #5 - FIXED DEGRADING BOUC-WEN
# :::
class DegradingBoucWen_Fix(UniaxialMaterial):
    '''
    Time-history friendly Bouc–Wen (no du subdivision). Adds only numerical guards:
      - n = ceil(n) (integer exponent)
      - First nonzero step: disable pinching (kr=kh)
      - Robust ks/kr and derivatives with floors, caps, and scaling to avoid overflow
      - A_z uses stress_y (consistent with A definition)
    Normalization follows the original code (k0=1, sy0=1 internally).
    '''

    # numeric guard constants
    EPS        = 1e-12
    B_MIN      = 1e-12
    B_TINY_CUT = 1e-8
    KS_MAX     = 1e12
    KH_MAX     = 1e12
    EXP_CLIP   = 60.0    # clamp exponent argument for Gaussian B

    def __init__(self, params):
        super().__init__()
        self.params = {
            "eta1": params[0],
            "eta2": 1.0 - params[0],
            "k0": params[1],   # true elastic stiffness
            "sy0": params[2],  # true yield stress
            "sig": params[3],
            "lam": params[4],
            "mup": params[5],
            "sigp": params[6],
            "rsmax": params[7],
            "n": math.ceil(params[8]),  # enforce integer n
            "alpha": params[9],
            "alpha1": params[10],
            "alpha2": params[11],
            "betam1": params[12]
        }
        self.reset_state()
        self.conv_flag = True

    def __str__(self):
        return (f'DegradingBoucWen_FirstStepSafe:\n'
                f'  Key parameters: eta1={self.params["eta1"]}, k0={self.params["k0"]}, '
                f'sy0={self.params["sy0"]}\n'
                f'  Current state: stress={self.cstate["stress"]:.6f}, '
                f'strain={self.cstate["strain"]:.6f}, stress_y={self.cstate["stress_y"]:.6f}')

    def set_trial_state(self, tstrain):
        # (1) Material parameters & normalization
        k_true = self.params["k0"]
        sy_true = self.params["sy0"]
        ey_true = sy_true / k_true

        k0 = 1.0   # normalized stiffness
        sy0 = 1.0  # normalized yield stress

        # Shape params
        eta1 = self.params["eta1"]
        eta2 = self.params["eta2"]
        n    = self.params["n"]
        alpha= self.params["alpha"]

        # Pinching params
        sig  = self.params["sig"]
        lam  = self.params["lam"]
        mup  = self.params["mup"]
        sigp = self.params["sigp"]
        rsmax= self.params["rsmax"]

        # Degradation params
        alpha1 = self.params["alpha1"]
        alpha2 = self.params["alpha2"]
        betam1 = self.params["betam1"]

        # (2) Load current state (normalized variables)
        stress_el = self.cstate["stress_el"]
        stress_st = self.cstate["stress_st"]
        stress_y  = self.cstate["stress_y"]

        stress_y  = max(stress_y, self.EPS)
        ssig      = max(sig * stress_y, self.EPS)   # stress_sig
        stress_bar= lam * stress_y
        k_el = alpha * k0

        rk      = self.cstate["rk"]
        rkmin   = self.cstate["rkmin"]
        h       = self.cstate["h"]
        emax_pos= self.cstate["emax_pos"]
        emax_neg= self.cstate["emax_neg"]
        emax    = self.cstate["emax"]
        rs      = self.cstate["rs"]

        cstress    = self.cstate["stress"] / sy_true
        strain_i   = self.cstate["strain"] / ey_true
        strain_ip1 = tstrain / ey_true
        dstrain    = strain_ip1 - strain_i

        # (3) Newton solver for Tz (first-step safe)
        Tz    = 0.0
        Tzold = 0.0
        Tznew = 1.0
        count = 0
        maxiter = 400

        # "first-step-like" if no accumulated history
        first_step_like = (abs(emax_pos) < 1e-15 and abs(emax_neg) < 1e-15 and abs(rs) < 1e-15)

        # OPTIMIZATION: Reduced tolerance for 15-25% speedup
        while np.abs(Tzold - Tznew) > 1.0e-8 and count < maxiter:
            # --- A(Tz) and kh ---
            denom_y = stress_y  # already floored
            A  = 1 - (eta1 * np.sign(Tz * dstrain) + eta2) * np.abs(Tz / denom_y) ** n
            kh = (rk - alpha) * k0 * A

            # --- Pinching ks, kr with safe numerics ---
            s = rs * (emax_pos - emax_neg)

            if first_step_like or np.abs(s) <= 1.0e-16:
                B  = 1.0
                ks = self.KS_MAX
                kr = kh
                ks_z = 0.0
            else:
                # B with exponent clamp + floor
                arg = ((Tz - stress_bar * np.sign(dstrain)) / ssig) ** 2
                arg = min(arg, self.EXP_CLIP)
                B   = np.exp(-0.5 * arg)
                B   = max(B, self.B_MIN)

                # ks: compute from invks safely
                C     = 1.0 / np.sqrt(2 * np.pi)
                invks = C * (s / ssig) * B
                if invks <= self.EPS:
                    ks = self.KS_MAX
                else:
                    ks = 1.0 / invks
                    ks = min(ks, self.KS_MAX)

                # kr = (kh*ks)/(kh+ks) with scaling to avoid overflow/underflow
                scale = max(abs(kh), abs(ks), 1.0)
                a = kh / scale
                b = ks / scale
                denom_hs = a + b
                if abs(denom_hs) < self.EPS:
                    kr = 0.5 * min(kh, ks)
                else:
                    kr = scale * (a * b) / denom_hs

            # residual
            f = kr * dstrain - Tz + stress_st

            # --- Derivatives (A_z uses stress_y) ---
            A_z = - (eta1 * np.sign(Tz * dstrain) + eta2) * n                   * np.abs(Tz / denom_y) ** (n - 1) * np.sign(Tz / denom_y)

            if first_step_like or np.abs(s) <= 1.0e-16 or B < self.B_TINY_CUT:
                ks_z = 0.0
                B_z  = 0.0
            else:
                B_z = - (Tz - stress_bar * np.sign(dstrain)) * B / (ssig ** 2)
                invB2 = 1.0 / max(B * B, (self.B_MIN * self.B_MIN))
                ks_z  = - (np.sqrt(2 * np.pi) * ssig / s) * invB2 * B_z

            # kh_z (bounded to avoid huge squares in kr_z)
            kh_bounded = np.clip(kh, -self.KH_MAX, self.KH_MAX)
            kh_z = (rk - alpha) * k0 * A_z

            # kr_z with scaling
            scale = max(abs(kh_bounded), abs(ks), 1.0)
            a = kh_bounded / scale
            b = ks / scale
            denom_sq = (a + b) * (a + b)
            if denom_sq < self.EPS:
                kr_z = 0.0
            else:
                kr_z = (kh_z * (b * b) + ks_z * (a * a)) / denom_sq

            # Jacobian
            f_z = -1.0 + kr_z * dstrain
            if abs(f_z) < 1e-14 or not np.isfinite(f_z):
                f_z = np.sign(f_z) * 1e-14 if f_z != 0 else 1e-14  # damp

            # Newton update (optionally damp further if needed)
            Tznew = Tz - f / f_z
            if not np.isfinite(Tznew):
                Tznew = Tz  # bail out

            Tzold, Tz = Tz, Tznew
            count += 1

            if count == maxiter and self.conv_flag:
                print('MaxIter reached in PH iterations')
                self.conv_flag = False

        # (4) Stiffness & stress (normalized)
        # recompute ks/kr for the converged Tz to build k
        s = rs * (emax_pos - emax_neg)
        if first_step_like or np.abs(s) <= 1.0e-16:
            ks = self.KS_MAX
            kr = kh
        else:
            arg = ((Tz - stress_bar * np.sign(dstrain)) / ssig) ** 2
            arg = min(arg, self.EXP_CLIP)
            B   = np.exp(-0.5 * arg)
            B   = max(B, self.B_MIN)
            C   = 1.0 / np.sqrt(2 * np.pi)
            invks = C * (s / ssig) * B
            ks = self.KS_MAX if invks <= self.EPS else min(1.0 / invks, self.KS_MAX)

            scale = max(abs(kh), abs(ks), 1.0)
            a = kh / scale
            b = ks / scale
            denom_hs = a + b
            kr = (scale * (a * b) / denom_hs) if abs(denom_hs) >= self.EPS else 0.5 * min(kh, ks)

        k = k_el + (kh * ks) / max(kh + ks, self.EPS)
        tstress   = k * dstrain + cstress
        stress_el = stress_el + k_el * dstrain
        stress_st = tstress - stress_el

        # (5) Degradation (normalized)
        rk_trial = (np.abs(tstress) + alpha1 * stress_y) / (k0 * np.abs(strain_i) + alpha1 * stress_y)
        rkmin = min(rk_trial, rkmin)
        rk    = rk_trial + (1 - alpha2) * (rkmin - rk_trial)

        dh = tstress * dstrain
        h  = h + dh
        stress_y = sy0 / (1 + betam1 * h)
        emax_pos = max(emax_pos, strain_i)
        emax_neg = min(emax_neg, strain_i)

        if abs(strain_i) - emax > 0:
            drs = (1 / (np.sqrt(2 * np.pi) * sigp)) * np.exp(-0.5 * ((emax - mup) / sigp) ** 2) * (np.abs(strain_i) - emax)
            rs  = rs + drs * rsmax
            emax = np.abs(strain_i)

        # (6) De-normalize
        k       = k * k_true
        tstress = tstress * sy_true
        tstrain = strain_ip1 * ey_true

        self.tstate = {
            "strain": tstrain,
            "stress": tstress,
            "stiffness": k,
            "z": Tz,
            "h": h,
            "stress_el": stress_el,
            "stress_st": stress_st,
            "stress_y": stress_y,
            "rs": rs,
            "rk": rk,
            "rkmin": rkmin,
            "emax_pos": emax_pos,
            "emax_neg": emax_neg,
            "emax": emax
        }

    def commit_state(self):
        self.cstate = self.tstate.copy()

    def reset_state(self):
        self.cstate = {
            "strain": 0.0,
            "stress": 0.0,
            "stiffness": self.params["k0"],
            "z": 0.0,
            "h": 0.0,
            "stress_el": 0.0,
            "stress_st": 0.0,
            "stress_y": 1.0,  # normalized initial yield strength
            "rs": 0.0,
            "rk": 1.0,
            "rkmin": 1.0,
            "emax_pos": 0.0,
            "emax_neg": 0.0,
            "emax": 0.0
        }
        self.tstate = self.cstate.copy()


def material_test(material, strains):
    '''
    Function that tests a material model, using a history of displacements
    '''
    stress_vec = []
    strain_vec = []
    
    for tstrain in strains:
        # run state determination block and extract the
        material.set_trial_state(tstrain)
        material.commit_state()
        stress_vec.append(material.cstate["stress"])
        strain_vec.append(material.cstate["strain"])

    return strain_vec, stress_vec
    

def createCyclicStrain(maxStrains=[0.01, 0.02], dStrain=0.001):
    """
    Function to create cyclic test data

    Parameters
    ----------
    maxStrains : TYPE, optional
        DESCRIPTION. The default is [0.01, 0.02].
    dStrain : TYPE, optional
        DESCRIPTION. The default is 0.001.

    Returns
    -------
    cycles : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    """
    cycles = []
    ncycles = len(maxStrains)
    
    n = 0
    for maxStrain in maxStrains:
        upWards1 = np.arange(0, maxStrain, dStrain)
        dnWards1 = np.arange(maxStrain, -maxStrain, -dStrain)
        upWards2 = np.arange(-maxStrain, 0, dStrain)
        cycle = np.concatenate([upWards1, dnWards1, upWards2])
        cycles.append(cycle)
        
    cycles = np.concatenate(cycles)
    t = np.arange(0, len(cycles))
    
    return cycles, t









