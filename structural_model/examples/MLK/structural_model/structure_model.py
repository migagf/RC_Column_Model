import numpy as np
from scipy.linalg import block_diag
from .material_models import UniaxialMaterial, DegradingBoucWenMod
from .elements import ZeroLength1D, ElasticFrameElement
import pandas as pd

# ::: Structural Model Class (Modified to include nonlinear geometry (P-Delta)) :::
class InelasticFrame():
    '''
    2D frame model with concentrated plasticity
    The model is a frame with ncol columns
        x--------x--------x --> Ulat    Top level
        o        o        o             Plastic hinges at each column
        |        |        | 
        |        |        |             Elastic beam elements
        |        |        |
        o        o        o             Plastic hinges at each column
        #--------#--------#             Ground level

    '''
    def __init__(self, ncolumns, E, I, h, hingedata, massdata, damping, weight, geometry="p-delta", units="metric"):
        # Store the model parameters in a dictionary
        self.parameters = {
            "ncolumns": ncolumns,
            "E": E,
            "I": I,
            "h": h,
            "hingedata": hingedata,  # [ph1, ph2, ...]
            "massdata": massdata,    # [mlat, J1, J2, ...]
            "damping": damping,      # [zeta]
            "weight": weight,        # weight for p-delta effects
            "geometry": geometry
            }
        
        self.units = units  # 'metric' or 'imperial'

        # Define number of degrees of freedom and number of element basic forces
        if self.parameters["ncolumns"] == 1:
            self.ndof = 3            # 1 lateral + 2 rotational
            self.nq = 3               # 1 ph and 2 element basic forces
            self.nfe = 1              # 1 frame element per column
            self.nph = 1              # 1 plastic hinge per column
        else:
            self.ndof = 2 * self.parameters["ncolumns"] + 1       # 2 rotations per columns + 1 lateral
            self.nq = 4 * self.parameters["ncolumns"]             # 2 phs per column + 2 element basic forces per column
            self.nfe = self.parameters["ncolumns"]                # 1 frame element per column
            self.nph = 2 * self.parameters["ncolumns"]            # 2 plastic hinges per column

        self.nelements = self.nfe + self.nph
        
        # Empty list to store the elements
        self.elements = []
        
        self.initialize()


    def get_Af_Bf(self):
        """
        Get the kinematic matrix Af
        """
        if self.parameters["ncolumns"] > 1:
            # Start with a zero matrix with ndof columns and nq rows
            Af = np.zeros((self.nq, self.ndof))

            # Fill up the column elastic deformations
            for i in range(4 * self.parameters["ncolumns"]):
                if i <= 2 * self.parameters["ncolumns"] - 1:
                    Af[i, 0] = 1 / self.parameters["h"]  # Lateral deformation
                    Af[i, i + 1] = 1
                else:
                    Af[i, i - 2 * self.parameters["ncolumns"] + 1] = 1
        else:
            # For a single column, the kinematic matrix is hard-coded
            Af = np.array([[1 / self.parameters["h"], 1, 0],
                           [1 / self.parameters["h"], 0, 1],
                           [0, 1, 0]]
                           )
        
        # Set the B matrix to be the transpose of Af
        Bf = Af.T
        self.Bf = Bf
        self.Af = Af


    def create_elements(self):
        """
        An element is a dictionary with the following keys:
            - 'element_type': 'frame'
            - 'ks': element stiffness matrix
            - 'elementdata': different element class depending on the type of element. If frame, leave empty.
        """
        elem_id = 0
        # Create frame elements
        for i in range(self.nfe):
            element = ElasticFrameElement(elem_id, self.parameters["E"], self.parameters["I"], self.parameters["h"])
            self.elements.append(element)
            elem_id += 1

        # Create plastic hinge elements
        for i in range(self.nph):
            element = ZeroLength1D(elem_id, self.parameters["hingedata"][i])
            self.elements.append(element)
            elem_id += 1


    def get_stiffness(self):
        # Loop through the elements and get the stiffness matrices and concatenate in a block diagonal matrix
        ks_list = []
        for element in self.elements:
            ks_list.append(element.get_stiffness())

        Ks = block_diag(*ks_list)  # Create a block diagonal matrix from the list of stiffness matrices
        # Check if the Af matrix is defined
        if hasattr(self, 'Af'):
            Kt = self.Af.T @ Ks @ self.Af
        else:
            # Call the method to get Af and Bf matrices
            self.get_Af_Bf()
            Kt = self.Af.T @ Ks @ self.Af

        # If using p-delta geometry, substract pv * k_g
        if self.parameters["geometry"] == "p-delta":
            pv = self.parameters["weight"]  # Axial force scalar
            # Check if k_g is defined, if not, calculate it
            if not hasattr(self, 'k_g'):
                self.get_kg()
            Kt  = Kt - pv * self.k_g
        
        return Ks, Kt
    

    def get_kg(self):
        # This is a ndof x ndof matrix with 1 / h in the first row and column
        # and zeros elsewhere. It is used to scale the p-delta effect.
        self.k_g = np.zeros((self.ndof, self.ndof))
        self.k_g[0, 0] = 1 / self.parameters["h"]  # Only the first degree of freedom is affected by the p-delta effect

    
    def set_trial_state(self, u_trial):
        '''
        Set trial state for the entire structural model
        u_trial is a vector with the 3dof trial displacement
        '''

        v_trial = self.Af @ u_trial

        # For each element, set the trial basic forces 
        # Start with the frame elements
        for i in range(self.nfe):
            element_vtrial = v_trial[i*2:(i+1)*2]
            self.elements[i].set_trial_state(element_vtrial)

        # Now, set the trial state for the plastic hinges
        for i in range(self.nph):
            element_vtrial = v_trial[self.nfe*2 + i]
            self.elements[self.nfe + i].set_trial_state(element_vtrial)
        
        # Now, compile the trial basic forces into a q_trial vector
        q_trial = []
        for i in range(self.nelements):
            try:
                # If iterable
                for val in self.elements[i].tstate["qtrial"]:
                    q_trial.append(val)
            except:
                # If not iterable, just append the value
                q_trial.append(self.elements[i].tstate["qtrial"])

        q_trial = np.array(q_trial)

        # Now, compute the stiffness matrix
        Ks, Kt = self.get_stiffness()

        # Update stiffness
        self.Kt = Kt

        if hasattr(self, 'k_g'):
            pv = self.parameters["weight"]  # Axial force scalar (check units)
            pr_trial = self.Bf @ q_trial - pv * self.k_g @ u_trial
        else:
            pr_trial = self.Bf @ q_trial
        
        self.tstate = {
            "stress": q_trial,
            "stiffness": Kt,
            "pr": pr_trial,
            "un": u_trial,
            "vn": np.zeros(self.ndof),  # Initial velocities
            "an": np.zeros(self.ndof),  # Initial accelerations
        }


    def commit_state(self):
        self.cstate = self.tstate
        # Commit the state for each element
        for element in self.elements:
            element.commit_state()

    def resetState(self):
        """
        Reset the structural model to initial state (zero displacements, velocities, accelerations)
        This should be called before each new time-history analysis to ensure clean initial conditions.
        """
        # Reset current state to zero
        self.cstate = {
            "stress": np.array([0.0] * self.nq),
            "stiffness": self.Kt,
            "pr": np.array([0.0] * self.ndof),
            "un": np.array([0.0] * self.ndof),
            "vn": np.array([0.0] * self.ndof),
            "an": np.array([0.0] * self.ndof)
        }
        
        # Reset trial state to zero
        self.tstate = {
            "stress": np.array([0.0] * self.nq),
            "stiffness": self.Kt,
            "pr": np.array([0.0] * self.ndof),
            "un": np.array([0.0] * self.ndof),
            "vn": np.array([0.0] * self.ndof),
            "an": np.array([0.0] * self.ndof)
        }
        
        # Reset each element to initial state
        for element in self.elements:
            if hasattr(element, 'resetState'):
                element.resetState()
            elif hasattr(element, 'reset_state'):
                element.reset_state()

    def get_mass(self):
        self.mass = np.diagflat(self.parameters["massdata"])  # Mass matrix is diagonal with the mass data


    def get_damping(self):
        # Call the Rayleigh damping method to compute the damping matrix
        self.damping_params = {"zeta": self.parameters["damping"][0], 
                               "type": "rayleigh"}
        
        self.rayleigh_damping()


    def rayleigh_damping(self):

        # Run eigenvalue to get natural frequencies
        self.eig(show=False)
        '''
        wi = 2 * np.pi * self.eigenfreqs[0]
        wj = 2 * np.pi * self.eigenfreqs[1]
        zetai = self.damping_params["zeta"]
        
        zetaj = self.damping_params["zeta"]  # Same damping on second mode

        a = np.array([[1 / wi, wi], [1 / wj, wj]])
        b = np.array([zetai, zetaj])

        x = np.linalg.solve(a, b)
        self.damping = x[1] * self.cstate["stiffness"] + x[0] * self.mass  # force/time
        '''

        # Just stiffness proportional, to test
        wi = 2 * np.pi * min(self.eigenfreqs)
        zetai = self.damping_params["zeta"]
        x = 2 * zetai / wi
        self.damping = x * self.cstate["stiffness"]  # force/time

        
    def eig(self, show=True, plot=False):
        """
        Runs an eigenvalue analysis on the train
        """
        g = self.g  # gravitational constant
        from numpy.linalg import eig, inv
        if len(self.cstate["stiffness"]) == 0 or len(self.mass) == 0:
            print("ERROR - Mass or Stiffness Matrix have not been defined... Run assemble() commands")
        else:
            # print("Running Eigenvalue Analysis ... ")
            # print(self.mass)
            # print(self.Kt)
            w, v = eig(inv(self.mass) @ self.Kt)
            self.eigenfreqs = np.sqrt(w) / (2 * np.pi)
            self.modeshapes = v

            if show:
                # Print modes as a table
                print("Mode | Frequency (Hz) | Period (sec) | Mode Shape")
                print("-" * 60)
                for i in range(min(3, len(self.eigenfreqs))):
                    freq = float(self.eigenfreqs[i])
                    period = float(1 / self.eigenfreqs[i])
                    ithmodeshape = v[:, i]
                    modeshape_str = ", ".join("{:6.3f}".format(float(val)) for val in ithmodeshape)
                    print(f"{i+1:4d} | {freq:13.4f} | {period:11.4f} | {modeshape_str}")
        
        if plot:
            pass
    

    def initialize(self):
        # Set gravitational constant
        if self.units == 'imperial':
            self.g = 386.4  # gravitational constant in in/s^2
        elif self.units == 'metric':
            self.g = 9.81  # gravitational constant in m/s^2
        else:
            print('Assuming metric units')
            self.g = 9.81  # gravitational constant in m/s^2

        self.get_Af_Bf()
        self.create_elements()
        
        Ks, Kt = self.get_stiffness()
        self.Ks = Ks
        self.Kt = Kt

        self.cstate = {
            "stress": np.array([0.0] * self.nq),
            "stiffness": self.Kt,
            "pr": np.array([0.0] * self.ndof),
            "un": np.array([0.0] * self.ndof),
            "vn": np.array([0.0] * self.ndof),
            "an": np.array([0.0] * self.ndof)
        }
        
        self.tstate = {
            "stress": np.array([0.0] * self.nq),
            "stiffness": self.Kt,
            "pr": np.array([0.0] * self.ndof),
            "un": np.array([0.0] * self.ndof),
            "vn": np.array([0.0] * self.ndof),
            "an": np.array([0.0] * self.ndof)
        }

        self.mass = np.zeros((self.ndof, self.ndof))
        self.damping = np.zeros((self.ndof, self.ndof))

        # Variables for eigenvalue analysis
        self.eigenfreqs = []
        self.modeshapes = []

        # Assemble stiffness, mass and damping matrices
        self.get_mass()
        self.get_damping()

    def get_base_shear(self):
        """
        Get the base shear from the restoring forces
        """
        base_shear = self.cstate["pr"][0]
        return base_shear


# ::: Function for pushover analysis :::
def load_step(pf, u_trial, model):
    """
    Here, pf is the applied external forces. How to determine this external force vector is a problem...
    Need to implement displacement control (control the pf vector to achieve a target displacement)
    """
    cont, maxiter = 0.0, 1000
    
    model.set_trial_state(u_trial)
    pu = pf - model.tstate["pr"]
    err = np.linalg.norm(pu)
    
    while err > 1.0e-8 and cont < maxiter:
        du_trial = np.linalg.solve(model.tstate["stiffness"], pu)
        u_trial = u_trial + du_trial
        
        model.set_trial_state(u_trial)
        pu = pf - model.tstate["pr"]
        err = np.linalg.norm(pu)
        
        cont += 1
        if cont == maxiter:
            print('Warning... MaxIter Reached in Global Equilibrium')
            
    model.commit_state()
    
    return u_trial


class DispContPushoverAnalysis:
    def __init__(self, controlnode, pref, targetdisp=None, maxiter=1000, tol=1.0e-8):
        '''
        Initialize the displacement controlled pushover analysis
        controlnode: Node where the displacement is controlled
        targetdisp: Target displacements to be achieved (in a list)
        maxiter: Maximum number of iterations for convergence
        tol: Tolerance for convergence
        '''
        self.controlnode = controlnode
        self.targetdisp = targetdisp
        self.maxiter = maxiter
        self.tol = tol
        self.pref = pref   # Reference external force vector (as numpy array)
        self.lambdas = []  # This is to store the scale factors

    def __str__(self):
        return f"DispContPushoverAnalysis(controlnode={self.controlnode}, targetdisp={self.targetdisp}, maxiter={self.maxiter}, tol={self.tol}, pref={self.pref}, lambdas={self.lambdas})"

    def run(self, model):
        """
        Run the displacement controlled pushover analysis
        model is an instance of InelasticFrame
        """
        # Check that targetdisp has been set
        if self.targetdisp is None:
            raise ValueError("Target displacements must be set.")

        u_trial = np.zeros(model.ndof)
        lam = 0.0  # Initial scale factor for the control node displacement
        
        for target_disp in self.targetdisp:
            cont = 0
            err = 1.0

            # Initialize the unknown vector x
            x_i = u_trial.copy()
            x_i[self.controlnode] = lam  # Initialize the scale factor x[controlnode] = lambda
            
            # Set the trial state with the current displacement
            u_trial[self.controlnode] = target_disp

            while err > self.tol and cont < self.maxiter:
                model.set_trial_state(u_trial)

                # Get the external force vector at the control node
                g = self.pref * lam - model.tstate["pr"]

                # Compute the x_ip1
                grad_g = - model.tstate["stiffness"]
                grad_g[:, self.controlnode] = self.pref  # Gradient in the load factor direction

                x_i += - np.linalg.inv(grad_g) @ g

                # Update the trial state with the new displacement
                u_trial = x_i.copy()
                u_trial[self.controlnode] = target_disp  # Ensure the control node displacement is maintained
                lam = x_i[self.controlnode]  # Update the scale factor
                err = np.linalg.norm(g)
                cont += 1
                # print(f"Iteration {cont}: Current displacement = {u_trial[self.controlnode]}, Target displacement = {target_disp}, Lambda = {lam:.4f}, Error = {err:.4f}")

            model.commit_state()

            print(f"Target displacement {target_disp} achieved with lambda = {lam:.4f} after {cont} iterations.")
            self.lambdas.append(lam)  # Store the scale factor for this target displacement

    def set_target_displacements(self, maxStrains=[0.01, 0.02], dStrain=0.001):
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

        # If dStrain is bigger than any of the values of maxStrains, use the smaller of maxStrains as dStrain
        if dStrain > min(maxStrains):
            dStrain = min(maxStrains)

        cycles = []
        
        for maxStrain in maxStrains:
            upWards1 = np.arange(0, maxStrain, dStrain)
            dnWards1 = np.arange(maxStrain, -maxStrain, -dStrain)
            upWards2 = np.arange(-maxStrain, 0, dStrain)
            cycle = np.concatenate([upWards1, dnWards1, upWards2])
            cycles.append(cycle)
            
        cycles = np.concatenate(cycles)

        self.targetdisp = cycles.tolist()  # Convert to list for consistency
        

    def plot_pushover_curve(self, filename):
        """
        Plot the pushover curve using matplotlib
        """
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Left subplot: Displacement path (lambda vs step)
        axs[0].plot(range(1, len(self.lambdas) + 1), self.targetdisp, marker='o', linestyle='-', color='g')
        axs[0].set_title('Displacement Path')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Target Displacement (in)')
        axs[0].grid()       

        # Right subplot: Pushover curve (target displacement vs lambda)
        axs[1].plot(self.targetdisp, self.lambdas, marker='o', linestyle='-', color='b')
        axs[1].set_title('Pushover Curve')
        axs[1].set_xlabel('Target Displacement (in)')
        axs[1].set_ylabel('Load Factor Lambda')
        axs[1].grid()

        '''# Mark the peak strength point
        peak_index = np.argmax(self.lambdas)
        axs[1].plot(self.targetdisp[peak_index], self.lambdas[peak_index], marker='o', color='r')
        axs[1].annotate(f'Peak Strength\n({self.targetdisp[peak_index]:.2f}, {self.lambdas[peak_index]:.2f})',
                        xy=(self.targetdisp[peak_index], self.lambdas[peak_index]),
                        xytext=(self.targetdisp[peak_index] + 0.1, self.lambdas[peak_index] + 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        '''
        # Set y axis limits symmetric and max is closest 100 multiple
        max_lambda = max(abs(min(self.lambdas)), abs(max(self.lambdas)))
        max_limit = np.ceil(max_lambda / 100) * 100
    
        axs[1].set_ylim(-max_limit, max_limit)
        axs[1].set_yticks(np.arange(-max_limit, max_limit + 1, max_limit / 5))
        if filename is not None:
            plt.savefig(filename)
        
        plt.close(fig)

        pass

    def save_results(self, filename):
        # save the targetdisp and the lambdas to a csv file
        
        df = pd.DataFrame({
            'Target Displacement': self.targetdisp,
            'Load Factor Lambda': self.lambdas
        })
        df.to_csv(filename, index=False)

        pass

# ::: Class for TH analysis :::
class TH_Solver:
    def __init__(self, dt=0.01, maxiter=1000, beta=0.25, gamma=0.5):
        self.dt = dt
        self.maxiter = maxiter
        self.beta = beta
        self.gamma = gamma
        self.constants = np.zeros(6)
        self.set_constants()
        self.tol = 1.0e-6


    def set_constants(self):
        self.constants[0] = 1 / (self.beta * self.dt ** 2)
        self.constants[1] = 1 / (self.beta * self.dt)             
        self.constants[2] = self.gamma / (self.beta * self.dt)
        self.constants[3] = 1 / (2 * self.beta) - 1.0
        self.constants[4] = self.gamma / self.beta - 1.0
        self.constants[5] = self.dt * (self.gamma / (2 * self.beta) - 1.0)

    def step_increment(self, model, pext):

        # Load values from previous step (commited state)
        un = model.cstate["un"]
        vn = model.cstate["vn"]
        an = model.cstate["an"]

        # From state at n, Un, Vn, An, determine the effective applied force vector
        p_eff = (pext + model.mass @ (self.constants[1] * vn + self.constants[3] * an) +
                 model.damping @ (self.constants[4] * vn + self.constants[5] * an))  # force

        # OPTIMIZATION: Pre-compute matrix terms to avoid recomputation in convergence loop
        coeff_mass = self.constants[0]
        coeff_damp = self.constants[2]
        mass_damp_term = (coeff_mass * model.mass + coeff_damp * model.damping)

        # Determine Pr0 = Pr(Un) and tangent stiffness
        pr_0 = model.cstate["pr"]  # force
        kt_0 = model.cstate["stiffness"]   # force/length

        # Determine Pu_n+1 = Peff - Pr0
        pu_n = p_eff - pr_0
        du_i = np.linalg.solve(kt_0, pu_n)   # length

        Du_i = du_i

        err = 1.0
        niter = 1

        ui = un + Du_i
        
        while err > self.tol and niter < self.maxiter:
            # Run state determination of the structure
            model.set_trial_state(ui)

            # Get effective stiffness (using pre-computed matrix term)
            kt_i = model.tstate["stiffness"] + mass_damp_term

            # Get updated restoring forces (using pre-computed matrix term)
            pr_i = model.tstate["pr"]
            pu_i = p_eff - pr_i - mass_damp_term @ Du_i
            du_i = np.linalg.solve(kt_i, pu_i)
            Du_i += du_i
            ui = un + Du_i

            err = np.linalg.norm(pu_i)
            
            # print('Eq. Iter No. {1:5.0f}, Error = {0:5.5e}'.format(err, niter))
            
            niter += 1
            if niter == self.maxiter:
                print("Warning: max number of iterations reached in global equilibrium")

        # Upon convergence, compute velocities and accelerations
        vi = self.constants[2] * (ui - un) - self.constants[4] * vn - self.constants[5] * an
        ai = self.constants[0] * (ui - un) - self.constants[1] * vn - self.constants[3] * an
        
        model.tstate["vn"] = vi
        model.tstate["an"] = ai
        
        # Commit state
        model.commit_state()