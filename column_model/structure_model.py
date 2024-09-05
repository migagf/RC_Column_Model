import numpy as np
import matplotlib.pyplot as plt

g = 386

# :::
# Structural Model Class
# :::

class structure_model():
    '''
    3dof structural model with plastic hinge at the bottom
    
    '''
    
    def __init__(self, el_params, ph_model, mass, damping):
        self.el_params = {
            "EI": el_params[0],
            "L": el_params[1]
        }
        
        self.ph = ph_model
        self.ks1 = []
        self.ks = []
        self.bmatrix = []
        self.kt = []
        
        self.mass_params = {
            "m_1": mass[0],    # Lateral mass of the bridge
            "i_1": mass[1],    # Rotational inertia at top
            "i_2": mass[2],    # Rotational inertia at bottom
        }

        self.damping_params = {
            "zeta": damping[0]  # Critical damping ratio for Rayleigh damping
        }
        
        self.initialize()

    
    def get_b_matrix(self):
        L = self.el_params["L"]
        B = np.array(
            [[1/L, 1/L, 0],
             [1, 0, 1],
             [0, 1, 0]]
        )
        self.bmatrix = B
        
    
    def get_ks1_matrix(self):
        EI = self.el_params["EI"]
        L = self.el_params["L"]
        ks1 = EI / L * np.array(
            [[4, 2],[2, 4]]
        )
        self.ks1 = ks1

    
    def get_ks_matrix(self):
        ks1 = self.ks1
        ks = np.zeros((3, 3))
        ks[:2, :2] = ks1
        ks[-1, -1] = self.ph.cstate["stiffness"]  # get stiffness of ph
        self.ks = ks

    
    def get_k_matrix(self):
        self.kt = self.bmatrix @ self.ks @ self.bmatrix.T

    
    def set_trial_state(self, u_trial):
        '''
        Set trial state for the entire structural model
        u_trial is a vector with the 3dof trial displacement
        '''
        b = self.bmatrix
        ks = self.ks
        ks1 = self.ks1
        
        v_trial = b.T @ u_trial
        q_1 = ks1 @ v_trial[0:2]

        # Set state for PH
        self.ph.set_trial_state(v_trial[-1])
        q_2 = self.ph.tstate["stress"]
        k_2 = self.ph.tstate["stiffness"]
        
        q_trial = np.concatenate((q_1, [q_2]))
        ks[-1, -1] = k_2
        k_t = b @ ks @ b.T
        self.kt = k_t
        
        pr_trial = self.bmatrix @ q_trial
        
        self.tstate = {
            "stress": q_trial,
            "stiffness": k_t,
            "pr": pr_trial,
            "un": u_trial,
            "vn": np.array([0.0, 0.0, 0.0]),
            "an": np.array([0.0, 0.0, 0.0])
        }

    def commit_state(self):
        self.cstate = self.tstate
        self.ph.commit_state()

    def assemble_mass(self):
        m_1 = self.mass_params["m_1"]
        i_1 = self.mass_params["i_1"]
        i_2 = self.mass_params["i_2"]

        self.mass = np.diagflat([m_1, i_1, i_2])
    
    def rayleigh_damping(self):
        # Run eigenvalue to get natural frequencies
        self.eig(False)
        wi = 2 * np.pi * self.eigenfreqs[0]
        wj = 2 * np.pi * self.eigenfreqs[1]
        zetai = self.damping_params["zeta"]
        zetaj = self.damping_params["zeta"] * 2  # Twice as much damping on second mode

        a = np.array([[1 / wi, wi], [1 / wj, wj]])
        b = np.array([zetai, zetaj])

        x = np.linalg.solve(a, 2 * b)
        self.damping = x[1] * self.cstate["stiffness"] * g + x[0] * self.mass   # force/time

        
    def eig(self, show=True, plot=False):
        """
        Runs an eigenvalue analysis on the train
        """
        global g
        from numpy.linalg import eig, inv
        if len(self.cstate["stiffness"]) == 0 or len(self.mass) == 0:
            print("ERROR - Mass or Stiffness Matrix have not been defined... Run assemble() commands")
        else:
            print("Running Eigenvalue Analysis ... ")
            w, v = eig(inv(self.mass) @ self.cstate["stiffness"] * g)
            self.eigenfreqs = np.sqrt(w) / (2 * np.pi)
            self.modeshapes = v

            if show:
                for i in range(0, 3):
                    print("::: \n Mode #", i + 1)
                    print("freq = {:6.4f} Hz ".format(float(self.eigenfreqs[i])))
                    print("T = {:6.4f} sec ".format(float(1/self.eigenfreqs[i])))
                    ithmodeshape = v[:, i]
                    for j in range(0, 3):
                        print("{:6.3f}".format(float(ithmodeshape[j])))
            print("Eigenvalue Analysis Done \n")
        
        if plot:
            pass
    
    def initialize(self):
        self.get_ks1_matrix()
        self.get_b_matrix()
        self.get_ks_matrix()
        self.get_k_matrix()
        
        self.cstate = {
            "stress": np.array([0.0, 0.0, 0.0]),
            "stiffness": self.kt,
            "pr": np.array([0.0, 0.0, 0.0]),
            "un": np.array([0.0, 0.0, 0.0]),
            "vn": np.array([0.0, 0.0, 0.0]),
            "an": np.array([0.0, 0.0, 0.0])
        }
        
        self.tstate = {
            "stress": np.array([0.0, 0.0, 0.0]),
            "stiffness": self.kt,
            "pr": np.array([0.0, 0.0, 0.0]),
            "un": np.array([0.0, 0.0, 0.0]),
            "vn": np.array([0.0, 0.0, 0.0]),
            "an": np.array([0.0, 0.0, 0.0])
        }

        self.mass = np.zeros((3, 3))
        self.damping = np.zeros((3, 3))

        # Variables for eigenvalue analysis
        self.eigenfreqs = []
        self.modeshapes = []

        # Assemble stiffness, mass and damping matrices
        self.assemble_mass()
        self.rayleigh_damping()

        
# :::
# Function for pushover analysis
# :::

def load_step(pf, u_trial, model):
    
    cont, maxiter = 0.0, 100
    
    model.set_trial_state(u_trial)
    pu = pf - model.tstate["pr"]
    err = np.linalg.norm(pu)
    
    while err > 1.0e-12 and cont < maxiter:
        du_trial = np.linalg.solve(model.tstate["stiffness"], pu)
        u_trial = u_trial + du_trial
        
        model.set_trial_state(u_trial)
        pu = pf - model.tstate["pr"]
        err = np.linalg.norm(pu)
        
        cont += 1
        if cont == maxiter:
            print('Warning... MaxIter Reached in Global Equilibrium')
        
    # model.commit_state()
    
    return u_trial


# :::
# Class for TH analysis
# :::

class TH_Solver:
    def __init__(self, dt=0.01, maxiter=100, beta=0.25, gamma=0.5):
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
        p_eff = (pext + model.mass / g @ (self.constants[1] * vn + self.constants[3] * an) +
                 model.damping / g @ (self.constants[4] * vn + self.constants[5] * an))  # force

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

            # Get effective stiffness
            kt_i = model.tstate["stiffness"] + (self.constants[0] * model.mass + self.constants[2] * model.damping) / g 

            # Get updated restoring forces
            pr_i = model.tstate["pr"]
            pu_i = p_eff - pr_i - (self.constants[0] * model.mass + self.constants[2] * model.damping) / g @ Du_i
            du_i = np.linalg.solve(kt_i, pu_i)
            Du_i += du_i
            ui = un + Du_i

            err = np.linalg.norm(pu_i)
            
            #print('Eq. Iter No. {1:5.0f}, Error = {0:5.5e}'.format(err, niter))
            
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
        

def run_pushover(model, disp, plot=False, show_info=False):
    '''
    Take the structure defined in model and run a pushover analysis
    '''
    
    # Reference load
    
    Pref = np.array([1, 0, 0])
    
    lf = []
    
    for u_target in disp:
        
        # Get last commited displacement state
        ut = model.cstate["un"]
        x = np.array([0.1, ut[1], ut[2]])
        
        err = 1.0
        
        iters = 0
        
        while err > 1.0e-6 and iters < 20:
            
            # Define u from state determination
            u_trial = np.array([u_target, x[1], x[2]])
            
            # Set trial state and get resisting forces
            model.set_trial_state(u_trial)
            pr = model.tstate['pr']
            kt = model.tstate['stiffness']
            
            Pu = x[0] * Pref - pr
            
            dPu = np.array([[1, 0, 0],[0, -kt[1,1], -kt[1,2]],[0, -kt[2,1], -kt[2,2]]])
            
            x =  x - np.linalg.inv(dPu) @ Pu
            
            err = np.linalg.norm(Pu)
            
            if show_info:
                print('Eq. iter=', iters, ' - Error=', err)
            iters += 1
            
        lf.append(x[0])
        #print(model.ph.tstate["rs"])

        model.commit_state()
        ut = model.cstate["un"]

        if show_info:
            print(ut, u_target)
    
    if plot:
        plt.figure(dpi = 300)
        plt.plot(disp, lf, 'k.-')
        plt.xlabel('Displacement (in)')
        plt.ylabel('Load Factor')
    
    return lf
















