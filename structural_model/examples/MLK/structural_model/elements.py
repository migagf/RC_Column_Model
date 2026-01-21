import numpy as np

class ZeroLength1D():
    def __init__(self, id, uniaxial_material):
        "uniaxial_material: An instance of a uniaxial material class, see material_models.py"
        self.id = id  # This is a unique identifier for the element
        self.material = uniaxial_material
        self.element_type = "zero_length_1d"
        
        self.ks = np.array([uniaxial_material.cstate["stiffness"]])   # Uniaxial material stiffness as a 1D array

        self.tstate = {"qtrial": np.array([0.0]),  # Basic forces in the element
                       "vtrial": np.array([0.0])
                       }
        
        self.cstate = {"q": np.array([0.0]),   # Basic forces in the element
                       "v": np.array([0.0])    # Element deformations
                       }

    def set_trial_state(self, vtrial):
        # Here, call the uniaxial_material's set_trial_state method
        self.material.set_trial_state(vtrial)
        self.tstate["vtrial"] = vtrial
        self.tstate["qtrial"] = self.material.tstate["stress"]
    
    def commit_state(self):
        # Commit the state of the uniaxial material
        self.material.commit_state()
        self.cstate["v"] = self.tstate["vtrial"]
        self.cstate["q"] = self.tstate["qtrial"]

    def get_stiffness(self):
        return self.ks
    
    def __str__(self):
        return f"ZeroLength1D(ks={self.ks}, tstate={self.tstate}, cstate={self.cstate})"
    

class ElasticFrameElement():
    def __init__(self, id, E, I, L):
        self.element_type = "elastic_frame_element"
        self.E = E  # Young's modulus
        self.I = I  # Moment of inertia
        self.L = L  # Length of the element

        self.id = id
        self.ks = E * I / L * np.array([[4, 2],
                                        [2, 4]])
        self.tstate = {"qtrial": np.array([0.0, 0.0]),  # Basic forces in the element
                       "vtrial": np.array([0.0, 0.0])   # Element deformations
                       }
        
        self.cstate = {"q": np.array([0.0, 0.0]),  # Basic forces in the element
                       "v": np.array([0.0, 0.0])   # Element deformations
                       }

    def set_trial_state(self, vtrial):
        self.tstate["vtrial"] = vtrial
        self.tstate["qtrial"] = self.ks @ self.tstate["vtrial"]

    def commit_state(self):
        self.cstate["v"] = self.tstate["vtrial"]
        self.cstate["q"] = self.tstate["qtrial"]

    def get_stiffness(self):
        return self.ks
    
    def __str__(self):
        return f"ElasticFrameElement(ks={self.ks}, tstate={self.tstate}, cstate={self.cstate})"