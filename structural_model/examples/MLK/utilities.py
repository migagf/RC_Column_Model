# Utilities for model generation

def getBeamProps(beam_type, beam_geometry, mat_props):
    """
    geom_props and mat_props are dictionaries
    """
    
    if beam_type == 'rect':
        # Calculations for rectangular beam
        h = beam_geometry["height"]
        b = beam_geometry["width"]
        
        Area = b * h
        Jxx = 1/12 * (b ** 3 * h + b * h ** 3)
        Iy = 1/12 * (b * h ** 3)
        Iz = 1/12 * (h * b ** 3)
        
    elif beam_type == 'girder':
        # Properties are calculated based on a simplified version (two rectangles)
        h_out = beam_geometry["height"]
        b_out = beam_geometry["width"]
        t = beam_geometry["thickness"]
        
        h_in = h_out - 2 * t
        b_in = b_out - 2 * t
        
        Area = h_out * b_out - h_in * b_in
        Jxx = 1/12 * (b_out ** 3 * h_out + b_out * h_out ** 3) - 1/12 * (b_in ** 3 * h_in + b_in * h_in ** 3)
        Iy = 1/12 * (b_out * h_out ** 3) - 1/12 * (b_in * h_in ** 3)
        Iz = 1/12 * (h_out * b_out ** 3) - 1/12 * (h_in * b_in ** 3)
    
    E_mod = mat_props["E"]
    G_mod = E_mod / (2 * (1 + mat_props["nu"]))
    
    # Create list with properties
    props = [Area, E_mod, G_mod, Jxx, Iy, Iz]
    
    # Get mass per unit length of element
    rho = mat_props["rho"]  # mass density
    dmass = Area * rho
    
    return props, dmass


def getSecTag(pierType, hc):
    # Get section Tag from the section properties
    
    if pierType == "5A":
        typeVal = 0
    elif pierType == "5B":
        typeVal = 1
    elif pierType == "5C":
        typeVal = 2
    elif pierType == "6C":
        typeVal = 3
    
    if hc == 25:
        hcVal = 1
    elif hc == 30:
        hcVal = 2
    elif hc == 35:
        hcVal = 3
    elif hc == 40:
        hcVal = 4
    elif hc == 45:
        hcVal = 5
        
    secTag = 100 + typeVal * 10 + hcVal * 1
    
    return secTag