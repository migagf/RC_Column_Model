import pandas as pd
from units import *
import numpy as np
import matplotlib.pyplot as plt

# Add nodes to structural model

def add_nodes(model, nodes_df):
    """Add nodes to the model.

    Args:
        model (xara.Model): The structural model.
        nodes_df (pandas.DataFrame): DataFrame containing node information with columns 'NodeID', 'x', 'y', 'z'.
    """

    # Loop over nodes_df to create all nodes
    for ii in range(0, len(nodes_df)):
        # Create tag string for node
        nodeTag = int(nodes_df.nodeID[ii])

        # Additional mass to go in the nodes
        massval = 0.0
        # Fix base nodes
        if str(nodeTag).endswith("0"):
            mass = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            model.node(nodeTag, (nodes_df.X[ii], nodes_df.Y[ii], nodes_df.Z[ii]), mass=mass)
            model.fix(nodeTag, *[1, 1, 1, 1, 1, 1])

            print("Fixing Node:", nodeTag)
        else:
            mass = (massval, massval, massval, massval/100, massval/100, massval/100)
            model.node(nodeTag, (nodes_df.X[ii], nodes_df.Y[ii], nodes_df.Z[ii]), mass=mass)
            
    pass


# Add uniaxial materials to structural model

def add_materials(model):

    # (1.1) Steel Material 
    # uniaxialMaterial('Steel02', matTag, Fy, E0, b)
    E0 = 29_000.0*ksi
    Fy = 55.5*ksi
    b = 0.01
    
    model.uniaxialMaterial('Steel02', 1, Fy, E0, b)
    
    # uniaxialMaterial('SteelMPF', matTag, fyp, fyn, E0, bp, bn, *params, a1=0.0, a2=1.0, a3=0.0, a4=1.0)
    # model.uniaxialMaterial('SteelMPF', 1, Fy, Fy, E0, b, b, *[20, 0.925, 0.15])
    
    
    # (1.2) Concrete Materials
    # uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets)
    fpc = -5.00 * ksi
    fpcu = fpc * 0.1
    epsc0 = - 0.002
    epsU = epsc0 * 10
    lam = 0.2
    f_t = - fpc / 30
    Ets = 2 * fpc / (epsc0 * 20)
    
    model.uniaxialMaterial('Concrete02', 2, fpc, epsc0, fpcu, epsU, lam, f_t, Ets)    # Confined
    model.uniaxialMaterial('Concrete02', 3, fpc, epsc0, fpcu, -0.003, lam, f_t, Ets)  # Unconfined
    
    pass


def add_sections(model, plot=False):
    '''
    Builds a Reinforced Concrete Fiber Section
    '''
    
    # ::: 
    # (2) Define geometry
    # :::
    hc_vals = [25, 30, 35, 40, 45]
    
    # (2.1) Sections 5A
    secTags = [101, 102, 103, 104, 105]
    
    for ii in range(0, len(hc_vals)):
        
        # Get Section Tag from array
        secTag = secTags[ii]
        hc = hc_vals[ii]
        
        # Diameter
        diameter = 5.0*ft
        
        GJ = 10_000_000  # Need to compute this value
        
        # Bars in outer and inner reinf. rings
        if hc == 25:
            n_out_bars = 28
            n_in_bars = 36 - n_out_bars
        elif hc == 30:
            n_out_bars = 28
            n_in_bars = 44 - n_out_bars
        elif hc == 35:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 40:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 45:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        else:
            print("Could not find hc value, adopting hc=25")
            n_out_bars = 28
            n_in_bars = 36 - n_out_bars

        d_bar = 2.257*inch
        a_bars = np.pi * (d_bar / 2) ** 2
        c_cover = 3.0*inch
        cp_cover = 2.5*inch
        
        # Discretization for concrete section
        nfibRad_Core = 4
        nfibRad_Cov = 2
        nfibPhi = 20

        # ---
        # Create Section
        # ---
        
        # Just for printing
        fib_sec_1 = [
            ['section', 'Fiber', secTag],
            ['patch', 'circ', 3, nfibPhi, nfibRad_Cov, *[0.0, 0.0], *[diameter/2 - c_cover, diameter/2], *[0.0, 360.0]], # Unconfined concrete
            ['patch', 'circ', 2, nfibPhi, nfibRad_Core, *[0.0, 0.0], *[0.0, diameter/2 - c_cover], *[0.0, 360.0]],  # Confined concrete
            ['layer', 'circ', 1, n_out_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - d_bar/2, *[0.0, 360.0 - 360 / n_out_bars]],  # Rebar, outer layer
            ['layer', 'circ', 1, n_in_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - cp_cover - d_bar/2, *[0.0, 360.0 - 360 / n_in_bars]]
        ]
        
        # Create Sections
        model.section('Fiber', secTag, '-GJ', GJ)
        model.patch(*fib_sec_1[1][1:])
        model.patch(*fib_sec_1[2][1:])
        model.layer(*fib_sec_1[3][1:])
        model.layer(*fib_sec_1[4][1:])

        # if plot:
            #matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
            #opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
            #plt.axis('equal')
            #plt.savefig('section_'+str(secTag)+'.pdf')
            # veux.render(sec1)

        # Integrator for the Fiber Section (intTag is same as secTag)
        # model.beamIntegration('lobatto', tag, secTag, N)
        model.beamIntegration('Lobatto', secTag, secTag, 10)
        
        
    # (2.2) Sections 5B
    secTags = [111, 112, 113, 114, 115]
    
    for ii in range(0, len(hc_vals)):
        
        # Get Section Tag from array
        secTag = secTags[ii]
        hc = hc_vals[ii]
        
        # Diameter
        diameter = 5.0*ft
        
        GJ = 10_000_000  # Need to compute this value
        
        # Bars in outer and inner reinf. rings
        if hc == 25:
            n_out_bars = 28
            n_in_bars = 40 - n_out_bars
        elif hc == 30:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 35:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 40:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 45:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        else:
            print("Could not find hc value, adopting hc=25")
            n_out_bars = 28
            n_in_bars = 40 - n_out_bars
            

        d_bar = 2.257*inch
        a_bars = np.pi * (d_bar / 2) ** 2
        c_cover = 3.0*inch
        cp_cover = 2.5*inch
        
        
        # Discretization for concrete section
        nfibRad_Core = 4
        nfibRad_Cov = 2
        nfibPhi = 20

        # ---
        # Create Section
        # ---
        
        # Just for printing
        fib_sec_1 = [
            ['section', 'Fiber', secTag],
            ['patch', 'circ', 3, nfibPhi, nfibRad_Cov, *[0.0, 0.0], *[diameter/2 - c_cover, diameter/2], *[0.0, 360.0]], # Unconfined concrete
            ['patch', 'circ', 2, nfibPhi, nfibRad_Core, *[0.0, 0.0], *[0.0, diameter/2 - c_cover], *[0.0, 360.0]],  # Confined concrete
            ['layer', 'circ', 1, n_out_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - d_bar/2, *[0.0, 360.0 - 360 / n_out_bars]],  # Rebar, outer layer
            ['layer', 'circ', 1, n_in_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - cp_cover - d_bar/2, *[0.0, 360.0 - 360 / n_in_bars]]
        ]
        
        # Create Sections
        model.section('Fiber', secTag, '-GJ', GJ)
        model.patch(*fib_sec_1[1][1:])
        model.patch(*fib_sec_1[2][1:])
        model.layer(*fib_sec_1[3][1:])
        model.layer(*fib_sec_1[4][1:])

        #if plot:
        #    matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
        #    opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
        #    plt.axis('equal')
        #    plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # model.beamIntegration('lobatto', tag, secTag, N)
        model.beamIntegration('Lobatto', secTag, secTag, 10)
    
    
    # (2.3) Sections 5C
    secTags = [121, 122, 123, 124, 125]
    
    for ii in range(0, len(hc_vals)):
        
        # Get Section Tag from array
        secTag = secTags[ii]
        hc = hc_vals[ii]
        
        # Diameter
        diameter = 5.0*ft
        
        GJ = 10_000_000  # Need to compute this value
        
        # Bars in outer and inner reinf. rings
        if hc == 25:
            n_out_bars = 28
            n_in_bars = 44 - n_out_bars
        elif hc == 30:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 35:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 40:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        elif hc == 45:
            n_out_bars = 28
            n_in_bars = 48 - n_out_bars
        else:
            print("Could not find hc value, adopting hc=25")
            n_out_bars = 28
            n_in_bars = 44 - n_out_bars
            

        d_bar = 2.257*inch
        a_bars = np.pi * (d_bar / 2) ** 2
        c_cover = 3.0*inch
        cp_cover = 2.5*inch
        
        
        # Discretization for concrete section
        nfibRad_Core = 4
        nfibRad_Cov = 2
        nfibPhi = 20

        # ---
        # Create Section
        # ---
        
        # Just for printing
        fib_sec_1 = [
            ['section', 'Fiber', secTag],
            ['patch', 'circ', 3, nfibPhi, nfibRad_Cov, *[0.0, 0.0], *[diameter/2 - c_cover, diameter/2], *[0.0, 360.0]], # Unconfined concrete
            ['patch', 'circ', 2, nfibPhi, nfibRad_Core, *[0.0, 0.0], *[0.0, diameter/2 - c_cover], *[0.0, 360.0]],  # Confined concrete
            ['layer', 'circ', 1, n_out_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - d_bar/2, *[0.0, 360.0 - 360 / n_out_bars]],  # Rebar, outer layer
            ['layer', 'circ', 1, n_in_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - cp_cover - d_bar/2, *[0.0, 360.0 - 360 / n_in_bars]]
        ]
        
        # Create Sections
        model.section('Fiber', secTag, '-GJ', GJ)
        model.patch(*fib_sec_1[1][1:])
        model.patch(*fib_sec_1[2][1:])
        model.layer(*fib_sec_1[3][1:])
        model.layer(*fib_sec_1[4][1:])

        # if plot:
        #    matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
        #    opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
        #    plt.axis('equal')
        #    plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # model.beamIntegration('lobatto', tag, secTag, N)
        model.beamIntegration('Lobatto', secTag, secTag, 4)
    
    
    # (2.4) Sections 6C
    secTags = [131, 132, 133, 134, 135]
    
    for ii in range(0, len(hc_vals)):
        
        # Get Section Tag from array
        secTag = secTags[ii]
        hc = hc_vals[ii]
        
        # Diameter
        diameter = 6.0*ft
        
        GJ = 10_000_000  # Need to compute this value
        
        # Bars in outer and inner reinf. rings
        if hc == 25:
            n_out_bars = 36
            n_in_bars = 36 - n_out_bars
        elif hc == 30:
            n_out_bars = 36
            n_in_bars = 40 - n_out_bars
        elif hc == 35:
            n_out_bars = 36
            n_in_bars = 48 - n_out_bars
        elif hc == 40:
            n_out_bars = 36
            n_in_bars = 56 - n_out_bars
        elif hc == 45:
            n_out_bars = 36
            n_in_bars = 64 - n_out_bars
        else:
            print("Could not find hc value, adopting hc=25")
            n_out_bars = 36
            n_in_bars = 36 - n_out_bars
            

        d_bar = 2.257*inch
        a_bars = np.pi * (d_bar / 2) ** 2
        c_cover = 3.0*inch
        cp_cover = 2.5*inch
        
        
        # Discretization for concrete section
        nfibRad_Core = 4
        nfibRad_Cov = 2
        nfibPhi = 20

        # ---
        # Create Section
        # ---
        
        # Just for printing
        if n_in_bars != 0:
            fib_sec_1 = [
                ['section', 'Fiber', secTag],
                ['patch', 'circ', 3, nfibPhi, nfibRad_Cov, *[0.0, 0.0], *[diameter/2 - c_cover, diameter/2], *[0.0, 360.0]], # Unconfined concrete
                ['patch', 'circ', 2, nfibPhi, nfibRad_Core, *[0.0, 0.0], *[0.0, diameter/2 - c_cover], *[0.0, 360.0]],  # Confined concrete
                ['layer', 'circ', 1, n_out_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - d_bar/2, *[0.0, 360.0 - 360 / n_out_bars]],  # Rebar, outer layer
                ['layer', 'circ', 1, n_in_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - cp_cover - d_bar/2, *[0.0, 360.0 - 360 / n_in_bars]]
            ]
        else:
            fib_sec_1 = [
                ['section', 'Fiber', secTag],
                ['patch', 'circ', 3, nfibPhi, nfibRad_Cov, *[0.0, 0.0], *[diameter/2 - c_cover, diameter/2], *[0.0, 360.0]], # Unconfined concrete
                ['patch', 'circ', 2, nfibPhi, nfibRad_Core, *[0.0, 0.0], *[0.0, diameter/2 - c_cover], *[0.0, 360.0]],  # Confined concrete
                ['layer', 'circ', 1, n_out_bars, a_bars, *[0.0, 0.0], diameter/2 - c_cover - d_bar/2, *[0.0, 360.0 - 360 / n_out_bars]],  # Rebar, outer layer
            ]
        
        # Create Sections
        model.section('Fiber', secTag, '-GJ', GJ)
        model.patch(*fib_sec_1[1][1:], section=secTag)
        model.patch(*fib_sec_1[2][1:], section=secTag)
        model.layer(*fib_sec_1[3][1:], section=secTag)
        
        if n_in_bars != 0: # Just add inner reinforcement layer if necessary
            model.layer(*fib_sec_1[4][1:], section=secTag)

        #print('Adding integration rule for secTag', secTag, secTag)
        model.beamIntegration('Lobatto', secTag+1000, secTag, 4)

        #if plot:
        #    matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
        #    opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
        #    plt.axis('equal')
        #    plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # model.beamIntegration('lobatto', tag, secTag, N)
        
    
    # Now, get elastic sections
    # (2.5) Beam Sections
    
    
    
    # :::
    # Geometric Transformations
    # :::
    
    # For vertical elements:
    model.geomTransf('Linear', 10, *[0, 1, 0])
    model.geomTransf('PDelta', 20, *[0, 1, 0])
    model.geomTransf('Corotational', 30, *[0, 1, 0])
    
    # For horizontal elements
    model.geomTransf('Linear', 11, *[0, 0, 1])
    model.geomTransf('PDelta', 21, *[0, 0, 1])
    model.geomTransf('Corotational', 31, *[0, 0, 1])
  
    pass


def build_model(model, elements_df):
    '''
    Adds elements to the structural model
    '''

    geomTrans = "linear"
    
    # Concrete properties (for weight computations)
    concrete_props = {
        "E": 29_000*ksi,  # Young's modulus
        "nu": 0.3,        # Poissom ratio
        "rho": 150*pcf/g
    }

    # Simplified girder geometry
    girder_props = {
        "width": 5.0*ft,
        "height": 4.0*ft,
        "thickness": 8.0*inch,
    }
    
    beam_props = {
        "height": 5.0*ft,
        "width": 5.0*ft,
    }
    
    # Define geometric transformation index
    if geomTrans == 'linear':
        col_gt = 10
        beam_gt = 11
    elif geomTrans == 'pdelta':
        col_gt = 20
        beam_gt = 21
    elif geomTrans == 'corotational':
        col_gt = 30
        beam_gt = 31
    

    # Define Elements    
    for ii in range(0, len(elements_df)): # Looping over elements dataframe
        # Get element tag, and nodes i-j
        eleTag = int(elements_df.eleTag[ii])
        node_i = int(elements_df.nodei[ii])
        node_j = int(elements_df.nodej[ii])
        eleType = elements_df.type[ii]
        
        if eleType == "pierCol":
            # If element corresponds to a pier, get the Type and HC values
            pierType = elements_df.ptype[ii]
            pierHC = elements_df.hc[ii]
            
            # print('Adding pier', ii, '\n', pierType)
            
            # Get section tag
            secTag = getSecTag(pierType, pierHC)
            
            # Some properties
            diam = float(pierType[0])
            area = np.pi * (diam / 2) ** 2
            dmass = area * concrete_props["rho"]
            
            # element('forceBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-iter', maxIter=10, tol=1e-12, '-mass', mass=0.0)
            #print('seems like there might be a problem here')
            #print('Section Tag:', secTag)
            #print('Transform Tag:', col_gt)
            #print('Integration Tag:', 131)

            model.element('ForceFrame', eleTag, (node_i, node_j), 4, secTag, col_gt, mass=dmass)
            #print('Done')
            # Here, change the col_gt to the integration tag.
            
        elif eleType == "pierRigid":
            
            # print("Adding Rigidn Column", eleTag)
            # model.element('forceBeamColumn', eleTag, *[node_i, node_j], col_gt, 101, '-iter', 10, 1e-12)
            model.element('ForceFrame', eleTag, (node_i, node_j), 4, 101, col_gt, mass=0.0)  # mass=0 to avoid doble-counting.
            # model.rigidLink("beam", node_i, node_j)
            # beamProps, dmass = getBeamProps('rect', beam_props, concrete_props)
            # model.element('PrismFrame', eleTag, (node_i, node_j), *beamProps, col_gt, mass=0.0)

            
        elif eleType == "westRigid" or eleType == "eastRigid":
            
            # print("Adding Pier Beams", eleTag)
            beamProps, dmass = getBeamProps('rect', beam_props, concrete_props)
            # element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag, <'-mass', mass>, <'-cMass'>)
            model.element('PrismFrame', eleTag, (node_i, node_j), *beamProps, beam_gt, mass=dmass)
            
            
        elif eleType == "eastGirder" or eleType == "westGirder":
            
            # print("Adding Girders", eleTag)
            beamProps, dmass = getBeamProps('girder', girder_props, concrete_props)
            # element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag, <'-mass', mass>, <'-cMass'>)
            # Model beams with moment releases at both ends
            model.element('PrismFrame', eleTag, (node_i, node_j), *beamProps, beam_gt, mass=dmass, releasey=3)
            
            # Model Beams with no releases
            # model.element('elasticBeamColumn', eleTag, *[node_i, node_j], *beamProps, beam_gt, '-mass', dmass)
            

    pass


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

