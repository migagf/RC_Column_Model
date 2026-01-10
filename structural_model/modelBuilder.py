import openseespy.opensees as ops
import opsvis as opsv  # for visualization
import numpy as np


from units import *
from utilities import *


def buildModel(nodes_df, elements_df):
    '''
    Builds a Cantilever Column Model 
    '''
    geomTrans = "linear"
    
    # Loop over nodes_df to create all nodes
    for ii in range(0, len(nodes_df)):
        # Create tag string for node
        nodeTag = int(nodes_df.nodeID[ii])
        ops.node(nodeTag, nodes_df.X[ii], nodes_df.Y[ii], nodes_df.Z[ii])
        
        # Fix base nodes
        if str(nodeTag).endswith("0"):
            ops.fix(nodeTag, *[1, 1, 1, 1, 1, 1])

    # Define material properties for elastic elements (as dictionary)
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
    for ii in range(0, len(elements_df)): # 
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
            ops.element('forceBeamColumn', eleTag, *[node_i, node_j], col_gt, secTag, '-iter', 10, 1e-12, '-mass', dmass)

            
        elif eleType == "pierRigid":
            
            # print("Adding Rigidn Column", eleTag)
            #ops.element('forceBeamColumn', eleTag, *[node_i, node_j], col_gt, 101, '-iter', 10, 1e-12)
            ops.rigidLink("beam", *[node_i, node_j])
            
            
        elif eleType == "westRigid" or eleType == "eastRigid":
            
            # print("Adding Pier Beams", eleTag)
            beamProps, dmass = getBeamProps('rect', beam_props, concrete_props)
            # element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag, <'-mass', mass>, <'-cMass'>)
            ops.element('elasticBeamColumn', eleTag, *[node_i, node_j], *beamProps, beam_gt, '-mass', dmass)
            
            
        elif eleType == "eastGirder" or eleType == "westGirder":
            
            # print("Adding Girders", eleTag)
            beamProps, dmass = getBeamProps('girder', girder_props, concrete_props)
            # element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag, <'-mass', mass>, <'-cMass'>)
            # Model beams with moment releases at both ends
            ops.element('elasticBeamColumn', eleTag, *[node_i, node_j], *beamProps, beam_gt, '-mass', dmass, 'releaseCode', 3)
            
            # Model Beams with no releases
            # ops.element('elasticBeamColumn', eleTag, *[node_i, node_j], *beamProps, beam_gt, '-mass', dmass)
            
    pass