import openseespy.opensees as ops
import matplotlib.pyplot as plt
import opsvis as opsv  # for visualization
import numpy as np

from units import *


def getSections(plot=True):
    '''
    Builds a Reinforced Concrete Fiber Section
    '''
    # :::
    # (1) Define materials
    # :::
    
    # (1.1) Steel Material 
    # uniaxialMaterial('Steel02', matTag, Fy, E0, b)
    
    E0 = 29_000.0*ksi
    Fy = 55.5*ksi
    b = 0.01
    
    # ops.uniaxialMaterial('Steel02', 1, Fy, E0, b)
    
    # uniaxialMaterial('SteelMPF', matTag, fyp, fyn, E0, bp, bn, *params, a1=0.0, a2=1.0, a3=0.0, a4=1.0)
    ops.uniaxialMaterial('SteelMPF', 1, Fy, Fy, E0, b, b, *[20, 0.925, 0.15])
    
    # (1.2) Concrete Materials
    # uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets)
    
    fpc = -5.00 * ksi
    fpcu = fpc * 0.1
    epsc0 = - 0.002
    epsU = epsc0 * 10
    lam = 0.2
    f_t = - fpc / 30
    Ets = 2 * fpc / (epsc0 * 20)
    
    ops.uniaxialMaterial('Concrete02', 2, fpc, epsc0, fpcu, epsU, lam, f_t, Ets)    # Confined
    ops.uniaxialMaterial('Concrete02', 3, fpc, epsc0, fpcu, -0.003, lam, f_t, Ets)  # Unconfined
    
    
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
        ops.section('Fiber', secTag, '-GJ', GJ)
        ops.patch(*fib_sec_1[1][1:])
        ops.patch(*fib_sec_1[2][1:])
        ops.layer(*fib_sec_1[3][1:])
        ops.layer(*fib_sec_1[4][1:])

        if plot:
            matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
            opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
            plt.axis('equal')
            plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # ops.beamIntegration('lobatto', tag, secTag, N)
        ops.beamIntegration('Lobatto', secTag, secTag, 10)
        
        
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
        ops.section('Fiber', secTag, '-GJ', GJ)
        ops.patch(*fib_sec_1[1][1:])
        ops.patch(*fib_sec_1[2][1:])
        ops.layer(*fib_sec_1[3][1:])
        ops.layer(*fib_sec_1[4][1:])

        if plot:
            matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
            opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
            plt.axis('equal')
            plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # ops.beamIntegration('lobatto', tag, secTag, N)
        ops.beamIntegration('Lobatto', secTag, secTag, 10)
    
    
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
        ops.section('Fiber', secTag, '-GJ', GJ)
        ops.patch(*fib_sec_1[1][1:])
        ops.patch(*fib_sec_1[2][1:])
        ops.layer(*fib_sec_1[3][1:])
        ops.layer(*fib_sec_1[4][1:])

        if plot:
            matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
            opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
            plt.axis('equal')
            plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # ops.beamIntegration('lobatto', tag, secTag, N)
        ops.beamIntegration('Lobatto', secTag, secTag, 10)
    
    
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
        ops.section('Fiber', secTag, '-GJ', GJ)
        ops.patch(*fib_sec_1[1][1:])
        ops.patch(*fib_sec_1[2][1:])
        ops.layer(*fib_sec_1[3][1:])
        
        if n_in_bars != 0: # Just add inner reinforcement layer if necessary
            ops.layer(*fib_sec_1[4][1:])

        if plot:
            matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']
            opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
            plt.axis('equal')
            plt.savefig('section_'+str(secTag)+'.pdf')
        
        # Integrator for the Fiber Section (intTag is same as secTag)
        # ops.beamIntegration('lobatto', tag, secTag, N)
        ops.beamIntegration('Lobatto', secTag, secTag, 10)
    
    # Now, get elastic sections
    # (2.5) Beam Sections
    
    
    
    
    
    # :::
    # Geometric Transformations
    # :::
    
    # For vertical elements:
    ops.geomTransf('Linear', 10, *[0, 1, 0])
    ops.geomTransf('PDelta', 20, *[0, 1, 0])
    ops.geomTransf('Corotational', 30, *[0, 1, 0])
    
    # For horizontal elements
    ops.geomTransf('Linear', 11, *[0, 0, 1])
    ops.geomTransf('PDelta', 21, *[0, 0, 1])
    ops.geomTransf('Corotational', 31, *[0, 0, 1])
    
    
    pass


