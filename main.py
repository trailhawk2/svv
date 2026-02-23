# -*- coding: utf-8 -*-


import numpy as np
from typing import Any
from pprint import pprint

from objects.assembly            import Assembly
from objects.mesh                import Mesh
from objects.element             import Element

from telescope import telescope_geometry
from validation import validation_geometry


def MechRes2D(inp: dict[str, Any]) -> dict[str, Any]:
    # =============================================================================
    #  AE3212-II Structures assignment 2021-2023
    # =============================================================================
    # ============================Start of preamble================================
    # This Python-function belongs to the AE3212-II Simulation, Verification and
    # Validation Structures assignment. The file contains a function that
    # calculates the mechanical response of a 2D structure that has been
    # specified by the user.
    #
    # The function uses a structure as input and returns a structure as output.
    #
    # Proper functioning of the code can be checked with the following input:
    #
    # Written by Antonio López Rivera and Václav Marek,
    # based on the original by Julien van Campen
    # Aerospace Structures and Computational Mechanics
    # TU Delft
    # October 2021 â€“ January 2022 & January 2023
    # ==============================End of Preamble================================
    
    # --------------------------------------------------------------------------
    # 0. User Input
    # --------------------------------------------------------------------------
    assembly = Assembly(inp)
    # --------------------------------------------------------------------------
    # 1. Input Check
    # --------------------------------------------------------------------------
    if PLOT:
        assembly.plot_input()
    
    # --------------------------------------------------------------------------
    # 2. Creation of Parts - Plot for check by user
    # --------------------------------------------------------------------------    
    # Create mesh
    assembly.mesh = Mesh(assembly)
        
    if PLOT:
        # Plot mesh
        assembly.mesh.plot_mesh()
        # Show the user the mesh overlaid over the geometry
        assembly.mesh.plot_mesh(False)
        assembly.plot_input()

        # Pause for user to do visual inspection of structure and mesh
        input('Please inspect the mesh shown in figures 1 and 2 and 3. \
               They may also be saved in the working path as "mesh_1.png", "mesh_2.png", and "mesh_3.png", if so configured. \
               Press enter to continue.')
    
    # --------------------------------------------------------------------------
    # 3. Creation of Elements
    # --------------------------------------------------------------------------    
    # Assign properties
    assembly.mesh.assign_element_properties()

    # Creation of Elements
    for ix in range(assembly.mesh.mesh['nElements']):
        # for each element, generate local properties
        element = Element(assembly, ix)
        # rewrite the local mesh with the mesh that now has the element properties included
        assembly.mesh = element.assembly.mesh

    # --------------------------------------------------------------------------
    # 4. Assembling the structure
    # --------------------------------------------------------------------------
    assembly.global_mass_matrix()
    assembly.global_stiffness_matrix()
    assembly.global_thermal_load_vector()
    
    # --------------------------------------------------------------------------
    # 5. Application of Loads and BCs
    # --------------------------------------------------------------------------
    assembly.loads_bc()
    
    # --------------------------------------------------------------------------
    # 6. Performing the analysis
    # TODO: Calculate stresses and strains
    # --------------------------------------------------------------------------
    o = assembly.output

    # Test to visualize dictionary and size of the different matrices

    #print('This is the dictionary')
    #print(o)

    # End of Test
    
    # Displacements and reaction forces
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    KrInv = np.linalg.inv(o['Kr'])
    Ur = KrInv@(o['Pr']-o['Qr'])
    # Displacements of entire system
    o['U'][o['activeDF']] = Ur
    # Reaction forces
    Rs = o['Ksr']@Ur + (o['Ps']-o['Qs'])
    # Reaction forces of entire system
    o['R'][o['inactiveDF']] = Rs

    # Eigenfrequency analysis
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Compute eigenvalues
    eigenfreqs = np.linalg.eig(-KrInv@o['mr'])[0]
    # Process first user-defined number of eigenvalues
    eigenfreqs = np.sqrt(np.abs(eigenfreqs[:inp['nFreq']]**(1)))
    # Return eigenfrequencies to assembly.output.
    o['eigenfrequencies'] = eigenfreqs

    def get_transformation_matrix(transformation_matrix, element_type):
        # element type 'rod' or 'beam',
        # and the coordinates of u and v are obtained from o('U'),
        #  but the index changes for rods and beams

        if element_type == 0:
            T = np.zeros((4, 4))
            tl = transformation_matrix[0:2, 0:2]
            br = transformation_matrix[3:5, 3:5]
            T[0:2, 0:2] = tl
            T[2:4, 2:4] = br
        elif element_type == 1:
            T = transformation_matrix

        return T

    element_strains = []
    element_stresses =  []
    
    o6 = {
        'stress': None,
        'strain': None
    }

    # Code to fill in to calculate the stress and strain
    for ix in range(assembly.mesh.mesh['nElements']):
        e = assembly.mesh.mesh['element'][ix]
        length = e['length']
        element_type = e['type']
        alpha = e['alpha']
        dT = e['DeltaT']
        T = get_transformation_matrix(e['T'], element_type)

        nodes = np.array([e['nodeNumber1'], e['nodeNumber2']])
        nodes -= np.array([1,1])

        if element_type == 0:
            DOFs = np.array([2*nodes[0], 2*nodes[0]+1, 2*nodes[1], 2*nodes[1]+1])
        elif element_type == 1:
            DOFs = np.array([3*nodes[0], 3*nodes[0]+1, 3*nodes[0]+2, 3*nodes[1], 3*nodes[1]+1, 3*nodes[1]+2])
        U_global = o['U'][DOFs]
        U_local = T @ U_global.reshape(-1,1)

        strain = 0
        E = e['E']

        if element_type == 0:
            strain = (U_local[2] - U_local[0])/length

        elif element_type == 1:
            strain = (U_local[3] - U_local[0])/length


        strain = strain.item()
        stress = E * (strain - alpha * dT)

        element_strains.append(strain)
        element_stresses.append(stress)

    o6['strain'] = element_strains
    o6['stress'] = element_stresses

        # get u2 and u1
        # strain = (u2-u1)/length
        # stress = e['K'] * strain

    # Create sub-dictionary to save stress and strain. This will be added to the main assembly output dictionary (o) late.

    # element_stress = [] matrix to store elementwise stresses
    # element_strain = [] matrix to store

    # o['U'] = global displacement matrix

    # T = Rotation Matrix (We need to calculate this)

    # Defining the rotation matrix T for an axial rod:
    # Rotation matrix will take different values of theta between each adjacent node

    # General Formation for s and c

    # s is the "sin(theta)" term and c is the "cos(theta)" term
    # Theta needs to be found for every rod element
    # Defining a function that finds s and c for every adjacent node

    # Indexing of U_global when the dish is modeled as a rod (change [gemonetry] -> type to 0 for beam in telescope.toml)
    # For the rod, there is a truss structure. The connections between the nodes are defined in telescope.py, line 83

    # Indexing of U_global when the dish is modeled as a beam (change [gemonetry] -> type to 1 for beam in telescope.toml)
    # For the beam, there is a beam structure. The connections between the nodes are defined in telescope.py, line 59

    # The assembly of the U column vector (nX1 array in the assembly.output dictionary) is created in assembly.py line 147
    # This constructs U by calling a class from mesh.py (?)

    # o['U_local'] = local displacement matrix = T @ o['U']

    # element length matrix = el['length']

    # rod: u_local = [u1, v1, u2, v2]
    #   axial_strain =  (u_local[2] - u_local[0])/element_length
    #   axial_stress = el['E'] * strain
    #   bending_stress = [0, 0, ...., 0]

    # beam: u_local = [u1, v1, theta1, u2, v2, theta2]
    #   axial_strain =  (u_local[3] - u_local[0])/element_length
    #   axial_stress = el['E'] * strain

    # o6['stress'] = axial_stress
    # o6['strain'] = axial_strain

    # Saving the sub-dictionary back to the main dictionary

    o['B6'] = o6

    #print('This is the strain')
    #print(o['B6']['strain'])

    # Write any changes in the output alias back to the assembly's output
    assembly.output = o
    
    # --------------------------------------------------------------------------
    # 7. Plot results
    # TODO: improve plots to ease comparison, etc.
    # --------------------------------------------------------------------------
    assembly.plot_output()

    # --------------------------------------------------------------------------
    # 8. Save simulation results
    # TODO: fill in inputs for data printing
    # --------------------------------------------------------------------------
    data_print = False  ### True => plot saved data
    data_name = 'SpaceTelescope_test1'  ### object name to be retrieved (same as the filename)
    assembly.save_output(data_print, data_name)
    
    return assembly.output


def ConstGeom(inp : dict[str, Any]) -> dict[str, Any]:
    # =============================================================================
    # This script constructs the geometry of the parts, as input for MechRes

    # angles dish
    phi1 = 0
    phi2 = inp['phi']/3/180*np.pi
    phi3 = 2*phi2
    phi4 = inp['phi']/180*np.pi
    # shared x-coordinates
    x1 = -inp['R']*np.sin(phi1)
    x2 = -inp['R']*np.sin(phi2)
    x3 = -inp['R']*np.sin(phi3)
    x4 = -inp['R']*np.sin(phi4)
    # shared y-coordinates
    y1 = inp['cl']+inp['R']*(1-np.cos(phi1))
    y2 = inp['cl']+inp['R']*(1-np.cos(phi2))
    y3 = inp['cl']+inp['R']*(1-np.cos(phi3))
    y4 = inp['cl']+inp['R']*(1-np.cos(phi4))
    # angle legs
    theta = np.arctan((inp['b']+x3)/y3)
    # create structure for selected type
    out = {}  # initialize output
    if inp['type']:
        # ---- beam structure ----
        # x-coordinates
        x5 = -inp['b']
        # y-coordinates
        y5 = 0
        # assemble points
        out['points'] = np.array([[x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5]])
        out['parts'] = np.array([[1, 2, 3, 3], [2, 3, 4, 5]])
        # element type
        out['type'] = np.ones((max(out['parts'].shape)), dtype=int)
    else:
        # ---- rod structure ----
        # x-coordinates
        x5 = x3 + inp['cl']/2*np.cos(np.pi/3+theta)
        x6 = x3 - inp['cl']/2*np.cos(np.pi/3-theta)
        x7 = -inp['b']
        # y-coordinates
        y5 = y3 - inp['cl']/2*np.sin(np.pi/3+theta)
        y6 = y3 - inp['cl']/2*np.sin(np.pi/3-theta)
        y7 = 0
        # assemble points
        out['points'] = np.array(
            [[x1, x2, x3, x4, x5, x6, x7], [y1, y2, y3, y4, y5, y6, y7]])

        out['parts'] = np.array([[1, 2, 3, 1, 2, 3, 3, 4, 5, 5, 6], [
                                2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7]])
        # element type
        out['type'] = np.zeros((max(out['parts'].shape)), dtype=int)

    # initialise seed for every part to 0
    out['seed_array'] = np.zeros((max(out['parts'].shape)))
    
    # initialise material for every part to 0
    out['material_indeces'] = np.zeros((max(out['parts'].shape)), dtype=int)
    # initialise section for every part to 0
    out['section_indeces'] = np.zeros((max(out['parts'].shape)), dtype=int)

    # create boundary conditions
    # 1 means d.o.f. is restrained, 0 means d.o.f. is free
    out['bc'] = np.zeros((3*max(out['points'].shape)))
    # final point is clamped
    out['bc'][-3:] = 1

    # create loading vector
    out['P'] = np.zeros((3*max(out['points'].shape), 1))

    # create thermal vector
    out['T'] = np.zeros((max(out['points'].shape), 1))
    if inp['type']:
        # first 4 nodes are exposed to deltaT
        out['T'][0:4] = inp['deltaT']*np.ones((4, 1))
    else:
        # all nodes are exposed to deltaT
        out['T'] = inp['deltaT']*np.ones((max(out['points'].shape)))

    # create mirror side of structure, if so selected
    if inp['sym']:
        # connectivity
        a = np.nonzero(out['parts'][0, :] == 1)[0]
        l = max(out['points'].shape)
        # points (first point should not be duplicated)
        out['points'] = np.vstack((np.concatenate((out['points'][0, :], -out['points'][0, 1:]), axis=0).reshape(
            1, 2*l-1), np.concatenate((out['points'][1, :], out['points'][1, 1:]), axis=0).reshape(1, 2*l-1)))
        # boundary conditions (first point should not be duplicated)
        out['bc'] = np.concatenate((out['bc'], out['bc'][3:]))
        # loading vector
        out['P'] = np.concatenate((out['P'], out['P'][3:]))
        # thermal vector
        out['T'] = np.concatenate((out['T'], out['T'][1:]))
        # advancing the node numbers to be connected apart for node 1
        c = (l-1)*np.ones((np.shape(out['parts'])), dtype=int)
        c[0, a] = np.zeros((1, len(a)), dtype=int)
        out['parts'] = np.concatenate((out['parts'], c+out['parts']), axis=1)
        # double the initialized vectors
        out['type'] = np.concatenate((out['type'], out['type']))
        out['seed_array'] = np.concatenate((out['seed_array'], out['seed_array']))
        out['material_indeces'] = np.concatenate((out['material_indeces'], out['material_indeces']))
        out['section_indeces'] = np.concatenate((out['section_indeces'], out['section_indeces']))
    else:
        # apply symmetry boundary conditions at point 1
        out['bc'][0] = 1 # 1 means d.o.f. is restrained, 0 means d.o.f. is free
        out['bc'][2] = 1

    # convert theta to degrees
    out['theta'] = theta*180/np.pi
    # store input data to output
    out['inp'] = inp
    out['mat'] = inp['mat']
    out['sec'] = inp['sec']
    out['seeds'] = inp['seeds']
    out['lump_mass'] = inp['lump_mass']
    out['name'] = inp['name']
    out['plot_scale'] = inp['plot_scale']
    return out


if __name__ == '__main__':

    # Set to True to show input validation plots
    PLOT = True
    
    # Create input geometry from TOML file
    # inp = telescope_geometry('telescope.toml')
    i=int(input("1 for telescope, 0 for validation... "))
    type=int(input("1 for beam, 0 for truss... "))
    if i:
        inp = telescope_geometry('telescope.toml', type)
    else:
        inp= validation_geometry('validation.toml',type)

    # Run simulation
    output = MechRes2D(inp)

