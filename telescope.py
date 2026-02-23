import toml
import numpy as np
from typing import Any, Union

from materials import read_materials


def create_simulation_input(
    # --------------------------------------------------------------------------
    name: str,
    # Geometry and discretisation
    phi: float, R: float, cl: float, b: float, type: Union[int, str], deltaT: float, sym: bool, seeds: int, lump_mass: float,
    # Material and section properties
    materials: list[str], sections,
    part_materials, part_sections,
    # Forces
    forces,
    # Boundary conditons
    bc,
    # Output settings
    nFreq, plot_scale
    # --------------------------------------------------------------------------
    ) -> dict:

    # Initialize output
    input = {}

    input['name'] = name

    # Geometry angles
    phi1 = 0
    phi2 = phi/3/180*np.pi
    phi3 = 2*phi2
    phi4 = phi/180*np.pi
    # Shared x-coordinates
    x1 = -R*np.sin(phi1)
    x2 = -R*np.sin(phi2)
    x3 = -R*np.sin(phi3)
    x4 = -R*np.sin(phi4)
    # Shared y-coordinates
    y1 = cl+R*(1-np.cos(phi1))
    y2 = cl+R*(1-np.cos(phi2))
    y3 = cl+R*(1-np.cos(phi3))
    y4 = cl+R*(1-np.cos(phi4))
    # Angle legs
    theta = np.arctan((b+x3)/y3)
    
    # Create structure for selected type
    # --------------------------------------------------------------------------
    if type:        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Beam structure
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # X coordinates
        x5 = -b
        # Y coordinates
        y5 = 0
        # Assemble points
        input['points'] = np.array([[x1, x2, x3, x4, x5],
                                    [y1, y2, y3, y4, y5]])
        input['parts'] = np.array([[1, 2, 3, 3],
                                   [2, 3, 4, 5]])        
        # Element type
        input['type'] = np.ones((max(input['parts'].shape)), dtype=int)
    else:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Truss structure
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # X coordinates
        x5 = x3 + cl/2*np.cos(np.pi/3+theta)
        x6 = x3 - cl/2*np.cos(np.pi/3-theta)
        x7 = -b
        # Y coordinates
        y5 = y3 - cl/2*np.sin(np.pi/3+theta)
        y6 = y3 - cl/2*np.sin(np.pi/3-theta)
        y7 = 0
        # Assemble points
        input['points'] = np.array([
            [x1, x2, x3, x4, x5, x6, x7],
            [y1, y2, y3, y4, y5, y6, y7]
        ])        
        # Assembly parts
        input['parts'] = np.array([
            [1, 2, 3, 1, 2, 3, 3, 4, 5, 5, 6],
            [2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7]
        ])        
        # Element type
        input['type'] = np.zeros((max(input['parts'].shape)), dtype=int)

    # Initialise seed for every part to 0
    input['seed_array'] = np.zeros((max(input['parts'].shape)))
    # Assign material indeces
    if type:
        assert len(part_materials[0]) == max(input['parts'].shape), \
            f"Length mismatch between the material list and the number of parts" \
            f"-----------------------------------------------------------------" \
            f"The material list specified has length {len(part_materials)}, \n" \
            f"while there are {max(input['parts'].shape)} parts in the structure."
        input['material_indeces'] = np.array(part_materials[0])
        # Assign section indeces
        assert len(part_sections[0]) == max(input['parts'].shape), \
            f"Length mismatch between the material list and the number of parts" \
            f"-----------------------------------------------------------------" \
            f"The material list specified has length {len(part_sections)}, \n" \
            f"while there are {max(input['parts'].shape)} parts in the structure."
        input['section_indeces'] = np.array(part_sections[0])
    else:
        assert len(part_materials[1]) == max(input['parts'].shape), \
            f"Length mismatch between the material list and the number of parts" \
            f"-----------------------------------------------------------------" \
            f"The material list specified has length {len(part_materials)}, \n" \
            f"while there are {max(input['parts'].shape)} parts in the structure."
        input['material_indeces'] = np.array(part_materials[1])
        # Assign section indeces
        assert len(part_sections[1]) == max(input['parts'].shape), \
            f"Length mismatch between the material list and the number of parts" \
            f"-----------------------------------------------------------------" \
            f"The material list specified has length {len(part_sections)}, \n" \
            f"while there are {max(input['parts'].shape)} parts in the structure."
        input['section_indeces'] = np.array(part_sections[1])

    # Create boundary conditions
    # --------------------------------------------------------------------------
    # 1 means d.o.f. is restrained, 0 means d.o.f. is free
    input['bc'] = np.zeros((3*max(input['points'].shape)))
    for condition in bc:        
        input['bc'][(condition['point']-1)*3]   = condition['DOF'][0]
        input['bc'][(condition['point']-1)*3+1] = condition['DOF'][1]
        input['bc'][(condition['point']-1)*3+2] = condition['DOF'][2]
    
    # Create loading vector
    # --------------------------------------------------------------------------
    input['P'] = np.zeros((3*max(input['points'].shape), 1))

    # Create thermal vector
    # --------------------------------------------------------------------------
    input['T'] = np.zeros((max(input['points'].shape), 1))
    if type:
        # First 4 nodes are exposed to deltaT
        input['T'][0:4] = deltaT*np.ones((4, 1))
    else:
        # All nodes are exposed to deltaT
        input['T'] = deltaT*np.ones((max(input['points'].shape)))

    # Create mirror side of structure, if so selected
    # --------------------------------------------------------------------------
    if sym:

        # Connectivity
        a = np.nonzero(input['parts'][0, :] == 1)[0]
        l = max(input['points'].shape)
        # Points (first point should not be duplicated)
        input['points'] = np.vstack((
            np.concatenate((input['points'][0, :], -input['points'][0, 1:]), axis=0).reshape(1, 2*l-1),
            np.concatenate((input['points'][1, :],  input['points'][1, 1:]), axis=0).reshape(1, 2*l-1)
        ))

        # Boundary conditions (first point should not be duplicated)
        input['bc'] = np.concatenate((input['bc'], input['bc'][3:]))
        # Loading vector
        input['P'] = np.concatenate((input['P'], input['P'][3:]))
        # Thermal vector
        input['T'] = np.concatenate((input['T'], input['T'][1:]))
        
        # Define node connections
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Advancing the node numbers to be connected apart for node 1
        c = (l-1)*np.ones((np.shape(input['parts'])), dtype=int)
        c[0, a] = np.zeros((1, len(a)), dtype=int)
        # Symmetric side connections
        input['parts'] = np.concatenate((
               input['parts'],
            (c+input['parts'])[::-1]
        ), axis=1)
        # Double the initialized vectors
        input['type'] = np.concatenate((input['type'], input['type']))
        input['seed_array'] = np.concatenate((input['seed_array'], input['seed_array']))
        input['material_indeces'] = np.concatenate((input['material_indeces'], input['material_indeces']))
        input['section_indeces'] = np.concatenate((input['section_indeces'], input['section_indeces']))
    else:
        # Apply symmetry boundary conditions at point 1
        input['bc'][0] = 1   # 1 means d.o.f. is restrained, 0 means d.o.f. is free
        input['bc'][2] = 1

    # Applied loading
    # --------------------------------------------------------------------------
    for force in forces:
        input['P'][3*(force['point']-1) + force['axis']] = force['magnitude']
    
    # Convert theta to degrees
    input['theta'] = theta*180/np.pi
    
    # Store input data to output
    input['mat']       = materials
    input['sec']       = sections
    input['seeds']     = seeds
    input['lump_mass'] = lump_mass

    # Output settings
    # --------------------------------------------------------------------------
    # Number of eigenfrequencies returned
    input['nFreq'] = nFreq
    # Plot scale
    input['plot_scale'] = plot_scale

    return input


def read_input(input_file: str) -> tuple[str, Any]:
    d = toml.load(input_file)

    # Simulation name
    name = d['name']

    # Geometry
    geometry = d['Geometry']

    # Section properties
    sections = d['Sections']

    # Part properties
    part_properties = d['PartProperties']

    # Applied forces
    forces = d['Forces']

    # Boundary conditions
    bc = d['BoundaryConditions']

    # Output settings
    output_settings = d['Output']

    return name, geometry, sections, part_properties, forces, bc, output_settings

def telescope_geometry(input_file: str, type: int) -> dict[str, Any]:
    
    # Read input from TOML file
    name, geometry, sections, part_properties, forces, bc, output_settings = read_input(input_file)

    # Read materials from input TOML file
    materials_file = 'materials.toml'
    materials      = read_materials(materials_file)
    geometry['type'] = type

    # Call the function that creates the geometry
    input = create_simulation_input(
        name = name,
        # Geometry, discretisation
        **geometry,
        # Material and section properties
        materials      = materials,
        sections       = sections,
        part_materials = part_properties['materials'],
        part_sections  = part_properties['sections'],
        forces         = forces,
        # Boundary condition application
        bc             = bc,
        # Output settings
        **output_settings
        )
    return input

