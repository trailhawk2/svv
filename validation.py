import toml
import numpy as np
from typing import Any, Union

from materials import read_materials


def create_simulation_input(
    # --------------------------------------------------------------------------
    name: str,
    # Geometry and discretisation
    L_beam: float, d: float, L_truss: float, alpha_truss: float, type: Union[int, str], deltaT: float, sym: bool, seeds: int, lump_mass: float,
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

    # Create structure for selected type
    # --------------------------------------------------------------------------
    if type:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Beam structure
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # X,Y coordinates
        x1 = 0
        x2 = -L_beam / 2
        x3 = L_beam / 2
        y = 0
        # Assemble points
        input['points'] = np.array([[x1, x2, x3],
                                    [y, y, y]])
        input['parts'] = np.array([[1, 1],[2, 3]])
        # Element type
        input['type'] = np.ones((max(input['parts'].shape)), dtype=int)
    else:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Truss structure
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # X coordinates
        x1 = 0
        x2 = -L_truss/2
        x3 = 0
        x4 = L_truss/2
        # Y coordinates
        y1 = 0
        y2 = 0
        y3 = np.tan(alpha_truss*np.pi/180)*L_truss/2
        y4 = 0

        # Assemble points tan = v/h
        input['points'] = np.array([
            [x1, x2, x3, x4],
            [y1, y2, y3, y4]
        ])
        # Assembly parts
        input['parts'] = np.array([
            [1, 1, 1, 2, 3],
            [2, 3, 4, 3, 4]
        ])
        # Element type
        input['type'] = np.zeros((max(input['parts'].shape)), dtype=int)

    # Initialise seed for every part to 0
    if type:
        input['seed_array'] = np.zeros((max(input['parts'].shape)))
    else:
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
        assert len(part_sections[0]) == min(input['parts'].shape), \
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
    input['bc'] = np.zeros((3 * max(input['points'].shape)))
    for condition in bc:
        input['bc'][(condition['point'] - 1) * 3] = condition['DOF'][0]
        input['bc'][(condition['point'] - 1) * 3 + 1] = condition['DOF'][1]
        input['bc'][(condition['point'] - 1) * 3 + 2] = condition['DOF'][2]

    # Create loading vector
    # --------------------------------------------------------------------------
    input['P'] = np.zeros((3 * max(input['points'].shape), 1))

    # Create thermal vector
    # --------------------------------------------------------------------------
    input['T'] = np.zeros((max(input['points'].shape), 1))
    # if type:
    #     # First 4 nodes are exposed to deltaT
    #     input['T'][0:4] = deltaT * np.ones((4, 1))
    # else:
    #     # All nodes are exposed to deltaT
    #     input['T'] = deltaT * np.ones((max(input['points'].shape)))

    # Create mirror side of structure, if so selected
    # --------------------------------------------------------------------------
    # if sym and not type:
    #
    #
    #     # Connectivity
    #     a = np.nonzero(input['parts'][0, :] == 1)[0]
    #     l = max(input['points'].shape)
    #     # Points (first point should not be duplicated)
    #     input['points'] = np.vstack((
    #         np.concatenate((input['points'][0, :], -input['points'][0, 1:]), axis=0).reshape(1, 2 * l - 1),
    #         np.concatenate((input['points'][1, :], input['points'][1, 1:]), axis=0).reshape(1, 2 * l - 1)
    #     ))
    #
    #     # Boundary conditions (first point should not be duplicated)
    #     input['bc'] = np.concatenate((input['bc'], input['bc'][3:]))
    #     # Loading vector
    #     input['P'] = np.concatenate((input['P'], input['P'][3:]))
    #     # Thermal vector
    #     input['T'] = np.concatenate((input['T'], input['T'][1:]))
    #
    #     # Define node connections
    #     # ~~~~~~~~~~~~~~~~~~~~~~~
    #     # Advancing the node numbers to be connected apart for node 1
    #     c = (l - 1) * np.ones((np.shape(input['parts'])), dtype=int)
    #     c[0, a] = np.zeros((1, len(a)), dtype=int)
    #     # Symmetric side connections
    #     input['parts'] = np.concatenate((
    #         input['parts'],
    #         (c + input['parts'])[::-1]
    #     ), axis=1)
    #     # Double the initialized vectors
    #     input['type'] = np.concatenate((input['type'], input['type']))
    #     input['seed_array'] = np.concatenate((input['seed_array'], input['seed_array']))
    #     input['material_indeces'] = np.concatenate((input['material_indeces'], input['material_indeces']))
    #     input['section_indeces'] = np.concatenate((input['section_indeces'], input['section_indeces']))
    # else:
    #     # Apply symmetry boundary conditions at point 1
    #     input['bc'][0] = 1  # 1 means d.o.f. is restrained, 0 means d.o.f. is free
    #     input['bc'][2] = 1

    # Applied loading
    # --------------------------------------------------------------------------
    for force in forces:
        input['P'][3 * (force['point'] - 1) + force['axis']] = force['magnitude']

    # Convert theta to degrees
    # input['theta'] = theta * 180 / np.pi

    # Store input data to output
    input['mat'] = materials
    input['sec'] = sections
    input['seeds'] = seeds
    input['lump_mass'] = lump_mass

    # Output settings
    # --------------------------------------------------------------------------
    # Number of eigenfrequencies returned
    input['nFreq'] = nFreq
    # Plot scale
    input['plot_scale'] = plot_scale

    return input

def read_input(input_file: str, type:int) -> tuple[str, Any]:
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
    if type:
        bc = d['BoundaryConditions_Beam']
    else:
        bc = d['BoundaryConditions_Truss']

    # Output settings
    output_settings = d['Output']

    return name, geometry, sections, part_properties, forces, bc, output_settings


def validation_geometry(input_file: str, type: int) -> dict[str, Any]:
    # Read input from TOML file

    name, geometry, sections, part_properties, forces, bc, output_settings= read_input(input_file, type)

    # Read materials from input TOML file
    materials_file = 'materials.toml'
    materials = read_materials(materials_file)
    geometry['type'] = type
    # Call the function that creates the geometry
    input = create_simulation_input(
        name=name,
        # Geometry, discretisation
        **geometry,
        # Material and section properties
        materials=materials,
        sections=sections,
        part_materials=part_properties['materials'],
        part_sections=part_properties['sections'],
        forces=forces,
        # Boundary condition application
        bc=bc,
        # Output settings
        **output_settings
    )
    return input

