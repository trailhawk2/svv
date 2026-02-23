import numpy as np
from objects.assembly import Assembly

class Element:
    def __init__(self, assembly: Assembly, ix: int) -> None:
        '''Initializes the element class
        INPUTS:
            assembly: Assembly -> assembly object
            ix: int -> index of the element
        OUTPUTS:
            None
        '''
        self.assembly = assembly
        self.ix = ix
        self.transformation_matrix = None

        # Local stiffness matrix
        # ----------------------------------------------------------------------
        self.local_stiffness_matrix()

        # Thermal load vector
        # ----------------------------------------------------------------------
        self.local_thermal_load_vector()

        # Rotation matrix
        # ----------------------------------------------------------------------
        self.rotation_matrix()

        # Mass matrix
        # ----------------------------------------------------------------------
        self.local_mass_matrix()

    def local_stiffness_matrix(self) -> None:
        '''Computes the local stiffness matrix of the element
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # ----------------------------------------------------------------------
        
        e = self.assembly.mesh['element'][self.ix]
        
        e['Kbar'] = ((e['E'] * e['A']) / e['length']) \
                    * \
                    np.concatenate((
                        np.array([1, 0, 0, -1, 0, 0]).reshape(1, 6),
                        np.zeros((2, 6)),
                        np.array([-1, 0, 0, 1, 0, 0]).reshape(1, 6),
                        np.zeros((2, 6))), axis=0)

        # Add terms for bending stiffness if the element is a beam element
        if e['type']:
            e['Kbar'] = e['Kbar'] + ((e['E'] * e['I']) / (e['length'] ** 3)) * \
                np.concatenate((
                    np.zeros((1, 6)),
                    np.array([
                        0,
                        12, 
                        6 * e['length'],
                        0,
                        -12,
                        6 * e['length']
                    ]).reshape(1, 6),
                    np.array([
                        0,
                        6 * e['length'],
                        4 * e['length'] ** 2,
                        0,
                        -6 * e['length'],
                        2 * e['length'] ** 2
                    ]).reshape(1, 6),
                    np.zeros((1, 6)),
                    np.array([
                        0,
                        -12, 
                        -6 * e['length'],
                        0,
                        12,
                        -6 * e['length']
                    ]).reshape(1, 6),
                    np.array([
                        0, 
                        6 * e['length'], 
                        2 * e['length'] ** 2, 
                        0, 
                        -6 * e['length'], 
                        4 * e['length'] ** 2
                    ]).reshape(1, 6)
                ))

    def local_thermal_load_vector(self) -> None:
        '''Computes the local thermal load vector of the element
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # ----------------------------------------------------------------------

        e = self.assembly.mesh['element'][self.ix]
        
        e['Qbar'] = \
            e['E'] * e['A'] * \
            e['alpha'] * \
            e['DeltaT'] * \
            np.array([[1], [0], [0], [-1], [0], [0]], dtype=float)
    

    def rotation_matrix(self) -> None:
        '''Computes the rotation matrix of the element
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # ----------------------------------------------------------------------

        e = self.assembly.mesh['element'][self.ix]

        T = np.diag([
            np.cos(e['rotation']),
            np.cos(e['rotation']), 1,
            np.cos(e['rotation']),
            np.cos(e['rotation']), 1
        ])

        T[0, 1] = np.sin(e['rotation'])
        T[1, 0] = -np.sin(e['rotation'])
        T[3, 4] = np.sin(e['rotation'])
        T[4, 3] = -np.sin(e['rotation'])

        e['T'] = T

        # Rotate the local stiffness matrix to the global coordinate system
        e['K'] = np.transpose(T) @ e['Kbar'] @ T
        # Rotate the local thermal load vector to the global coordinate system
        e['Q'] = np.transpose(T) @ e['Qbar']

    def local_mass_matrix(self) -> None:
        '''Computes the local mass matrix of the element
        INPUTS:
            None
        OUTPUTS:
            None
        '''

        # ----------------------------------------------------------------------

        e = self.assembly.mesh['element'][self.ix]

        rhoAL = e['rho'] * e['A'] * e['length']
        rhoIL = e['rho'] * e['I'] / e['length']

        if e['type']:
            # Mass matrix for beam element
            mRhoA = rhoAL / 420 * np.concatenate((
                np.array([
                    140,
                    0,
                    0,
                    70,
                    0,
                    0
                ]).reshape(1, 6),
                np.array([
                    0,
                    156,
                    22 * e['length'],
                    0,
                    54,
                    -13 * e['length']
                ]).reshape(1, 6),
                np.array([
                    0,
                    22 * e['length'],
                    4 * e['length'] ** 2,
                    0,
                    13 * e['length'],
                    -3 * e['length'] ** 2
                ]).reshape(1, 6),
                np.array([
                    70,
                    0,
                    0,
                    140,
                    0,
                    0
                ]).reshape(1, 6),
                np.array([
                    0,
                    54,
                    13 * e['length'],
                    0,
                    156,
                    -22 * e['length']]
                ).reshape(1, 6),
                np.array([
                    0,
                    -13 * e['length'],
                    -3 * e['length'] ** 2,
                    0,
                    -22 * e['length'],
                    4 * e['length'] ** 2
                ]).reshape(1, 6)))

            mRhoI = rhoIL / 30 * np.concatenate((
                np.zeros((1, 6)),
                np.array([
                    0,
                    36,
                    3 * e['length'],
                    0,
                    -36,
                    3 * e['length']]).reshape(1, 6),
                np.array([
                    0,
                    3 * e['length'],
                    4 * e['length'] ** 2,
                    0,
                    -3 * e['length'],
                    -e['length'] ** 2]).reshape(1, 6),
                np.zeros((1, 6)),
                np.array([
                    0,
                    -36,
                    -3 * e['length'],
                    0,
                    36,
                    -3 * e['length']]).reshape(1, 6),
                np.array([
                    0, 3 * e['length'],
                    -e['length'] ** 2,
                    0,
                    -3 * e['length'],
                    4 * e['length'] ** 2]).reshape(1, 6)))

            e['m'] = mRhoA + mRhoI

        else:
            # Mass matrix for rod element
            if self.assembly.input['lump_mass']:
                # Lumped mass matrix
                e['m'] = rhoAL / 2 * np.eye(6)
            else:
                # Regular mass matrix
                e['m'] = \
                    rhoAL / 6 * np.concatenate((
                        np.concatenate((2 * np.eye(3), np.eye(3)), axis=1),
                        np.concatenate((np.eye(3), 2 * np.eye(3)), axis=1),
                    ))
