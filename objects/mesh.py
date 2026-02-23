import numpy as np
import matplotlib.pyplot as plt
from objects.assembly import Assembly
from typing import Any


class Mesh:
    def __init__(self, assembly: Assembly) -> None:
        '''Initializes the mesh class
        INPUTS:
            assembly: Assembly -> assembly object
        OUTPUTS:
            None
        '''

        # Make the Mesh aware of which assembly it's a part of
        self.assembly = assembly

        # Make this Mesh object an attribute of its parent assembly
        self.assembly.mesh = self

        # Create mesh
        self.mesh = self.create_mesh()

    def __getitem__(self, key: str) -> Any:
        '''Allow users to retrieve values from the Mesh's dictionary
        INPUTS:
            key: str -> key of the dictionary
        OUTPUTS:
            Any -> value of the dictionary
        '''
        return self.mesh[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        '''Allow users to assign values to items in the Mesh's dictionary
        INPUTS:
            key: Any -> key of the dictionary
            value: Any -> value of the dictionary
        OUTPUTS:
            None
        '''
        self.mesh[key] = value

    def create_mesh(self) -> dict[str, Any]:
        '''Creates the mesh
        INPUTS:
            None
        OUTPUTS:
            dict[str, Any] -> dictionary containing the mesh
        '''

        # Loop over the specified parts and create nodes and elements per part
        # --------------------------------------------------------------------------

        # Seed
        if self.assembly.input['seeds'] != 0:
            if np.all(self.assembly.input['type'] == 1):
                # Create a parts-sized array filled with 1*(number of seeds)
                self.assembly.input['seed_array'] = self.assembly.input['seeds'] * np.ones((max(self.assembly.input['parts'].shape)))

        # Initialize counter for nodes
        iN = max(self.assembly.input['points'].shape)
        
        # Initialize mesh
        mesh = {}
        mesh['part'] = [dict() for i_dic in range(len(self.assembly.input['parts'][0, :]))]

        # Loop over parts
        # ---------------
        for ix in range(len(self.assembly.input['parts'][0, :])):
            
            # Length of the part
            mesh['part'][ix]['length'] = np.sqrt(sum(
                (self.assembly.input['points'][:, self.assembly.input['parts'][1, ix]-1] - \
                 self.assembly.input['points'][:, self.assembly.input['parts'][0, ix]-1])**2
            ))

            # Raise error for overlapping nodes. Implemented after failure of UT3.4
            if mesh['part'][ix]['length'] == 0:
                raise ValueError("Two or more nodes located at the same coordinates! Ensure no overlapping nodes.")


            # Unit vector of the part (local x-axis)
            mesh['part'][ix]['direction'] = (self.assembly.input['points'][:, self.assembly.input['parts'][1, ix]-1] -
                                             self.assembly.input['points'][:, self.assembly.input['parts'][0, ix]-1])/mesh['part'][ix]['length']
            
            # Rotation of element w.r.t. global coordinate system [radians]
            mesh['part'][ix]['rotation'] = np.arctan2(
                mesh['part'][ix]['direction'][1], mesh['part'][ix]['direction'][0])
            
            # Element length is obtained by dividing the length of the elemnt by the number of seeded nodes
            mesh['part'][ix]['elementLength'] = mesh['part'][ix]['length'] / (self.assembly.input['seed_array'][ix]+1)
            
            # Create an array with the coordinates of the nodes
            multiplierVector = np.arange(
                0, 1+(1/(self.assembly.input['seed_array'][ix]+1))/2, (1/(self.assembly.input['seed_array'][ix]+1)))*mesh['part'][ix]['length']

            mesh['part'][ix]['nodes'] = self.assembly.input['points'][:, self.assembly.input['parts'][0, ix]-1].reshape((2, 1)) @\
                np.ones((len(multiplierVector)))[
                np.newaxis] + mesh['part'][ix]['direction'].reshape((2, 1))@multiplierVector[np.newaxis]
            
            # Number nodes in the part
            # ------------------------
            # First node
            firstNode = self.assembly.input['parts'][0, ix]
            # Last node
            lastNode = self.assembly.input['parts'][1, ix]
            # Intermediate nodes
            nInterNodes = len(multiplierVector)-2
            if nInterNodes < 1:
                interNodes = []
            else:
                interNodes = [k for k in range(iN+1, iN+nInterNodes+1)]

            # Advance counter node number
            iN = iN+nInterNodes
            # Assign node numbers to part
            mesh['part'][ix]['nodeNumbers'] = [firstNode] + interNodes + [lastNode]
            mesh['part'][ix]['nNodes'] = 2+nInterNodes
            # Number of elements in the part
            mesh['part'][ix]['nElements'] = len(multiplierVector)-1
            # Temperature of the part (only if first node and last node have equal non-zero temperature
            if (self.assembly.input['T'][firstNode-1] > 0) and (self.assembly.input['T'][firstNode-1] == self.assembly.input['T'][lastNode-1]):
                mesh['part'][ix]['DeltaT'] = self.assembly.input['T'][firstNode-1]
            else:
                mesh['part'][ix]['DeltaT'] = 0

        # Store total amount of nodes
        mesh['nNodes'] = iN

        # Transfer part nodes to global nodes
        # --------------------------------------------------------------------------
        
        # Initialise array for nodal coordinates (this is done here, because before the total amount of nodes was unknown)
        mesh['nodes'] = np.zeros((2, iN))

        # Each of the specified points is a node. These nodes have the lowest node numbers
        mesh['nodes'][:, :max(self.assembly.input['points'].shape)] = self.assembly.input['points']

        # Loop over parts to collect nodal coordinates
        for ix in range(len(self.assembly.input['parts'][0, :])):
            # Only the intermediate nodes need to be added to mesh.nodes
            if mesh['part'][ix]['nNodes'] > 2:
                for jx in range(2, mesh['part'][ix]['nNodes']):
                    mesh['nodes'][:, mesh['part'][ix]['nodeNumbers'][jx-1]-1] = mesh['part'][ix]['nodes'][:, jx-1]
        
        return mesh

    def assign_element_properties(self) -> None:
        '''Assigns material properties to each element of each part in the mesh
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # --------------------------------------------------------------------------
        
        # Inintialize counter for elements
        iE = 0

        # Initialize list of dict with large number of empty dict (here 1000)
        self.mesh['element'] = [dict() for i_dic in range(1000)]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assembly properties
        materials = self.assembly.input['mat']
        sections  = self.assembly.input['sec']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Loop over all parts
        for ix in range(len(self.assembly.input['parts'][0, :])):
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Part properties
            part           = self.mesh['part'][ix]
            type           = self.assembly.input['type'][ix]
            part_material  = self.assembly.input['material_indeces'][ix]
            part_section   = self.assembly.input['section_indeces'][ix]
            # Retrieve mechanical properties from part properties
            E              = materials[part_material]['E']
            rho            = materials[part_material]['rho']
            alpha          = materials[part_material]['alpha']
            A              = sections[part_section]['A']
            I              = sections[part_section]['I']
            # Retrieve geometric parameters from part properties
            rotation       = part['rotation']
            element_length = part['elementLength']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # Register first element number of part
            firstElementNumber = iE+1

            # Loop over elements in part
            for jx in range(part['nElements']):

                # Advance element count
                iE = iE+1

                # Number of the part that the element belongs to
                self.mesh['element'][iE]['partNumber1'] = ix

                # Node numbers belonging to the element
                self.mesh['element'][iE]['nodeNumber1'] = part['nodeNumbers'][jx]
                self.mesh['element'][iE]['nodeNumber2'] = part['nodeNumbers'][jx+1]

                # Retrieve properties
                self.mesh['element'][iE]['E']        = E
                self.mesh['element'][iE]['rho']      = rho
                self.mesh['element'][iE]['alpha']    = alpha
                self.mesh['element'][iE]['A']        = A
                self.mesh['element'][iE]['I']        = I
                self.mesh['element'][iE]['type']     = type
                self.mesh['element'][iE]['rotation'] = rotation
                self.mesh['element'][iE]['length']   = element_length

                # Calculate mass of the element
                self.mesh['element'][iE]['mass'] = self.mesh['element'][iE]['length'] * \
                    self.mesh['element'][iE]['A']*self.mesh['element'][iE]['rho']
                
                # Divide the mass of the element over its two nodes
                self.mesh['element'][iE]['lumpedMassNodeNumber1'] = self.mesh['element'][iE]['mass']/2
                self.mesh['element'][iE]['lumpedMassNodeNumber2'] = self.mesh['element'][iE]['mass']/2

                # Assign element temperature
                self.mesh['element'][iE]['DeltaT'] = part['DeltaT']

            # Register last element number of part
            lastElementNumber = iE
            part['elementNumbers'] = range(
                firstElementNumber, lastElementNumber+1)
    
        # Store total amount of elements
        self.mesh['nElements'] = iE

        # Keep only filled dict
        self.mesh['element'] = self.mesh['element'][1:iE+1]
    
    def plot_mesh(self, show: bool = True, save: bool = False) -> None:
        '''Plots the mesh
        INPUTS:
            show: bool -> whether to show the plot
            save: bool -> whether to save the plot
        OUTPUTS:
            None
        '''

        # Plot mesh for visual check
        # --------------------------------------------------------------------------

        # Loop over all parts and plot them
        for ix in range(len(self.assembly.input['parts'][0, :])):
            plt.plot(self.assembly.input['points'][0, self.assembly.input['parts'][:, ix]-1], self.assembly.input['points']
                    [1, self.assembly.input['parts'][:, ix]-1], '0.8', linewidth=2)

        # Plot all nodes
        plt.plot(self.mesh['nodes'][0, :], self.mesh['nodes'][1, :], 'ok',
                linewidth=4, markerfacecolor='None', markersize=5)
        
        # Format the axis of the plot
        plt.xlim([min(self.assembly.input['points'][0, :])-0.5, max(self.assembly.input['points'][0, :])+0.5])
        plt.ylim([min(self.assembly.input['points'][1, :])-0.5, max(self.assembly.input['points'][1, :])+0.5])

        # Optionally, save and show figure
        if save:
            plt.savefig('mesh_2.png')
        if show:
            plt.show()
