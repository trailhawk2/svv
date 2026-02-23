import math
import numpy as np
from pprint import pprint
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from docutils.nodes import label
from matplotlib.ticker import FormatStrFormatter
from typing import Union
import pickle
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import Normalize


class Assembly:
    def __init__(self, input: dict[str, Union[int, float]]) -> None:
        '''Initializes the Assembly class
        INPUTS:
            input: dict -> dictionary containing the input data
        OUTPUTS:
            None
        '''
        
        # Read user input dictionary
        self.input = input

        # Transform Young's modulus input in GPa to MPa
        for material in self.input['mat']: material['E'] *= 10**3

        # Initialise output
        self.output = {'name': self.input['name']}

    def plot_input(self, show: bool = True, save: bool = False) -> None:

        '''Plot the input data
        INPUTS:
            show: bool -> show the plot
            save: bool -> save the plot
        OUTPUTS:
            None
        '''

        # Check the provided input
        # --------------------------------------------------------------------------
        if 'parts' in self.input.keys() is False:
            raise ('Error: self.input.parts has not been specified by user')

        # Display the structure for the user to check it visually
        # --------------------------------------------------------------------------
        # Plot the parts
        colors = ['g-', 'r-', 'b-'] #'g-' for aluminium, 'r-' for steel, 'b-' for CFRP
        materials = ['aluminium', 'steel', 'CFRP']
        pprint(self.input['material_indeces'])

        for ix in range(len(self.input['parts'][0, :])):
            mat_idx = int(self.input['material_indeces'][ix])
            plt.plot(self.input['points'][0, self.input['parts'][:, ix]-1],
                     self.input['points'][1, self.input['parts'][:, ix]-1], colors[mat_idx],
                     marker='o', linewidth=2, markersize=0, label=materials[mat_idx])

        # Plot the points
        plt.plot(self.input['points'][0, :], self.input['points']
                    [1, :], 'ok', linewidth=4, markersize=5)

        # Plot labels for each point
        #iterations = 0
        if self.input['type'][0]: ### use parts for beam structure, points for truss
            # Beam structure
            iterations = len(self.input['parts'][0, :])
        else:
            # truss structure
            iterations = len(self.input['points'][0, :])

        for ix in range(iterations):
            # add labels to each point
            plt.annotate(ix + 1, (self.input['points'][0, ix], self.input['points'][1, ix]),
                         xytext=(self.input['points'][0, ix] + 50,
                                 self.input['points'][1, ix] + 50),
                         arrowprops=dict(arrowstyle="-"))

        # Set axis labels and titel
        plt.xlabel('x location [mm]', fontweight='bold')
        plt.ylabel('y location [mm]', fontweight='bold')
        plt.title(f'Initial position of the structure', loc='left', fontsize=12)

        # Filter duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Format the axes of the plot
        xmin = min(self.input['points'][0, :])
        xmax = max(self.input['points'][0, :])
        ymin = min(self.input['points'][1, :])
        ymax = max(self.input['points'][1, :])
        xrange = xmax - xmin
        yrange = ymax - ymin
        plt.xlim([xmin - 0.05 * xrange, xmax + 0.05 * xrange])
        plt.ylim([ymin - 0.05 * yrange, ymax + 0.05 * yrange])

        # Include grid and ensure equal spacing
        plt.axis('equal')  # Ensures 1mm on X equals 1mm on Y
        plt.grid(True, linestyle=':', alpha=0.6)

        # Resize to tight layout
        plt.tight_layout()

        # Optionally, save and show figure
        if save:
            plt.savefig('mesh_1.png')
        if show:
            plt.show()

    def global_stiffness_matrix(self) -> None:
        '''Initialize the global stiffness matrix. Each node has 3 degrees of freedom: u,v, and theta
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # --------------------------------------------------------------------------
        
        self.mesh['K'] = np.zeros((3 * self.mesh['nNodes'], 3 * self.mesh['nNodes']))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemlbe the global stiffness matrix
            for jx in self.mesh['part'][ix]['elementNumbers']:
                # Bounds
                lb1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber1'] - 1) + 1
                lb2 = 3 * self.mesh['element'][jx - 1]['nodeNumber1']
                ub1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber2'] - 1) + 1
                ub2 = 3 * self.mesh['element'][jx - 1]['nodeNumber2']
                bounds = np.concatenate(
                    (np.arange(lb1, lb2 + 1), np.arange(ub1, ub2 + 1))) - 1
                # Assemble
                self.mesh['K'][np.ix_(bounds, bounds)] = self.mesh['K'][np.ix_(
                    bounds, bounds)] + self.mesh['element'][jx - 1]['K']

    def global_thermal_load_vector(self) -> None:
        '''Create global thermal load vector
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # --------------------------------------------------------------------------

        self.output['Q'] = np.zeros((np.shape(self.mesh['K'])[0], 1))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemlbe the thermal load vector
            for jx in self.mesh['part'][ix]['elementNumbers']:
                # Bounds
                lb1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber1'] - 1) + 1
                lb2 = 3 * self.mesh['element'][jx - 1]['nodeNumber1']
                ub1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber2'] - 1) + 1
                ub2 = 3 * self.mesh['element'][jx - 1]['nodeNumber2']
                bounds = np.concatenate(
                    (np.arange(lb1, lb2 + 1), np.arange(ub1, ub2 + 1))) - 1
                # Assemble
                self.output['Q'][bounds] = self.output['Q'][bounds] + self.mesh['element'][jx - 1]['Q']

    def global_mass_matrix(self) -> None:
        '''Initialize the global mass matrix
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # --------------------------------------------------------------------------
        
        self.mesh['m'] = np.zeros((3 * self.mesh['nNodes'], 3 * self.mesh['nNodes']))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemble the global stiffness matrix
            for jx in self.mesh['part'][ix]['elementNumbers']:
                # Bounds
                lb1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber1'] - 1) + 1
                lb2 = 3 * self.mesh['element'][jx - 1]['nodeNumber1']
                ub1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber2'] - 1) + 1
                ub2 = 3 * self.mesh['element'][jx - 1]['nodeNumber2']
                bounds = np.concatenate((np.arange(lb1, lb2 + 1), np.arange(ub1, ub2 + 1))) - 1
                # Assemble
                self.mesh['m'][np.ix_(bounds, bounds)] = \
                    self.mesh['m'][np.ix_(bounds, bounds)] + self.mesh['element'][jx - 1]['m']

    def loads_bc(self) -> None:
        '''Applying Loads and Boundary Conditions
        INPUTS:
            None
        OUTPUTS:
            None
        '''
        # --------------------------------------------------------------------------
        
        # Assemble the loading vector
        # Applied loads
        self.output['P'] = np.zeros((np.shape(self.mesh['K'])[0], 1))
        self.output['P'][:max((self.input['points']).shape) * 3] = self.input['P']

        # Displacements of entire system
        self.output['U'] = np.zeros((np.shape(self.mesh['K'])[0], 1))

        # Reaction forces of entire system
        self.output['R'] = np.zeros((np.shape(self.mesh['K'])[0], 1))

        # Reduce the amount of degrees of freedom if rod element is selected
        # --------------------------------------------------------------------------
        if np.all(self.input['type'] == 1):
            # Nothing happens
            pass
        else:
            # Find the indices of the remaining DOFs
            remainDF = np.nonzero(matlib.repmat(
                [1, 1, 0], 1, int(max(self.mesh['K'].shape) / 3)))[1]
            remainBC = np.nonzero(matlib.repmat(
                [1, 1, 0], 1, int(max(self.input['bc'].shape) / 3)))[1]
            # Reduce vector with boundary conditions
            self.input['bc'] = self.input['bc'][remainBC]
            # Reduce stiffness and mass matrices
            self.mesh['K'] = self.mesh['K'][np.ix_(remainDF, remainDF)]
            self.mesh['m'] = self.mesh['m'][np.ix_(remainDF, remainDF)]
            # Reduce displacement, load, thermal load and reaction force vectors
            self.output['U'] = self.output['U'][remainDF]
            self.output['P'] = self.output['P'][remainDF]
            self.output['Q'] = self.output['Q'][remainDF]
            self.output['R'] = self.output['R'][remainDF]

        # Remove blocked degrees of freedom
        # --------------------------------------------------------------------------
        # Active degrees of freedom
        # 1 means d.o.f. is restrained, 0 means d.o.f. is free
        activeDF = np.ones((np.shape(self.mesh['K'])[0]))
        # Find the clamped nodes
        inactiveDF = np.nonzero(self.input['bc'])[0]
        # Inactive degrees of freedom
        activeDF[inactiveDF] = 0
        inactiveDF = np.ones((np.shape(self.mesh['K'])[0])) - activeDF
        # Convert to zeros and ones to indices
        activeDF = np.nonzero(activeDF)[0]
        inactiveDF = np.nonzero(inactiveDF)[0]

        # Reduced stiffness matrices
        Kr = self.mesh['K'][np.ix_(activeDF, activeDF)]
        Ksr = self.mesh['K'][np.ix_(inactiveDF, activeDF)]
        # Reduced load vectors
        Pr = self.output['P'][activeDF]
        Ps = self.output['P'][inactiveDF]
        Qr = self.output['Q'][activeDF]
        Qs = self.output['Q'][inactiveDF]
        # Reduced mass matrix
        mr = self.mesh['m'][np.ix_(activeDF, activeDF)]

        # Return mesh as output
        self.output['Kr'] = Kr
        self.output['Ksr'] = Ksr
        self.output['Pr'] = Pr
        self.output['Ps'] = Ps
        self.output['Qr'] = Qr
        self.output['Qs'] = Qs
        self.output['mr'] = mr

        # Active and inactive degrees of freedom
        self.output['activeDF'] = activeDF
        self.output['inactiveDF'] = inactiveDF

        # Store a reference to the mesh in the output dictionary for ease of access
        self.output['mesh'] = self.mesh

    def plot_output(self, color: str = 'red', show: bool = True, displacement_plot: bool = True, stress_plot: bool = False) -> None:
        # # --------------------------------------------------------------------------
        # # New custom plot that displays the internal stresses.
        # # --------------------------------------------------------------------------
        # if stress_plot:
        #     # 1. Data Preparation
        #     stress = np.array(self.output['B6']['stress'])
        #     strain = np.array(self.output['B6']['strain'])
        #
        #     # 2. Logic for Displacement Extraction
        #     n_dims = 3 if np.all(self.input['type'] == 1) else 2
        #     locDisp3D = self.output['U'].reshape(n_dims, self.mesh.mesh['nNodes'], order='F')
        #     locDisp = self.mesh.mesh['nodes'] + self.input['plot_scale'] * locDisp3D[0:2, :]
        #
        #     # 3. Initialize Figure with two subplots
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        #     cmap = plt.colormaps["plasma"]
        #
        #     # Helper function to prevent Normalize crashes if values are constant
        #     def get_norm(data):
        #         d_min, d_max = data.min(), data.max()
        #         if d_min == d_max:
        #             return Normalize(vmin=d_min - 0.1, vmax=d_max + 0.1)
        #         return Normalize(vmin=d_min, vmax=d_max)
        #
        #     norm_stress = get_norm(stress)
        #     norm_strain = get_norm(strain)
        #
        #     # 4. Plotting Loop
        #     for ix in range(self.input['parts'].shape[1]):
        #         node_indices = self.input['parts'][:, ix] - 1
        #         x_pos = locDisp[0, node_indices]
        #         y_pos = locDisp[1, node_indices]
        #
        #         # Plot Stress (Left Subplot)
        #         ax1.plot(x_pos, y_pos, color=cmap(norm_stress(stress[ix])),
        #                  linewidth=2, linestyle='-')
        #
        #         # Plot Strain (Right Subplot)
        #         ax2.plot(x_pos, y_pos, color=cmap(norm_strain(strain[ix])),
        #                  linewidth=2, linestyle='-')
        #
        #     # 5. Add Nodes and formatting for both plots
        #     for ax in [ax1, ax2]:
        #         ax.plot(locDisp[0, :], locDisp[1, :], color='black', marker='o',
        #                 linestyle='', markerfacecolor='white', markersize=4, zorder=5)
        #         ax.set_aspect('equal')
        #         ax.set_xlabel("X [mm]")
        #         ax.grid(True, linestyle=':', alpha=0.6)
        #
        #     ax1.set_ylabel("Y [mm]")
        #     ax1.set_title(f"Stresses [Pa]")
        #     ax2.set_title(f"Strains [-]")
        #
        #     # 6. Add individual colorbars
        #     # Stress Colorbar
        #     # Stress Colorbar
        #     sm_stress = plt.cm.ScalarMappable(cmap=cmap, norm=norm_stress)
        #     sm_stress.set_array([])
        #     fig.colorbar(sm_stress, ax=ax1, label='Axial Stress [Pa]', fraction=0.046, pad=0.04, shrink=0.4)
        #
        #     # Strain Colorbar
        #     sm_strain = plt.cm.ScalarMappable(cmap=cmap, norm=norm_strain)
        #     sm_strain.set_array([])
        #     fig.colorbar(sm_strain, ax=ax2, label='Axial Strain [-]', fraction=0.046, pad=0.04, shrink=0.4)
        #
        #
        #     plt.tight_layout()
        #     if show:
        #         plt.show()

        # 1. Data Preparation
        # Pulling stress and strain from your calculation results
        stress = np.array(self.output['B6']['stress'])
        strain = np.array(self.output['B6']['strain'])

        # 2. Logic for Displacement Extraction
        n_dims = 3 if np.all(self.input['type'] == 1) else 2

        # Reshape global U to match (dims, nNodes)
        # This contains the positions of all nodes, including the seeds
        locDisp3D = self.output['U'].reshape(n_dims, self.mesh.mesh['nNodes'], order='F')

        # Calculate current coordinates (Initial + Scaled Displacement)
        locDisp = self.mesh.mesh['nodes'] + self.input['plot_scale'] * locDisp3D[0:2, :]

        # 3. Initialize Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        cmap = plt.colormaps["cividis"]

        def get_norm(data):
            d_min, d_max = data.min(), data.max()
            if d_min == d_max:
                return Normalize(vmin=d_min - 1e-6, vmax=d_max + 1e-6)
            return Normalize(vmin=d_min, vmax=d_max)

        norm_stress = get_norm(stress)
        norm_strain = get_norm(strain)

        # 4. Plotting Loop (Drawing Element by Element)
        # We iterate through the actual mesh elements (which connect the seeds)
        for ix in range(self.mesh.mesh['nElements']):
            # Get the two specific seed/node indices for this element
            # Your dictionary uses 'nodeNumber1' and 'nodeNumber2' for each element
            e = self.mesh.mesh['element'][ix]
            n1 = e['nodeNumber1'] - 1
            n2 = e['nodeNumber2'] - 1

            x_seg = [locDisp[0, n1], locDisp[0, n2]]
            y_seg = [locDisp[1, n1], locDisp[1, n2]]

            # Plot Stress on Subplot 1
            ax1.plot(x_seg, y_seg, color=cmap(norm_stress(stress[ix])),
                     linewidth=4, linestyle='-')

            # Plot Strain on Subplot 2
            ax2.plot(x_seg, y_seg, color=cmap(norm_strain(strain[ix])),
                     linewidth=4, linestyle='-')

        # 5. Visual Formatting

        for ax in [ax1, ax2]:
            # Plot the seeds (all nodes in the mesh)
            ax.plot(locDisp[0, :], locDisp[1, :], color='black', marker='o',
                    linestyle='', markerfacecolor='white', markersize=3, zorder=5)

            ax.set_aspect('equal')
            ax.set_xlabel("X [mm]")
            ax.grid(True, linestyle=':', alpha=0.6)

        ax1.set_ylabel("Y [mm]")
        ax1.set_title(f"Stresses [Pa]")
        ax2.set_title(f"Strains [-]")

        # 6. Add Colorbars with aligned height fix
        sm_stress = plt.cm.ScalarMappable(cmap=cmap, norm=norm_stress)
        sm_stress.set_array([])
        fig.colorbar(sm_stress, ax=ax1, label='Axial Stress [Pa]', fraction=0.046, pad=0.04, shrink=0.4)

        sm_strain = plt.cm.ScalarMappable(cmap=cmap, norm=norm_strain)
        sm_strain.set_array([])
        fig.colorbar(sm_strain, ax=ax2, label='Axial Strain [-]', fraction=0.046, pad=0.04, shrink=0.4)

        plt.tight_layout()
        if show:
            plt.show()
        # --------------------------------------------------------------------------
        # New custom plot that displays deformation
        # --------------------------------------------------------------------------
        if displacement_plot:

            # 1. Logic for Displacement Extraction
            n_dims = 3 if np.all(self.input['type'] == 1) else 2
            locDisp3D = self.output['U'].reshape(n_dims, self.mesh['nNodes'], order='F')

            # 2D locations of the displaced nodes
            locDisp = self.mesh['nodes'] + self.input['plot_scale'] * locDisp3D[0:2, :]

            # 2. Plotting (Hierarchy: Undeformed in background, Deformed in foreground)
            # Plot elements (Undeformed)
            for ix in range(self.input['parts'].shape[1]):
                plt.plot(self.input['points'][0, self.input['parts'][:, ix] - 1],
                         self.input['points'][1, self.input['parts'][:, ix] - 1],
                         color='0.8', linestyle='--', linewidth=1, label='Initial positions' if ix == 0 else "")

            # Plot original nodes
            plt.scatter(self.mesh['nodes'][0, :], self.mesh['nodes'][1, :],
                        edgecolor='0.7', facecolor='None', s=20, zorder=2)

            # Plot displaced nodes/structure
            plt.plot(locDisp[0, :], locDisp[1, :], color=color, marker='o',
                     linestyle='', linewidth=2, markerfacecolor='white', markersize=6,
                     label=f'Deformed structure')

            # 3. Smart Axis Formatting
            # Use 'g' for 3 significant digits: handles 100 and 0.00123 beautifully
            fmt = StrMethodFormatter('{x:.3g}')
            plt.gca().xaxis.set_major_formatter(fmt)
            plt.gca().yaxis.set_major_formatter(fmt)

            # 4. Geometry and Scaling
            plt.axis('equal')  # CRITICAL: Ensures 1mm on X equals 1mm on Y
            plt.grid(True, linestyle=':', alpha=0.6)

            # Auto-adjust limits with a slight margin
            plt.margins(0.1)

            # 5. Labels and Legend
            plt.xlabel('x location [mm]', fontweight='bold')
            plt.ylabel('y location [mm]', fontweight='bold')
            plt.title('Structural Deformation Analysis', loc='left', fontsize=12)
            plt.legend()

            # adjust horizontal label alignment
            plt.xticks(rotation=30, ha='right')

            plt.tight_layout()

            if show:
                plt.show()

        # --------------------------------------------------------------------------
        # Plot the default plot that is present in the original version of the code
        # --------------------------------------------------------------------------
        if False:
            '''Plot output
            INPUTS:
                color: str -> color of the plot
                show: bool -> show the plot
            OUTPUTS:
                None
            '''
            # TODO: improve plots to ease comparison, etc.
            # --------------------------------------------------------------------------

            if np.all(self.input['type'] == 1):
                # Reshape 3 displacements per node
                locDisp3D = self.output['U'].reshape(3, self.mesh['nNodes'], order='F')
            else:
                # Reshape 2 displacements per node
                locDisp3D = self.output['U'].reshape(2, self.mesh['nNodes'], order='F')

            # 2D locations of the displaced nodes
            locDisp = self.mesh['nodes'] + self.input['plot_scale']*locDisp3D[0:2, :]

            # Plot elements
            for ix in range(max(self.input['parts'][0, :].shape)):
                plt.plot(self.input['points'][0, self.input['parts'][:, ix]-1], self.input['points'][1, self.input['parts'][:, ix]-1],
                        '0.8', linewidth=2)

            # Plot original nodes
            plt.plot(self.mesh['nodes'][0, :], self.mesh['nodes'][1, :], 'o', '0.8',
                    linewidth=2, markerfacecolor='None', markersize=5)

            # Plot displaced nodes
            plt.plot(locDisp[0, :], locDisp[1, :], 'or', linewidth=2,
                    markerfacecolor='None', markersize=5, color=color)

            # Set horizontal axis limits
            x_min = min(min(self.input['points'][0, :]), min(locDisp[0, :]))
            x_max = max(max(self.input['points'][0, :]), max(locDisp[0, :]))
            horizontal_span = x_max - x_min
            plt.xlim([
                x_min - horizontal_span*0.05,
                x_max + horizontal_span*0.05
            ])

            # Set vertical axis limits
            y_min = min(min(self.input['points'][1, :]), min(locDisp[1, :]))
            y_max = max(max(self.input['points'][1, :]), max(locDisp[1, :]))
            vertical_span = y_max - y_min
            plt.ylim([
                y_min - vertical_span*0.05,
                y_max + vertical_span*0.05,
            ])

            # Set axis labels
            plt.xticks(np.linspace(x_min, x_max, 5))
            plt.yticks(np.linspace(y_min, y_max, 5))
            plt.xlabel('x location in mm')
            plt.ylabel('y location in mm')

            # Vertical axis label decimal places
            scaled_displacements_x, scaled_displacements_y = abs(self.input['plot_scale']*locDisp3D[0:2, :])
            defmin_x = min(scaled_displacements_x[scaled_displacements_x > 0]) if not all(np.isclose(scaled_displacements_x, 0)) else 1
            defmin_y = min(scaled_displacements_y[scaled_displacements_y > 0]) if not all(np.isclose(scaled_displacements_y, 0)) else 1
            order_of_magnitude = lambda n: math.floor(math.log(n, 10))
            decimals_required = lambda n: abs(order_of_magnitude(n)) + 2 if order_of_magnitude(n) < 0 else 0
            plt.gca().xaxis.set_major_formatter(
                FormatStrFormatter(f'%.{decimals_required(defmin_x)}f')
            )
            plt.gca().yaxis.set_major_formatter(
                FormatStrFormatter(f'%.{decimals_required(defmin_y)}f')
            )

            # adjust horizontal label alignment
            plt.xticks(rotation=30, ha='right')


            # Grid
            plt.grid(True)

            # Resize
            plt.tight_layout()


            if show:
                plt.show()





    def save_output(self, data_print, data_name) -> None:
        '''Save output
        INPUTS:
            Boolean value to toggle saved data printing, and the name of the saved data
        OUTPUTS:
            None
        '''

        ## save object into datasaves folder
        filename = "datasaves/" + self.input['name']
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

        ## retrieve object from datasaves folder, plot output
        if data_print == True:
            with (open("datasaves/" + data_name, 'rb') as file):
                data = pickle.load(file)

            data.plot_output()
            for key in data.output:
                print(key, ' --- ', (data.output)[key])
        pass