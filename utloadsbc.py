import main
from main import MechRes2D
from telescope import read_input, create_simulation_input
from materials import read_materials
import numpy as np
import matplotlib.pyplot as plt

def run_sim(overrides=None, inspect=False):
    if overrides is None: # Just override instead of the previous manipulation attempt
        overrides = {}

    # Read raw input files
    name, geometry, sections, part_properties, forces, bc, output_settings = read_input('telescope.toml')
    materials = read_materials('materials.toml')
    geometry['type'] = 'placeholder'

    # Apply overrides to the raw dictionaries BEFORE creating the mesh
    for key, value in overrides.items():
        if key in geometry:
            geometry[key] = value
        elif key in output_settings:
            output_settings[key] = value
        elif key == 'forces':
            forces = value
        elif key == 'bc':
            bc = value

    # Generate the simulation input dictionary
    inp = create_simulation_input(
        name=name,
        **geometry,
        materials=materials,
        sections=sections,
        part_materials=part_properties['materials'],
        part_sections=part_properties['sections'],
        forces=forces,
        bc=bc,
        **output_settings
    )

    # Run the simulation
    main.PLOT = False
    plt.figure()
    result = MechRes2D(inp)

    # Process outputs for inspection because I need to understand what we have available
    outputs = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    inputs = {k: v for k, v in inp.items() if isinstance(v, np.ndarray)}

    if inspect:
        print('Outputs')
        for n, matrix in outputs.items():
            print(f"{n}: {matrix.shape}")
        print('Inputs')
        for n, matrix in inputs.items():
            print(f"{n}: {matrix.shape}")
    # inp is the actual input data!!! inputs belongs to inspection
    return inputs, outputs, result, inp


def test_total_dof_count(seeds_to_test=[0, 2, 8]):
    # Test to check that the correct number of DOFs are present in the assembly
    for type_val in [0, 1]:
        for seeds in seeds_to_test:
            if type_val == 1:
                n_points = 5
                n_parts = 4
                n_dof_per_node = 3  # Beams keep rotation
                label = "Beam"
                expected_nodes = n_points + (n_parts * seeds)
            else:
                n_points = 7
                n_dof_per_node = 2
                label = "Rod"
                expected_nodes = n_points

            expected_dofs = expected_nodes * n_dof_per_node

            overrides = {
                'type': type_val,
                'seeds': seeds,
                'sym': 0,
                'deltaT': 0
            }

            _, outputs, _, result = run_sim(overrides=overrides)

            active_count = len(outputs['activeDF'])
            inactive_count = len(outputs['inactiveDF'])
            total_count = active_count + inactive_count

            assert total_count == expected_dofs, \
                f"Fail: {label} with {seeds} seeds. Expected {expected_dofs} DOFs, got {total_count}."

            print(f"Pass: {label} DOF count (Seeds={seeds}). Total DOFs: {total_count} ({expected_nodes} nodes).")

    return None

def test_sym_duplicate_force_vert(type_val): #UT5.1.3 as stated on plan
    # Test to determine whether an applied force on node 1 (center) is equal to itself and not duplicated

    overrides = {'type': type_val, 'seeds': 0, 'bc': [{'point': 5 if type_val == 1 else 7, 'DOF': [1, 1, 1]}],
                 'forces': [{'point': 1, 'axis': 1, 'magnitude': -50000}], 'sym': 0, 'deltaT': 0,}

    _, _, result, inp = run_sim(overrides=overrides)

    before = inp['P'].flatten()
    after = result['P'].flatten()
    assert before[1] == after[1], f"Fail: Symmetry duplicates load on center node for type {type_val}!"
    print(f"Pass: Symmetry duplicate vertical force test passed for type {type_val}.")

    return None


def test_symmetry_deflection_ratio(type_val): # Wasn't on plan but very important to check
    # Test to verify that Node 1 vertical deflection is doubled in a half-model (sym=1)
    # compared to a full-model (sym=0) when the same center force is applied.

    overrides = {'type': type_val, 'seeds': 0, 'bc': [{'point': 5 if type_val == 1 else 7, 'DOF': [1, 1, 1]}],
                 'forces': [{'point': 1, 'axis': 1, 'magnitude': -50000}], 'sym': 1, 'deltaT': 0,}

    # Full Model
    _, _, res_full, _ = run_sim(overrides=overrides)
    uy_full = res_full['U'].flatten()[1]

    # Half Model (Symmetry)
    overrides['sym'] = 0
    _, _, res_sym, _ = run_sim(overrides=overrides)
    uy_sym = res_sym['U'].flatten()[1]

    # The half-model has half the stiffness, so deflection should be exactly double.
    ratio = uy_sym / uy_full
    assert np.allclose(ratio, 2.0, rtol=1e-5), f"Fail: Expected ratio 2.0, got {ratio:.4f} for type {type_val}"

    print(f"Pass: Symmetry deflection ratio test passed for type {type_val} (Ratio: {ratio:.2f}).")

    return None


def test_symmetry_boundarycondition(type_val):
    # Test to determine whether the middle node can move horizontally/rotate (it should not!)

    overrides = {
        'type': type_val,
        'deltaT': 0,
        'seeds': 8,
        'sym': 0,
        'forces': [{'point': 1, 'axis': 0, 'magnitude': -50000000}]
    }

    _, _, result, inp = run_sim(overrides=overrides)

    deflection = result['U'].flatten()
    #print(deflection)
    assert deflection[0] == 0 and deflection[2] == 0, f"Fail: Center node moved in the x-direction by: {deflection[0]} and rotated by {deflection[2]}!"
    print(f"Pass: Symmetry boundary condition test passed for type {type_val}")

    return None

def test_nosym_singular_force(axis, type_val):
    # Test to determine whether an applied force on node 2 (not on axis of sym) and check its mirror for no force
    magnitude = -50000
    overrides = {
        'type': type_val,
        'deltaT': 0,
        'seeds': 8,
        'sym': 1,  # 1 builds the full symmetrical telescope
        'forces': [{'point': 2, 'axis': axis, 'magnitude': magnitude}]  # Force on node 2 only
    }

    _, _, result, inp = run_sim(overrides=overrides)

    n_dof = 3 if type_val == 1 else 2 # Rods have 2 DOFs!

    P = result['P'].flatten()

    node_2_idx = n_dof * (2 - 1) + axis

    assert P[node_2_idx] == magnitude, "Fail: Force was not properly applied to Node 2!"

    # Determine the node number of the mirrored node 2
    l_base = 5 if type_val == 1 else 7
    mirrored_node = l_base + 1
    mirrored_y_dof = 3 * (mirrored_node - 1) + axis

    assert P[mirrored_y_dof] == 0, f"Fail: Force duplicated to mirrored node {mirrored_node} in axis {axis}!"

    print(f"Pass: Singular force test passed for type {type_val}, axis {axis}.")
    #print(mirrored_y_dof, mirrored_node, node_2_idx)
    return None

def test_bc_mirroring(type_val):
    # Test to verify that fixed BCs are mirrored when full model is used (unlike forces)
    base = 5 if type_val == 1 else 7
    n_dof = 3 if type_val == 1 else 2

    overrides = {
        'type': type_val,
        'seeds': 0,
        'deltaT': 0,
        'sym': 1,
        'bc': [{'point': base, 'DOF': [1, 1, 1]}],
        'forces': [{'point': 1, 'axis': 1, 'magnitude': -50000}]
    }

    _, _, result, _ = run_sim(overrides=overrides)
    U = result['U'].flatten()

    original_uy_idx = n_dof * (base - 1) + 1

    total_nodes = len(U) // n_dof
    mirrored_uy_idx = n_dof * (total_nodes - 1) + 1

    uy_original = U[original_uy_idx]
    uy_mirrored = U[mirrored_uy_idx]

    # Verification: Both must be zero
    assert np.isclose(uy_original, 0, atol=1e-15), f"Fail: Original Node {base} moved!"
    assert np.isclose(uy_mirrored, 0, atol=1e-15), \
        f"Fail: Mirror of Node {base} (Index {mirrored_uy_idx}) moved {uy_mirrored}m!"

    print(f"Pass: Fixed BC mirroring verified for Type {type_val}.")
    print(f"Indices {original_uy_idx} and {mirrored_uy_idx} are both fixed.")
    return None

def test_rigid_body_modes(type_val): #UT5.4.1 as stated on plan
    # Test to check for at least 3 rigid body modes for an unconstrained structure which makes it floating in space
    overrides = {
        'type': type_val,
        'seeds': 0,
        'deltaT': 0,
        'sym': 1, # Not really possible for 0 since the symmetry BC is forced upon the structure later
        'bc': []
    }

    _, outputs, _, _ = run_sim(overrides=overrides)
    Kr = outputs['Kr'] # The entire K matrix should be equal to Kr if there are no constraints!!!

    # Perform SVD to check for singular values near zero
    s = np.linalg.svd(Kr, compute_uv=False)

    nullity = np.sum(s < (np.max(s) * 1e-12))

    label = "Beam" if type_val == 1 else "Rod"
    assert nullity >= 3, f"Fail: {label} expected more than 2 RBMs, found {nullity}"
    print(f"Pass: {label} (unconstrained) has exactly {nullity} Rigid Body Modes.")

    return None


def test_overconstrained_stability(): #UT5.4.2 as stated on plan
    # Test to verify that an overconstrained model is stable and solvable (unlike what I said on the plan, oops)
    for type_val in [0, 1]:
        n_points, label = (5, "Beam") if type_val == 1 else (7, "Rod")
        overrides = {
            'type': type_val,
            'seeds': 0,
            'deltaT': 0,
            'sym': 0,
            # Clamp every point except one so that Kr is practically the most constrained it can get w/o being empty
            'bc': [{'point': i + 1, 'DOF': [1, 1, 1]} for i in range(n_points - 1)]
        }

        _, outputs, _, _ = run_sim(overrides=overrides)
        Kr = outputs['Kr']

        eigenvals = np.linalg.eigvals(Kr)
        assert np.all(eigenvals > 1e-6), f"Fail: {label} has near-zero or negative eigenvalues!"

        cond_num = np.linalg.cond(Kr)
        assert cond_num < 1e8, f"Fail: {label} is numerically unstable (Cond: {cond_num:.2e})"

        print(f"Pass: {label} overconstrained model is stable and solvable (Cond: {cond_num:.2e}).")

    return None

def test_all_nodes_clamped_and_loaded(): #UT5.2 as stated on plan
    # Test to verify that clamped nodes have zero displacement regardless of load, for every node
    for type_val in [0, 1]:
        if type_val == 1:
            n_points, n_parts, label = 5, 4, "Beam"
            expected_nodes = n_points + (n_parts * 8)
        else:
            n_points, label = 7, "Rod"
            expected_nodes = n_points

        overrides = {
            'type': type_val,
            'seeds': 8,
            'sym': 0,
            'deltaT': 0,
            # Clamp every node and apply loads on all
            'bc': [{'point': i + 1, 'DOF': [1, 1, 1]} for i in range(n_points)],
            'forces': [{'point': i + 1, 'axis': a, 'magnitude': 1e6}
                       for i in range(n_points) for a in [0, 1]]
        }

        _, _, result, _ = run_sim(overrides=overrides)

        # Displacement must be exactly zero
        assert np.all(result['U'] == 0), f"Fail: {label} moved!"

        print(f"Pass: All {expected_nodes} nodes constrained as expected.")

    return None
# Run the tests
test_nosym_singular_force(0,0)
test_nosym_singular_force(0,1)
test_nosym_singular_force(1,0)
test_nosym_singular_force(1, 1)


test_sym_duplicate_force_vert(0)
test_sym_duplicate_force_vert(1)

test_symmetry_deflection_ratio(0)
test_symmetry_deflection_ratio(1)

test_symmetry_boundarycondition(0)
test_symmetry_boundarycondition(1)

test_total_dof_count()

test_rigid_body_modes(0)
test_rigid_body_modes(1)

test_overconstrained_stability()

test_all_nodes_clamped_and_loaded()

test_bc_mirroring(0)
test_bc_mirroring(1)