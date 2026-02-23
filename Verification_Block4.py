import numpy as np
from objects.assembly import Assembly


def test_shared_node_additivity_K_M_Q(explain=False):
    if explain:
        print("\n=== Test 1: shared-node additivity + size (K/M/Q) ===")

    inp = {
        "name": "test-shared",
        "mat": [{"E": 1, "rho": 1, "alpha": 0}],
        "parts": np.array([[1], [2]]),  # dummy, only used for loop length
    }

    assembly = Assembly(inp)

    # fake element matrices/vectors (easy to track)
    K1 = np.ones((6, 6))
    K2 = 2 * np.ones((6, 6))
    M1 = 3 * np.ones((6, 6))
    M2 = 4 * np.ones((6, 6))
    Q1 = np.ones((6, 1))
    Q2 = 2 * np.ones((6, 1))

    # two elements share node 2
    assembly.mesh = {
        "nNodes": 3,
        "part": [{"elementNumbers": [1, 2]}],
        "element": [
            {"nodeNumber1": 1, "nodeNumber2": 2, "K": K1, "m": M1, "Q": Q1},
            {"nodeNumber1": 2, "nodeNumber2": 3, "K": K2, "m": M2, "Q": Q2},
        ],
    }

    ndof = 3 * assembly.mesh["nNodes"]  # 9

    assembly.global_stiffness_matrix()
    K = assembly.mesh["K"]
    assert K.shape == (ndof, ndof)
    assert np.all(K[3:6, 3:6] == 3)  # 1 + 2

    assembly.global_mass_matrix()
    M = assembly.mesh["m"]
    assert M.shape == (ndof, ndof)
    assert np.all(M[3:6, 3:6] == 7)  # 3 + 4

    assembly.global_thermal_load_vector()
    Q = assembly.output["Q"]
    assert Q.shape == (ndof, 1)
    assert np.all(Q[3:6] == 3)  # 1 + 2

    if explain:
        print("K shared block:\n", K[3:6, 3:6])
        print("M shared block:\n", M[3:6, 3:6])
        print("Q shared entries:", Q[3:6].reshape(-1))
        print("Test 1 passed")


def test_nonsequential_numbering_K_M_Q(explain=False):
    if explain:
        print("\n=== Test 2: non-sequential numbering (1 -> 3) ===")

    inp = {
        "name": "test-nonsequential",
        "mat": [{"E": 1, "rho": 1, "alpha": 0}],
        "parts": np.array([[1], [2]]),  # dummy
    }

    assembly = Assembly(inp)

    # unique patterns so we can check exact placement
    K13 = np.arange(1, 37, dtype=float).reshape(6, 6)           # 1..36
    M13 = 1000 + np.arange(1, 37, dtype=float).reshape(6, 6)    # 1001..1036
    Q13 = np.arange(1, 7, dtype=float).reshape(6, 1)            # 1..6

    assembly.mesh = {
        "nNodes": 3,
        "part": [{"elementNumbers": [1]}],
        "element": [
            {"nodeNumber1": 1, "nodeNumber2": 3, "K": K13, "m": M13, "Q": Q13},
        ],
    }

    ndof = 3 * assembly.mesh["nNodes"]  # 9
    b = np.array([0, 1, 2, 6, 7, 8])    # DOFs for node 1 and node 3

    assembly.global_stiffness_matrix()
    K = assembly.mesh["K"]
    assert K.shape == (ndof, ndof)
    assert np.allclose(K[np.ix_(b, b)], K13)
    assert np.allclose(K[3:6, :], 0)  # node 2 rows untouched
    assert np.allclose(K[:, 3:6], 0)  # node 2 cols untouched

    assembly.global_mass_matrix()
    M = assembly.mesh["m"]
    assert M.shape == (ndof, ndof)
    assert np.allclose(M[np.ix_(b, b)], M13)
    assert np.allclose(M[3:6, :], 0)
    assert np.allclose(M[:, 3:6], 0)

    assembly.global_thermal_load_vector()
    Q = assembly.output["Q"]
    assert Q.shape == (ndof, 1)

    expected_Q = np.zeros((ndof, 1))
    expected_Q[b, 0] = Q13[:, 0]
    assert np.allclose(Q, expected_Q)

    if explain:
        print("bounds b:", b.tolist())
        print("node 2 rows of K (should be all zeros):\n", K[3:6, :])
        print("Q (1D):", Q.reshape(-1))
        print("Test 2 passed")


def test_order_invariance_K_M_Q(explain=False):
    if explain:
        print("\n=== Test 3: order invariance (elementNumbers [1,2] vs [2,1]) ===")

    inp = {
        "name": "test-order",
        "mat": [{"E": 1, "rho": 1, "alpha": 0}],
        "parts": np.array([[1], [2]]),  # dummy
    }

    # fake matrices/vectors
    K1 = np.ones((6, 6))
    K2 = 2 * np.ones((6, 6))
    M1 = 3 * np.ones((6, 6))
    M2 = 4 * np.ones((6, 6))
    Q1 = np.ones((6, 1))
    Q2 = 2 * np.ones((6, 1))

    elements = [
        {"nodeNumber1": 1, "nodeNumber2": 2, "K": K1, "m": M1, "Q": Q1},
        {"nodeNumber1": 2, "nodeNumber2": 3, "K": K2, "m": M2, "Q": Q2},
    ]

    def assemble(element_numbers):
        assembly = Assembly(inp)
        assembly.mesh = {
            "nNodes": 3,
            "part": [{"elementNumbers": element_numbers}],
            "element": elements,
        }
        assembly.global_stiffness_matrix()
        assembly.global_mass_matrix()
        assembly.global_thermal_load_vector()
        return assembly.mesh["K"], assembly.mesh["m"], assembly.output["Q"]

    K_a, M_a, Q_a = assemble([1, 2])
    K_b, M_b, Q_b = assemble([2, 1])

    assert np.allclose(K_a, K_b)
    assert np.allclose(M_a, M_b)
    assert np.allclose(Q_a, Q_b)

    if explain:
        print("Test 3 passed")


if __name__ == "__main__":
    # run all 3 tests with prints
    test_shared_node_additivity_K_M_Q(explain=True)
    test_nonsequential_numbering_K_M_Q(explain=True)
    test_order_invariance_K_M_Q(explain=True)
    print("\nAll three Block-4 tests passed.")
