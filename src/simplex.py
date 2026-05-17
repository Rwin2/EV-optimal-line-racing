"""
Simplex algorithm for solving linear programs.

Implements the tableau-form simplex from:
  Kochenderfer & Wheeler, Algorithms for Optimization, 2nd ed., Chapter 12
    - Section 12.1: LP standard/equality form (eq 12.7–12.10)
    - Section 12.2: Simplex Algorithm (Alg 12.1–12.5)
    - Section 12.2.3: Optimization Phase — pivoting (eq 12.21–12.32)
    - Section 12.2.4: Initialization Phase — auxiliary LP (eq 12.33–12.34)

Uses the simplex tableau for O(mn) per pivot instead of O(m^2 n).
"""

import numpy as np


def _simplex_tableau(c, A, b, max_iters=50000):
    """
    Solve LP in standard form via the simplex tableau method:

        min  c^T x
        s.t. Ax = b,  x >= 0

    Assumes b >= 0 and A contains slack columns that form an identity
    block, giving an immediate feasible basis (no Phase I needed).

    The tableau is:
        [  A   | b ]      rows 0..m-1: constraints
        [ c^T  | 0 ]      row m: objective (negative reduced costs)

    Pivoting applies Gaussian elimination to maintain the tableau.

    Returns x_opt (n,).
    """
    m, n = A.shape

    # Build tableau  (m+1) x (n+1)
    T = np.zeros((m + 1, n + 1))
    T[:m, :n] = A
    T[:m, n] = b
    T[m, :n] = c

    # Identify initial basis (slack columns = unit vectors)
    basis = np.full(m, -1, dtype=int)
    for j in range(n):
        col = T[:m, j]
        nz = np.nonzero(np.abs(col) > 1e-12)[0]
        if len(nz) == 1 and abs(col[nz[0]] - 1.0) < 1e-12:
            if basis[nz[0]] == -1:
                basis[nz[0]] = j

    if np.any(basis < 0):
        raise RuntimeError("Cannot find slack basis — need Phase I")

    # Make sure objective row has zeros in basis columns
    for i in range(m):
        j = basis[i]
        if abs(T[m, j]) > 1e-12:
            T[m] -= T[m, j] * T[i]

    # Simplex iterations
    for _ in range(max_iters):
        # Dantzig's rule: entering variable = most negative reduced cost
        rc = T[m, :n]
        q = np.argmin(rc)
        if rc[q] >= -1e-10:
            break  # optimal

        # Minimum ratio test (eq 12.24)
        col = T[:m, q]
        rhs = T[:m, n]
        ratios = np.full(m, np.inf)
        pos = col > 1e-12
        ratios[pos] = rhs[pos] / col[pos]

        p = np.argmin(ratios)
        if ratios[p] == np.inf:
            break  # unbounded

        # Pivot: make T[p, q] = 1, zero out column q elsewhere
        T[p] /= T[p, q]
        for i in range(m + 1):
            if i != p and abs(T[i, q]) > 1e-14:
                T[i] -= T[i, q] * T[p]

        basis[p] = q

    # Extract solution
    x = np.zeros(n)
    for i in range(m):
        x[basis[i]] = T[i, n]
    return x


def _simplex_two_phase(c, A, b, max_iters=50000):
    """
    Two-phase simplex (Alg 12.5):
      Phase I:  solve auxiliary LP to find feasible basis
      Phase II: optimize original objective from that basis

    Uses a single extended tableau for both phases.
    """
    m, n = A.shape

    # Make b >= 0 by flipping rows
    for i in range(m):
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]

    # Phase I: build tableau for  min 1^T z  s.t. [A I][x;z] = b
    # Tableau: (m+1) x (n+m+1)
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n+m] = np.eye(m)
    T[:m, -1] = b
    T[m, n:n+m] = 1.0   # objective: min sum(z)

    basis = np.arange(n, n + m, dtype=int)

    # Zero out objective row for basis columns
    for i in range(m):
        T[m] -= T[i]

    # Phase I pivots
    for _ in range(max_iters):
        rc = T[m, :n+m]
        q = np.argmin(rc)
        if rc[q] >= -1e-10:
            break
        col = T[:m, q]
        rhs = T[:m, -1]
        ratios = np.full(m, np.inf)
        pos = col > 1e-12
        ratios[pos] = rhs[pos] / col[pos]
        p = np.argmin(ratios)
        if ratios[p] == np.inf:
            break
        T[p] /= T[p, q]
        for i in range(m + 1):
            if i != p and abs(T[i, q]) > 1e-14:
                T[i] -= T[i, q] * T[p]
        basis[p] = q

    # Check feasibility
    if T[m, -1] > 1e-6:
        raise RuntimeError(f"LP infeasible (Phase I obj = {T[m,-1]:.6f})")

    # Phase II: rebuild objective row with original c, keep basis from Phase I
    T2 = np.zeros((m + 1, n + 1))
    T2[:m, :n] = T[:m, :n]   # constraint rows (only x columns)
    T2[:m, n] = T[:m, -1]     # RHS
    T2[m, :n] = c

    # Zero out objective in basis columns
    for i in range(m):
        j = basis[i]
        if j < n and abs(T2[m, j]) > 1e-14:
            T2[m] -= T2[m, j] * T2[i]

    # Phase II pivots
    for _ in range(max_iters):
        rc = T2[m, :n]
        q = np.argmin(rc)
        if rc[q] >= -1e-10:
            break
        col = T2[:m, q]
        rhs = T2[:m, n]
        ratios = np.full(m, np.inf)
        pos = col > 1e-12
        ratios[pos] = rhs[pos] / col[pos]
        p = np.argmin(ratios)
        if ratios[p] == np.inf:
            break
        T2[p] /= T2[p, q]
        for i in range(m + 1):
            if i != p and abs(T2[i, q]) > 1e-14:
                T2[i] -= T2[i, q] * T2[p]
        basis[p] = q

    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T2[i, n]
    return x


def solve_lp(c_obj, A_ub, b_ub, lb, ub):
    """
    Solve a general-form LP with the simplex algorithm (Ch 12):

        min  c_obj^T x
        s.t. A_ub @ x <= b_ub
             lb <= x <= ub

    Converts to equality form (eq 12.8–12.10) via:
      1. Variable shift:  y = x - lb  so  y >= 0
      2. Slack variables for inequality constraints:  A y + s1 = b'
      3. Slack variables for upper bounds:  y + s2 = ub'

    The slack variables [s1, s2] form an identity block in A_eq, giving
    an immediate feasible basis when b' >= 0 and ub' >= 0 (always true
    for ub' = ub - lb > 0; usually true for b' in the SCP context).

    Returns x_opt in the original variable space.
    """
    n = len(c_obj)
    m = len(b_ub)

    # Shift: y = x - lb >= 0,  y <= ub - lb
    ub_s = ub - lb
    b_s = b_ub - A_ub @ lb

    # Equality form with slacks
    n_total = n + m + n
    m_total = m + n

    A_eq = np.zeros((m_total, n_total))
    A_eq[:m, :n] = A_ub
    A_eq[:m, n:n+m] = np.eye(m)
    A_eq[m:, :n] = np.eye(n)
    A_eq[m:, n+m:] = np.eye(n)

    b_eq = np.concatenate([b_s, ub_s])
    c_eq = np.concatenate([c_obj, np.zeros(m + n)])

    if np.all(b_eq >= -1e-12):
        b_eq = np.maximum(b_eq, 0.0)
        x_full = _simplex_tableau(c_eq, A_eq, b_eq)
    else:
        x_full = _simplex_two_phase(c_eq, A_eq.copy(), b_eq.copy())

    return x_full[:n] + lb
