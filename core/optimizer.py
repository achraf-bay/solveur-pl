import math
from typing import List, Tuple, Dict, Any

import numpy as np
from itertools import combinations
from scipy.optimize import linprog

def _to_float(x: Any):
    try:
        return float(x)
    except Exception:
        return x

def _to_tuple_float(pt) -> Tuple[float, float]:
    return (float(pt[0]), float(pt[1]))


class LinearProgrammingOptimizer:
    def __init__(self, c: List[float], A: List[List[float]], b: List[float],
                 operators: List[str], objective_type: str = 'max'): 
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float) if A is not None else np.zeros((0, 2), dtype=float)
        self.b = np.array(b, dtype=float) if b is not None else np.zeros((0,), dtype=float)
        self.operators = list(operators or [])
        self.objective_type = (objective_type or 'max').lower()

    def prepare_for_scipy(self):
        c_scipy = -self.c if self.objective_type == 'max' else self.c

        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        for i, op in enumerate(self.operators):
            if op == '<=':
                A_ub.append(self.A[i]); b_ub.append(self.b[i])
            elif op == '>=':
                A_ub.append(-self.A[i]); b_ub.append(-self.b[i])
            elif op == '=':
                A_eq.append(self.A[i]); b_eq.append(self.b[i])

        A_ub = np.array(A_ub, dtype=float) if len(A_ub) else None
        b_ub = np.array(b_ub, dtype=float) if len(b_ub) else None
        A_eq = np.array(A_eq, dtype=float) if len(A_eq) else None
        b_eq = np.array(b_eq, dtype=float) if len(b_eq) else None

        bounds = [(0, None), (0, None)]
        return c_scipy, A_ub, b_ub, A_eq, b_eq, bounds

    def _build_all_ineq_with_nonneg(self) -> Tuple[np.ndarray, np.ndarray]: 
        A_all = []
        b_all = []
        for i, op in enumerate(self.operators):
            if op == '<=':
                A_all.append(self.A[i]); b_all.append(self.b[i])
            elif op == '>=':
                A_all.append(-self.A[i]); b_all.append(-self.b[i])
            elif op == '=':
                A_all.append(self.A[i]); b_all.append(self.b[i])
                A_all.append(-self.A[i]); b_all.append(-self.b[i])

        A_all.extend([[-1.0, 0.0], [0.0, -1.0]])
        b_all.extend([0.0, 0.0])

        return np.array(A_all, dtype=float), np.array(b_all, dtype=float)
        
    def find_extreme_points_manual(self, eps_axis=1e-8, eps_feas=1e-8, eps_det=1e-12) -> List[Tuple[float, float]]: 
        A_all, b_all = self._build_all_ineq_with_nonneg()
        extreme_points: List[Tuple[float, float]] = []
        n = len(A_all)

        for i, j in combinations(range(n), 2):
            A_sys = np.array([A_all[i], A_all[j]], dtype=float)
            b_sys = np.array([b_all[i], b_all[j]], dtype=float)

            if abs(np.linalg.det(A_sys)) < eps_det:
                continue
            try:
                x = np.linalg.solve(A_sys, b_sys)
            except Exception:
                continue

            if x[0] < -eps_axis or x[1] < -eps_axis:
                continue
            if np.any(A_all @ x > b_all + eps_feas):
                continue

            if not any(abs(px - x[0]) <= 1e-7 and abs(py - x[1]) <= 1e-7 for (px, py) in extreme_points):
                extreme_points.append((float(x[0]), float(x[1])))

        if extreme_points:
            ctr = np.mean(np.array(extreme_points), axis=0)
            extreme_points.sort(key=lambda p: math.atan2(p[1] - ctr[1], p[0] - ctr[0]))

        return extreme_points

    def evaluate_objective(self, point: Tuple[float, float]) -> float:
        return float(self.c[0] * point[0] + self.c[1] * point[1])

    def check_multiple_solutions(self, x_opt: Tuple[float, float], extreme_points, atol=1e-3, rtol=1e-6):
        z_star = self.evaluate_objective(x_opt)
        optimal_points = []
        for p in extreme_points:
            if abs(self.evaluate_objective(p) - z_star) <= (atol + rtol * max(1.0, abs(z_star))):
                optimal_points.append(p)

        if len(optimal_points) > 2: 
            pts = np.array(optimal_points, dtype=float)
            dmax = -1.0; ii=jj=0
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = np.linalg.norm(pts[i]-pts[j])
                    if d > dmax:
                        dmax = d; ii, jj = i, j
            optimal_points = [tuple(pts[ii]), tuple(pts[jj])]
        return optimal_points

    def reconstruct_optimal_edge_points(self, z_star: float, tol_val=1e-3) -> List[Tuple[float, float]]: 
        A_all, b_all = self._build_all_ineq_with_nonneg()

        def feasible(x):
            return x[0] >= -1e-8 and x[1] >= -1e-8 and np.all(A_all @ x <= b_all + 1e-8)

        cand: List[Tuple[float, float]] = []
        n = len(A_all)
        for i in range(n):
            for j in range(i+1, n):
                M = np.vstack([A_all[i], A_all[j]])
                if abs(np.linalg.det(M)) < 1e-12:
                    continue
                try:
                    x = np.linalg.solve(M, np.array([b_all[i], b_all[j]]))
                except Exception:
                    continue
                if feasible(x) and abs(self.evaluate_objective((x[0], x[1])) - z_star) <= tol_val:
                    cand.append((float(x[0]), float(x[1])))

        uniq: List[Tuple[float, float]] = []
        for p in cand:
            if not any(abs(p[0]-q[0])<=1e-7 and abs(p[1]-q[1])<=1e-7 for q in uniq):
                uniq.append(p)

        if len(uniq) <= 1:
            return uniq

        pts = np.array(uniq, dtype=float)
        dmax = -1.0; ii=jj=0
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i]-pts[j])
                if d > dmax:
                    dmax = d; ii, jj = i, j
        return [tuple(pts[ii]), tuple(pts[jj])]

    def detect_region_boundedness_via_aux_lp(self) -> bool: 
        _, A_ub, b_ub, A_eq, b_eq, bounds = self.prepare_for_scipy()
        if A_ub is None:
            A_ub = np.zeros((0, 2)); b_ub = np.zeros((0,))
        if A_eq is None:
            A_eq = np.zeros((0, 2)); b_eq = np.zeros((0,))

        res_x1 = linprog(c=np.array([-1.0, 0.0]), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        res_x2 = linprog(c=np.array([0.0, -1.0]), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res_x1.status == 3 or res_x2.status == 3:
            return False
        return True

    def detect_recession_direction(self, x_opt=None, tol=1e-6): 
        _, A_ub, _, A_eq, _, _ = self.prepare_for_scipy()
        if A_ub is None:
            A_ub = np.zeros((0, 2))
        if A_eq is None:
            A_eq = np.zeros((0, 2))

        
        b_ub_zero = np.zeros(A_ub.shape[0]) if A_ub.shape[0] > 0 else np.zeros(0)
        b_eq_zero = np.zeros(A_eq.shape[0]) if A_eq.shape[0] > 0 else np.zeros(0)
        
        candidates = [
            np.array([1.0, 0.0]),  
            np.array([0.0, 1.0]),   
            np.array([1.0, 1.0]),  
            np.array([1.0, 2.0]),   
            np.array([2.0, 1.0]),   
        ]

        if x_opt is not None: 
            c_perp = np.array([-self.c[1], self.c[0]])
            if np.linalg.norm(c_perp) > 1e-6:
                candidates.append(c_perp / np.linalg.norm(c_perp))

        for d_test in candidates: 
            if np.linalg.norm(d_test) < 1e-10:
                continue
            d_test = d_test / np.linalg.norm(d_test)
            
            # Vérifier si c'est une direction de récession 
            if A_ub.shape[0] > 0:
                violations = A_ub @ d_test
                if np.any(violations > 1e-6):
                    continue

            if A_eq.shape[0] > 0:
                eq_violations = np.abs(A_eq @ d_test)
                if np.any(eq_violations > 1e-6):
                    continue
            
            if x_opt is not None:
                test_point = np.array(x_opt) + 10 * d_test 
                if test_point[0] < -1e-6 or test_point[1] < -1e-6:
                    continue
                # Vérifier les contraintes
                if A_ub.shape[0] > 0:
                    ub_vals = A_ub @ test_point
                    b_vals = np.array([self.b[i] for i, op in enumerate(self.operators) if op in ['<=', '>=']])
                    if len(b_vals) > 0 and np.any(ub_vals > b_vals + 1e-6):
                        continue
            
            c_dot_d = np.dot(self.c, d_test)
            if abs(c_dot_d) > tol:
                continue 
            return d_test, float(c_dot_d) 
        return None, None 
    def optimize(self) -> Dict:
        c_scipy, A_ub, b_ub, A_eq, b_eq, bounds = self.prepare_for_scipy()
        res = linprog(c=c_scipy, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        extreme_points = self.find_extreme_points_manual()
        evaluations = [(p, self.evaluate_objective(p)) for p in extreme_points]

        # infeasible
        if res.status == 2:
            return {
                'success': False, 'status': 'infeasible', 'x': [None, None], 'z': None,
                'message': '❌ RÉGION ADMISSIBLE VIDE',
                'extreme_points': [], 'optimal_points': [], 'solution_type': 'no_solution',
                'region_bounded': False, 'objective_type': self.objective_type, 'all_evaluations': []
            }

        # pas de solution finie
        if res.status == 3:
            return {
                'success': False, 'status': 'unbounded', 'x': [None, None], 'z': None,
                'message': '❌ SOLUTION NON BORNÉE',
                'extreme_points': extreme_points, 'optimal_points': [], 'solution_type': 'unbounded_no_finite',
                'region_bounded': False, 'objective_type': self.objective_type, 'all_evaluations': []
            }

        if res.status != 0:
            return {
                'success': False, 'status': 'error', 'x': [None, None], 'z': None,
                'message': f'❌ ERREUR : {res.message}',
                'extreme_points': extreme_points, 'optimal_points': [],
                'solution_type': 'error', 'region_bounded': False,
                'objective_type': self.objective_type, 'all_evaluations': evaluations
            }

        # Optimal fini trouvé par scipy
        x_star = (float(res.x[0]), float(res.x[1]))
        z_star = self.evaluate_objective(x_star) 

        # Vérifier si région bornée
        region_bounded = self.detect_region_boundedness_via_aux_lp()
        
        # Chercher direction de récession (en passant x_star)
        d, dir_val = self.detect_recession_direction(x_opt=x_star, tol=1e-5)
        recession_direction = None
        if d is not None:
            recession_direction = (float(d[0]), float(d[1])) 

        optimal_points = self.check_multiple_solutions(x_star, extreme_points, atol=1e-3, rtol=1e-6)

        if len(optimal_points) < 2:
            rebuilt = self.reconstruct_optimal_edge_points(z_star, tol_val=1e-3)
            if len(rebuilt) >= 2:
                optimal_points = rebuilt 


        
        # CAS 1: Région non bornée + direction de récession
        if not region_bounded and recession_direction is not None: 
            return {
                'success': True, 'status': 'optimal', 'x': [x_star[0], x_star[1]], 'z': z_star,
                'message': '✅ INFINITÉ DE SOLUTIONS (rayon optimal)',
                'extreme_points': [tuple(map(float, p)) for p in extreme_points],
                'optimal_point': x_star, 'optimal_value': z_star,
                'optimal_points': [x_star] if len(optimal_points) < 2 else [tuple(map(float, p)) for p in optimal_points],
                'objective_type': self.objective_type,
                'all_evaluations': [(tuple(map(float, p)), float(v)) for p, v in evaluations],
                'solution_type': 'infinite_edge',
                'region_bounded': False,
                'recession_direction': recession_direction
            }

        # CAS 2: région bornée, plusieurs points optimaux
        if len(optimal_points) >= 2:
            return {
                'success': True, 'status': 'optimal', 'x': [x_star[0], x_star[1]], 'z': z_star,
                'message': '✅ INFINITÉ DE SOLUTIONS (arête optimale)',
                'extreme_points': [tuple(map(float, p)) for p in extreme_points],
                'optimal_point': x_star, 'optimal_value': z_star,
                'optimal_points': [tuple(map(float, p)) for p in optimal_points],
                'objective_type': self.objective_type,
                'all_evaluations': [(tuple(map(float, p)), float(v)) for p, v in evaluations],
                'solution_type': 'infinite_edge',
                'region_bounded': bool(region_bounded),
                'recession_direction': recession_direction
            }

        # CAS 3: Solution unique
        return {
            'success': True, 'status': 'optimal', 'x': [x_star[0], x_star[1]], 'z': z_star,
            'message': '✅ OPTIMUM UNIQUE',
            'extreme_points': [tuple(map(float, p)) for p in extreme_points],
            'optimal_point': x_star, 'optimal_value': z_star,
            'optimal_points': [x_star],
            'objective_type': self.objective_type,
            'all_evaluations': [(tuple(map(float, p)), float(v)) for p, v in evaluations],
            'solution_type': 'unique',
            'region_bounded': bool(region_bounded),
            'recession_direction': recession_direction
        }


def solve_linear_program(c: List[float], A: List[List[float]], b: List[float],
                        operators: List[str], objective_type: str = 'max') -> Dict:
    optimizer = LinearProgrammingOptimizer(c, A, b, operators, objective_type)
    return optimizer.optimize()
