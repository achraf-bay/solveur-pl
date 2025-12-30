import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch

def create_plot(A, b, solution, c, obj_type, operators=None, optimal_points=None, 
                region_bounded=True, status='optimal', solver_result=None):

    if operators is None:
        operators = ['<='] * len(A)
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    
    # Récupérer la direction de récession depuis solver_result
    recession_direction = None
    if solver_result:
        recession_direction = solver_result.get('recession_direction')
 

    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111)
    
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#ffffff')
    
    # Déterminer les limites du graphique
    try:
        x1_vals = [abs(bi / ai) if abs(ai) > 0.001 else 20 for ai, bi in zip([row[0] for row in A], b)]
        x2_vals = [abs(bi / ai) if abs(ai) > 0.001 else 20 for ai, bi in zip([row[1] for row in A], b)]
        x1_max = max(x1_vals + [10]) * 1.3
        x2_max = max(x2_vals + [10]) * 1.3
    except:
        x1_max, x2_max = 20, 20
    
    x1 = np.linspace(-x1_max*0.1, x1_max, 500)
    
    x1_grid = np.linspace(-x1_max*0.2, x1_max, 400)
    x2_grid = np.linspace(-x2_max*0.2, x2_max, 400)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    def check_constraint(a1, a2, bi, operator, X1, X2):
        lhs = a1 * X1 + a2 * X2
        if operator == '<=':
            return lhs <= bi + 1e-6
        elif operator == '>=':
            return lhs >= bi - 1e-6
        elif operator == '=':
            return np.abs(lhs - bi) < 0.1
        return np.ones_like(X1, dtype=bool)
    
    # Calculer la région faisable
    feasible = np.ones_like(X1, dtype=bool)
    
    for i, (row, bi) in enumerate(zip(A, b)):
        a1, a2 = row
        operator = operators[i] if i < len(operators) else '<='
        constraint_satisfied = check_constraint(a1, a2, bi, operator, X1, X2)
        feasible &= constraint_satisfied
    
    feasible &= (X1 >= 0) & (X2 >= 0)
    
    has_feasible_region = np.any(feasible)
    region_annotation_added = False
    
    # Tracer la région faisable
    if has_feasible_region:
        if status == 'infeasible':
            pass
        elif not region_bounded:
            # Région NON BORNÉE 
            ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5], 
                       colors=['#90EE90'], alpha=0.5, label='_nolegend_')
            ax.contour(X1, X2, feasible.astype(float), levels=[0.5], 
                      colors=['#32CD32'], linewidths=2.5, linestyles='--')
            ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5],
                       colors='none', hatches=['///'], alpha=0.3)
            region_annotation_added = True
        else:
            # Région BORNÉE 
            ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5], 
                       colors=['#4A90E2'], alpha=0.4, label='_nolegend_')
            ax.contour(X1, X2, feasible.astype(float), levels=[0.5], 
                      colors=['#FF6B35'], linewidths=2.5, linestyles='--')
            region_annotation_added = True
    else:
        # CAS SANS RÉGION FAISABLE
        colors_incompatible = ['#FFB3B3', '#B3E0FF', '#FFFFB3', '#D9FFB3', '#FFD9B3']
        
        for i, (row, bi) in enumerate(zip(A, b)):
            a1, a2 = row
            operator = operators[i] if i < len(operators) else '<='
            constraint_mask = check_constraint(a1, a2, bi, operator, X1, X2) & (X1 >= 0) & (X2 >= 0)
            
            if np.any(constraint_mask):
                color = colors_incompatible[i % len(colors_incompatible)]
                ax.contourf(X1, X2, constraint_mask.astype(float), levels=[0.5, 1.5], 
                           colors=[color], alpha=0.3, zorder=1)
                
       
        ax.text(x1_max * 0.75, x2_max * 0.85, 
                '❌ AUCUNE\nSOLUTION', 
                ha='center', va='center',
                fontsize=11, color='#8B0000', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE6E6', 
                         edgecolor='#FF0000', linewidth=2, alpha=0.9),
                zorder=10)
    
    # Tracer les contraintes avec équations
    colors = ['#FF6B35', '#004E89', '#00A896', '#F77F00', '#9B59B6']
    used_positions = []
    equation_labels = []
    
    def is_position_free(x_pos, y_pos, used_positions, min_dist_factor=0.12):
        min_distance = (x1_max + x2_max) * min_dist_factor
        for used_x, used_y in used_positions:
            dist = np.sqrt((x_pos - used_x)**2 + (y_pos - used_y)**2)
            if dist < min_distance:
                return False
        return True
    
    def find_best_label_position(x_vals, y_vals, used_positions, x_max, y_max, constraint_idx):
        n_points = len(x_vals)
        candidates = [
            (n_points // 8, 'top', 25),
            (n_points // 5, 'bottom', -25),
            (n_points // 3, 'top', 25),
            (n_points // 2, 'bottom', -25),
            (2 * n_points // 3, 'top', 25),
            (4 * n_points // 5, 'bottom', -25),
            (7 * n_points // 8, 'top', 25),
            (n_points // 10, 'top', 30),
            (3 * n_points // 10, 'bottom', -30),
            (7 * n_points // 10, 'top', 30),
            (9 * n_points // 10, 'bottom', -30),
        ]
        
        for idx, va, offset_y in candidates:
            if 0 <= idx < len(x_vals) and 0 <= idx < len(y_vals):
                x_pos, y_pos = x_vals[idx], y_vals[idx]
                
                if not (-x_max*0.05 <= x_pos <= x_max * 1.05 and -y_max*0.05 <= y_pos <= y_max * 1.05):
                    continue
                
                if is_position_free(x_pos, y_pos, used_positions):
                    return idx, va, offset_y, x_pos, y_pos
        
        return None, None, None, None, None
    
    # Tracer toutes les contraintes
    for i, (row, bi) in enumerate(zip(A, b)):
        a1, a2 = row
        color = colors[i % len(colors)]
        operator = operators[i] if i < len(operators) else '<='
        
        def format_coef(val):
            if abs(val) < 0.01:
                return "0"
            elif abs(val - round(val)) < 0.01:
                return str(int(round(val)))
            else:
                return f"{val:.1f}"
        
        a1_str = format_coef(a1)
        a2_str = format_coef(abs(a2))
        sign = '+' if a2 >= 0 else '-'
        bi_str = format_coef(bi)
        
        # Afficher l'opérateur correct
        op_display = {'<=': '≤', '>=': '≥', '=': '='}
        equation = f"{a1_str}x₁ {sign} {a2_str}x₂ {op_display.get(operator, operator)} {bi_str}"
        
        if abs(a2) > 0.001:
            x2 = (bi - a1 * x1) / a2
            valid_mask = (x1 >= -x1_max*0.05) & (x2 >= -x2_max*0.1) & (x2 <= x2_max * 1.15)
            x1_valid = x1[valid_mask]
            x2_valid = x2[valid_mask]
            
            if len(x1_valid) > 0:
                ax.plot(x1_valid, x2_valid, linestyle='--', label=f'C{i+1}: {equation}', 
                       linewidth=2.5, color=color, alpha=1.0, zorder=3)
                
                idx, va_pos, offset_y, x_pos, y_pos = find_best_label_position(
                    x1_valid, x2_valid, used_positions, x1_max, x2_max, i)
                
                if idx is not None and x_pos is not None:
                    equation_labels.append({
                        'equation': equation,
                        'x': x_pos,
                        'y': y_pos,
                        'offset_y': offset_y,
                        'va': va_pos,
                        'color': color
                    })
                    used_positions.append((x_pos, y_pos))
                
        elif abs(a1) > 0.001:
            x1_line = bi / a1
            if -x1_max*0.05 <= x1_line <= x1_max * 1.05:
                ax.axvline(x=x1_line, label=f'C{i+1}: {equation}',
                          linewidth=2.5, color=color, linestyle='--', alpha=1.0, zorder=3)
                
                y_positions = [x2_max * 0.15, x2_max * 0.30, x2_max * 0.45, 
                              x2_max * 0.60, x2_max * 0.75, x2_max * 0.90]
                
                for y_pos in y_positions:
                    if is_position_free(x1_line, y_pos, used_positions, min_dist_factor=0.15):
                        ax.text(x1_line + x1_max*0.02, y_pos, equation,
                               fontsize=8, color=color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor=color, linewidth=1.2, alpha=0.95),
                               rotation=90, ha='left', va='center', zorder=5)
                        used_positions.append((x1_line, y_pos))
                        break
    
    # Afficher les équations sur le graphique
    for label_info in equation_labels:
        ax.annotate(label_info['equation'], 
                   xy=(label_info['x'], label_info['y']),
                   xytext=(0, label_info['offset_y']), textcoords='offset points',
                   fontsize=8, color=label_info['color'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=label_info['color'], linewidth=1.2, alpha=0.95),
                   ha='center', va=label_info['va'], zorder=5)
    
    # Annotation de la région
    if region_annotation_added and has_feasible_region:
        feasible_points = np.column_stack([X1[feasible], X2[feasible]])
        if len(feasible_points) > 0:
            center = feasible_points.mean(axis=0)
            
            if 0 <= center[0] <= x1_max and 0 <= center[1] <= x2_max:
                region_positions = [
                    (center[0], center[1], 60, 60),
                    (center[0], center[1], -60, 60),
                    (center[0], center[1], 60, -60),
                    (center[0], center[1], -60, -60),
                    (center[0] + x1_max*0.15, center[1], 40, 40),
                    (center[0] - x1_max*0.15, center[1], -40, 40),
                ]
                
                for cx, cy, ox, oy in region_positions:
                    if is_position_free(cx, cy, used_positions, min_dist_factor=0.15):
                        if not region_bounded:
                            ax.annotate('Région Non Bornée', 
                                       xy=(cx, cy), 
                                       xytext=(ox, oy), textcoords='offset points',
                                       fontsize=10, color='#006400', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#90EE90', 
                                               edgecolor='#32CD32', linewidth=1.5, alpha=0.9),
                                       arrowprops=dict(arrowstyle='->', color='#32CD32', lw=1.5),
                                       ha='center', zorder=15)
                        else:
                            ax.annotate('Région\nAdmissible', 
                                       xy=(cx, cy), 
                                       xytext=(ox, oy), textcoords='offset points',
                                       fontsize=10, color='#000000', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#4A90E2', 
                                               edgecolor='#000000', linewidth=1.5, alpha=0.85),
                                       arrowprops=dict(arrowstyle='->', color='#000000', lw=1.5),
                                       ha='center', zorder=15)
                        used_positions.append((cx, cy))
                        break
    
    # Trouver et marquer les sommets
    if has_feasible_region:
        vertices = []
        n = len(A)
        
        for i in range(n):
            for j in range(i+1, n):
                try:
                    A_sys = np.array([A[i], A[j]])
                    b_sys = np.array([b[i], b[j]])
                    
                    det = np.linalg.det(A_sys)
                    if abs(det) < 1e-10:
                        continue
                    
                    point = np.linalg.solve(A_sys, b_sys)
                    
                    if point[0] < -1e-6 or point[1] < -1e-6:
                        continue
                    
                    if point[0] > x1_max * 1.1 or point[1] > x2_max * 1.1:
                        continue
                    
                    valid = True
                    for k in range(n):
                        op = operators[k] if k < len(operators) else '<='
                        lhs = np.dot(A[k], point)
                        
                        if op == '<=':
                            if lhs > b[k] + 1e-6:
                                valid = False
                                break
                        elif op == '>=':
                            if lhs < b[k] - 1e-6:
                                valid = False
                                break
                        elif op == '=':
                            if abs(lhs - b[k]) > 1e-6:
                                valid = False
                                break
                    
                    if valid:
                        vertices.append(point)
                except:
                    pass
        
        # Intersections avec les axes
        for i in range(n):
            if abs(A[i][1]) > 0.001:
                x2_int = b[i] / A[i][1]
                if 0 <= x2_int <= x2_max * 1.1:
                    point = np.array([0, x2_int])
                    valid = True
                    for k in range(n):
                        op_k = operators[k] if k < len(operators) else '<='
                        lhs = np.dot(A[k], point)
                        if op_k == '<=' and lhs > b[k] + 1e-6:
                            valid = False
                            break
                        elif op_k == '>=' and lhs < b[k] - 1e-6:
                            valid = False
                            break
                        elif op_k == '=' and abs(lhs - b[k]) > 1e-6:
                            valid = False
                            break
                    if valid:
                        vertices.append(point)
            
            if abs(A[i][0]) > 0.001:
                x1_int = b[i] / A[i][0]
                if 0 <= x1_int <= x1_max * 1.1:
                    point = np.array([x1_int, 0])
                    valid = True
                    for k in range(n):
                        op_k = operators[k] if k < len(operators) else '<='
                        lhs = np.dot(A[k], point)
                        if op_k == '<=' and lhs > b[k] + 1e-6:
                            valid = False
                            break
                        elif op_k == '>=' and lhs < b[k] - 1e-6:
                            valid = False
                            break
                        elif op_k == '=' and abs(lhs - b[k]) > 1e-6:
                            valid = False
                            break
                    if valid:
                        vertices.append(point)
        
        # Vérifier l'origine
        origin = np.array([0, 0])
        valid_origin = True
        for k in range(n):
            op_k = operators[k] if k < len(operators) else '<='
            lhs = np.dot(A[k], origin)
            if op_k == '<=' and lhs > b[k] + 1e-6:
                valid_origin = False
                break
            elif op_k == '>=' and lhs < b[k] - 1e-6:
                valid_origin = False
                break
            elif op_k == '=' and abs(lhs - b[k]) > 1e-6:
                valid_origin = False
                break
        if valid_origin:
            vertices.append(origin)
        
        if len(vertices) > 0:
            vertices = np.array(vertices)
            vertices = np.unique(vertices.round(decimals=6), axis=0)
            
            if len(vertices) > 0:
                ax.plot(vertices[:, 0], vertices[:, 1], 'o', 
                       color='#004E89', markersize=12, zorder=4,
                       label='Sommets', markeredgecolor='black', markeredgewidth=2)
    
    #  TRACER LA SOLUTION OPTIMALE 
    if solution is not None and len(solution) >= 2 and solution[0] is not None:

        # CAS 1: région non bornée + direction de récession
        if not region_bounded and recession_direction is not None:
 
            
            # Point de départ du demi-droite 
            start_point = np.array(solution)
            direction = np.array(recession_direction)
            
            # Normaliser la direction
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
                
                # Calculer le point final du rayon (jusqu'aux limites du graphique)
                # Trouver l'intersection avec les bords du graphique
                t_max = float('inf')
                
                # Intersection avec x1 = x1_max
                if direction[0] > 1e-6:
                    t_x1 = (x1_max - start_point[0]) / direction[0]
                    t_max = min(t_max, t_x1)
                
                # Intersection avec x2 = x2_max
                if direction[1] > 1e-6:
                    t_x2 = (x2_max - start_point[1]) / direction[1]
                    t_max = min(t_max, t_x2)
                
                # Si pas d'intersection trouvée, utiliser une grande valeur
                if t_max == float('inf'):
                    t_max = min(x1_max, x2_max) * 1.2
                
                end_point = start_point + direction * t_max 
                
                # Tracer la demi droite  comme une GROSSE DEMI-DROITE JAUNE 
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       color='#FFD700', linewidth=6, 
                       solid_capstyle='round',
                       zorder=8, alpha=0.95,
                       label='La droite des solutions optimales')
                
                # Marquer le point de départ
                ax.plot(start_point[0], start_point[1], 'o', color='#FFD700', 
                       markersize=9, zorder=9,
                       markeredgecolor='#000000', markeredgewidth=1.6,
                       label='1er point de contact')
                
                # Annotation pour demi droite optimal
                mid_ray = (start_point + end_point) / 2
                ax.annotate('Demi-droite de\nsolutions opimales)',
                           xy=(mid_ray[0], mid_ray[1]), 
                           xytext=(50, -40), textcoords='offset points',
                           fontsize=10, color='#000000', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFD700', 
                                   edgecolor='#000000', linewidth=2, alpha=0.95),
                           arrowprops=dict(arrowstyle='->', color='#000000', lw=2),
                           ha='center', zorder=16)
                
                # Annotation au point de départ
                ax.annotate(f'({start_point[0]:.2f}, {start_point[1]:.2f})', 
                           xy=(start_point[0], start_point[1]), 
                           xytext=(20, -30), textcoords='offset points',
                           fontsize=9, color='#000000', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                   edgecolor='#FFD700', linewidth=2, alpha=0.95),
                           ha='left', zorder=16)
        
        # CAS 2: ARÊTE OPTIMALE (plusieurs points optimaux)
        elif optimal_points is not None and len(optimal_points) >= 2:
            
            optimal_points_array = np.array(optimal_points)
            
            # Tracer les segments entre sommets optimaux
            center = optimal_points_array.mean(axis=0)
            angles = np.arctan2(optimal_points_array[:, 1] - center[1], 
                               optimal_points_array[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_points = optimal_points_array[sorted_indices]
            
            for i in range(len(sorted_points)):
                p1 = sorted_points[i]
                p2 = sorted_points[(i + 1) % len(sorted_points)]
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color='#FFD700', linewidth=4, solid_capstyle='round',
                       label='Arête optimale' if i == 0 else '', zorder=6, alpha=0.8)
            
            # Marquer sommets optimaux
            ax.plot(optimal_points_array[:, 0], optimal_points_array[:, 1], 'o', 
                   color='#FFD700', markersize=16, zorder=7,
                   markeredgecolor='#000000', markeredgewidth=1.5)
            
            # Annotation pour arête optimale
            mid_point = optimal_points_array.mean(axis=0)
            ax.annotate(f'Infinité de solutions\n(sur segment jaune)', 
                       xy=(mid_point[0], mid_point[1]), 
                       xytext=(40, 40), textcoords='offset points',
                       fontsize=9, color='#000000', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCCB79', 
                               edgecolor='#000000', linewidth=1.2, alpha=0.95),
                       arrowprops=dict(arrowstyle='->', color='#000000', lw=1.5),
                       ha='center', zorder=16)
        
        # CAS 3: SOLUTION UNIQUE
        else:
            
            ax.plot(solution[0], solution[1], 'o', color='#FFD700', 
                   markersize=18, label='Solution optimale', zorder=6,
                   markeredgecolor='#000000', markeredgewidth=1.5)
            
            ax.annotate(f'Optimal\n({solution[0]:.2f}, {solution[1]:.2f})', 
                       xy=(solution[0], solution[1]), 
                       xytext=(40, 40), textcoords='offset points',
                       fontsize=9, color='#000000', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCCB79', 
                               edgecolor='#000000', linewidth=1.2, alpha=0.95),
                       arrowprops=dict(arrowstyle='->', color='#000000', lw=1.5),
                       ha='center', zorder=16)
    
 
    ax.set_xlim(0, x1_max)
    ax.set_ylim(0, x2_max)
    ax.set_xlabel('x₁', fontsize=14, fontweight='bold', color='#000000')
    ax.set_ylabel('x₂', fontsize=14, fontweight='bold', color='#000000')
    
    if not has_feasible_region:
        title_text = 'Contraintes Incompatibles'
        title_color = '#8B0000'
    elif status == 'unbounded':
        title_text = 'Problème Non Borné (Optimum non finie)'
        title_color = '#FF8C00'
    elif not region_bounded and recession_direction is not None:
        title_text = 'Région Non Bornée avec Demi Droite Optimal'
        title_color = '#006400'
    elif optimal_points is not None and len(optimal_points) >= 2:
        title_text = 'Infinité de solutions sur une arête'
        title_color = '#DAA520'
    else:
        title_text = 'Région admissible et Solution Optimale'
        title_color = '#0B3B36'
    
    ax.set_title(title_text, fontsize=15, fontweight='bold', pad=15, color=title_color)
    
    ax.grid(True, alpha=0.3, linestyle='--', color='#888888', linewidth=0.8)
    ax.tick_params(colors='#000000', labelsize=10)

    legend = ax.legend(loc='lower right', fontsize=8, facecolor='white', 
                      edgecolor='#0B3B36', framealpha=0.95, shadow=True)
    for text in legend.get_texts():
        text.set_color('#000000')

    for spine in ax.spines.values():
        spine.set_edgecolor('#0B3B36')
        spine.set_linewidth(2.5)
    
    fig.tight_layout()
    return fig