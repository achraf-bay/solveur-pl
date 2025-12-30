from fpdf import FPDF
import os
from datetime import datetime

def generate_pdf(result, c, A, b, operators, obj_type, fig):
    
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête avec fond
    pdf.set_fill_color(11, 59, 54)
    pdf.rect(0, 0, 210, 45, 'F')
    
    pdf.set_font("Arial", "B", 26)
    pdf.set_text_color(252, 203, 121)
    pdf.cell(0, 25, "SOLVEUR DE PROBLEME LINEAIRE", ln=True, align='C')
    
    pdf.set_font("Arial", "I", 11)
    pdf.set_text_color(73, 115, 113)
    pdf.cell(0, 10, "Solution Optimale Generee", ln=True, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    
    # Date
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Date de generation: {datetime.now().strftime('%d/%m/%Y àk %H:%M')}", ln=True)
    pdf.ln(8)
    
    # Section Problème
    pdf.set_fill_color(252, 203, 121)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(11, 59, 54)
    pdf.cell(0, 10, " DEFINITION DU PROBLEME", ln=True, fill=True)
    pdf.ln(5)
    
    # Fonction objectif
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(73, 115, 113)
    pdf.cell(0, 8, "Fonction objectif:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    obj_text = f"  {'Maximiser' if obj_type == 'max' else 'Minimiser'} Z = {c[0]}x1 + {c[1]}x2"
    pdf.cell(0, 8, obj_text, ln=True)
    pdf.ln(5)
    
    # Contraintes
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(73, 115, 113)
    pdf.cell(0, 8, "Contraintes:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    
    op_symbols = {"<=": "<=", ">=": ">=", "=": "="}
    for i, (row, bi, op) in enumerate(zip(A, b, operators)):
        constraint_text = f"  ({i+1}) {row[0]}x1 + {row[1]}x2 {op_symbols.get(op, op)} {bi}"
        pdf.cell(0, 8, constraint_text, ln=True)
    
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "  x1, x2 >= 0", ln=True)
    pdf.ln(10)
    
    # Section Solution
    pdf.set_fill_color(252, 203, 121)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(11, 59, 54)
    pdf.cell(0, 10, " SOLUTION OPTIMALE", ln=True, fill=True)
    pdf.ln(8)
    
    if result['success']:
        # Encadré solution
        pdf.set_fill_color(245, 245, 245)
        pdf.rect(15, pdf.get_y(), 180, 45, 'F')
        
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(11, 59, 54)
        pdf.ln(5)
        
        # Point optimal
        pdf.set_x(25)
        pdf.cell(80, 10, f"x1* = {result['x'][0]:.6f}", ln=False)
        pdf.cell(80, 10, f"x2* = {result['x'][1]:.6f}", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(252, 203, 121)
        pdf.set_fill_color(11, 59, 54)
        pdf.cell(0, 12, f"Valeur optimale:  Z* = {result['z']:.6f}", ln=True, align='C', fill=True)
        
        pdf.ln(15)
        
    else:
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(255, 100, 100)
        pdf.cell(0, 10, "Aucune solution faisable trouvee", ln=True, align='C')
        pdf.ln(10)
    
    # Ajouter graphique
    pdf.add_page()
    
    # En-tête page 2
    pdf.set_fill_color(11, 59, 54)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(252, 203, 121)
    pdf.cell(0, 20, "VISUALISATION GRAPHIQUE", ln=True, align='C')
    pdf.ln(5)
    
    # Sauvegarder et ajouter graphique
    graph_path = "temp_graph.png"
    fig.savefig(graph_path, dpi=180, bbox_inches='tight', 
               facecolor="#ffffff", edgecolor='none')
    
    pdf.image(graph_path, x=10, y=45, w=190)
    
    # Pied de page
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Document genere automatiquement - {datetime.now().strftime('%d/%m/%Y')}", align='C')
    
    # Sauvegarder
    output_path = "solution_PL.pdf"
    pdf.output(output_path)
    
    # Nettoyer
    if os.path.exists(graph_path):
        os.remove(graph_path)
    
    return output_path