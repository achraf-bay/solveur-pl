from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox, 
                               QScrollArea, QFrame, QSizePolicy, QTextEdit, QDialog, QDialogButtonBox)  
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from utils.validators import validate_inputs
from core.optimizer import solve_linear_program
from core.plotting import create_plot
from pdf_export import generate_pdf
from LLM_GEMINI.llm_extractor import LLMExtractor
import os
from dotenv import load_dotenv
load_dotenv() 

#===========================LLM Worker Thread=================================
class LLMWorker(QThread):
    """Thread pour extraction LLM sans bloquer l'interface"""
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, extractor, text):
        super().__init__()
        self.extractor = extractor
        self.text = text
        api_key = os.getenv('GEMINI_API_KEY', '')
        self.llm_extractor = LLMExtractor(api_key)
    
    def run(self):
        try:
            result = self.extractor.extract_problem(self.text)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class APIKeyDialog(QDialog):
    """Dialog pour entrer l'API Key"""
    def __init__(self, parent=None, current_key=""):
        super().__init__(parent)
        self.setWindowTitle("Configuration API Key")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        label = QLabel("Entrez votre API Key Gemini:")
        label.setFont(QFont("Arial", 12))
        layout.addWidget(label)
        
        self.key_input = QLineEdit(current_key)
        self.key_input.setFont(QFont("Arial", 11))
        self.key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.key_input)
        
        info = QLabel("Obtenez votre clé sur: https://makersuite.google.com/app/apikey")
        info.setFont(QFont("Arial", 9))
        info.setStyleSheet("color: #666;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_key(self):
        return self.key_input.text()

#===========================Main Window=================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solveur de Problèmes Linéaires")
        self.setGeometry(0, 0, 1920, 1080)
        self.showMaximized()
        
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0B3B36, stop:1 #1a1a1a);
            }
            QWidget {
                background: transparent;
            }
            QLabel {
                color: #FCCB79;
                background: transparent;
            }
            QLineEdit, QTextEdit {
                background-color: rgba(73, 115, 113, 0.3);
                color: #FCCB79;
                border: 2px solid rgba(252, 203, 121, 0.4);
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #FCCB79;
                background-color: rgba(73, 115, 113, 0.5);
            }
            QComboBox {
                background-color: rgba(73, 115, 113, 0.3);
                color: #FCCB79;
                border: 2px solid rgba(252, 203, 121, 0.4);
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #0B3B36;
                color: #FCCB79;
                selection-background-color: #497371;
                border: 2px solid #FCCB79;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                background: rgba(73, 115, 113, 0.2);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #497371;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #FCCB79;
            }
        """)
        
        self.constraints = []
        self.result = None
        self.current_fig = None
        self.input_mode = 'standard'  # 'standard' ou 'ai'
        self.llm_extractor = LLMExtractor()
        self.llm_worker = None
        
        self.init_ui()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        left_panel = self.create_input_panel()
        right_panel = self.create_result_panel()
        
        main_layout.addWidget(left_panel, 50)
        main_layout.addWidget(right_panel, 50)
    
    def create_input_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(11, 59, 54, 0.95), stop:1 rgba(73, 115, 113, 0.85));
                border-right: 3px solid rgba(252, 203, 121, 0.3);
            }
        """)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(25)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Titre
        title = QLabel("⚡ SOLVEUR PL")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 32, QFont.Bold))
        title.setStyleSheet("color: #FCCB79; padding: 20px;")
        layout.addWidget(title)
        
        subtitle = QLabel("Programmation Linéaire Optimale")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setStyleSheet("color: #497371; margin-bottom: 10px;")
        layout.addWidget(subtitle)
        
        # NOUVEAUX BOUTONS MODE
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(15)
        
        self.btn_standard = QPushButton("📝 STANDARD")
        self.btn_standard.setMinimumHeight(55)
        self.btn_standard.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_standard.setCursor(Qt.PointingHandCursor)
        self.btn_standard.clicked.connect(lambda: self.switch_mode('standard'))
        
        self.btn_ai = QPushButton("🤖 AI-TEXT")
        self.btn_ai.setMinimumHeight(55)
        self.btn_ai.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_ai.setCursor(Qt.PointingHandCursor)
        self.btn_ai.clicked.connect(lambda: self.switch_mode('ai'))
        
        # Bouton API Key
        self.btn_api_key = QPushButton("🔑 API KEY")
        self.btn_api_key.setMinimumHeight(55)
        self.btn_api_key.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_api_key.setCursor(Qt.PointingHandCursor)
        self.btn_api_key.setStyleSheet("""
            QPushButton {
                background: rgba(73, 115, 113, 0.5);
                color: rgba(252, 203, 121, 0.6);
                border: 2px solid rgba(73, 115, 113, 0.8);
                border-radius: 15px;
            }
            QPushButton:hover {
                background: rgba(73, 115, 113, 0.7);
                color: #FCCB79;
            }
        """)
        self.btn_api_key.clicked.connect(self.configure_api_key)
        
        mode_layout.addWidget(self.btn_standard)
        mode_layout.addWidget(self.btn_ai)
        mode_layout.addWidget(self.btn_api_key)
        layout.addLayout(mode_layout)
        
        # Container principal avec stacked widget simulé
        self.standard_container = self.create_standard_inputs()
        self.ai_container = self.create_ai_inputs()
        
        layout.addWidget(self.standard_container)
        layout.addWidget(self.ai_container)
        
        # Initial state
        self.switch_mode('standard')
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        return panel
    
    def create_standard_inputs(self):
        """Interface standard originale"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: rgba(11, 59, 54, 0.6);
                border: 2px solid rgba(252, 203, 121, 0.3);
                border-radius: 20px;
                padding: 25px;
            }
        """)
        
        input_layout = QVBoxLayout(container)
        input_layout.setSpacing(20)
        
        # Boutons Max/Min
        type_layout = QHBoxLayout()
        type_layout.setSpacing(15)
        
        self.btn_max = QPushButton("📈 MAXIMISER")
        self.btn_max.setMinimumHeight(55)
        self.btn_max.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_max.setCursor(Qt.PointingHandCursor)
        self.btn_max.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FCCB79, stop:1 #f9b74d);
                color: #0B3B36;
                border: none;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ffd699;
            }
        """)
        self.btn_max.clicked.connect(lambda: self.set_objective_type('max'))
        
        self.btn_min = QPushButton("📉 MINIMISER")
        self.btn_min.setMinimumHeight(55)
        self.btn_min.setFont(QFont("Arial", 13, QFont.Bold))
        self.btn_min.setCursor(Qt.PointingHandCursor)
        self.btn_min.setStyleSheet("""
            QPushButton {
                background: rgba(73, 115, 113, 0.5);
                color: rgba(252, 203, 121, 0.6);
                border: 2px solid rgba(73, 115, 113, 0.8);
                border-radius: 15px;
            }
            QPushButton:hover {
                background: rgba(73, 115, 113, 0.7);
                color: #FCCB79;
            }
        """)
        self.btn_min.clicked.connect(lambda: self.set_objective_type('min'))
        
        type_layout.addWidget(self.btn_max)
        type_layout.addWidget(self.btn_min)
        input_layout.addLayout(type_layout)
        
        self.objective_type = 'max'
        
        # Fonction objectif
        obj_card = QFrame()
        obj_card.setStyleSheet("""
            QFrame {
                background: rgba(73, 115, 113, 0.2);
                border-radius: 15px;
                padding: 15px;
            }
        """)
        obj_card_layout = QVBoxLayout(obj_card)
        
        obj_label = QLabel("🎯 Fonction Objectif : Z =")
        obj_label.setFont(QFont("Arial", 14, QFont.Bold))
        obj_label.setStyleSheet("color: #FCCB79;")
        obj_card_layout.addWidget(obj_label)
        
        obj_inputs = QHBoxLayout()
        obj_inputs.setSpacing(10)
        
        self.c1_input = QLineEdit("3")
        self.c1_input.setFixedSize(90, 50)
        self.c1_input.setAlignment(Qt.AlignCenter)
        self.c1_input.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.c2_input = QLineEdit("2")
        self.c2_input.setFixedSize(90, 50)
        self.c2_input.setAlignment(Qt.AlignCenter)
        self.c2_input.setFont(QFont("Arial", 14, QFont.Bold))
        
        label_x1 = QLabel("x₁  +")
        label_x1.setFont(QFont("Arial", 14))
        label_x1.setStyleSheet("color: #497371;")
        
        label_x2 = QLabel("x₂")
        label_x2.setFont(QFont("Arial", 14))
        label_x2.setStyleSheet("color: #497371;")
        
        obj_inputs.addWidget(self.c1_input)
        obj_inputs.addWidget(label_x1)
        obj_inputs.addWidget(self.c2_input)
        obj_inputs.addWidget(label_x2)
        obj_inputs.addStretch()
        
        obj_card_layout.addLayout(obj_inputs)
        input_layout.addWidget(obj_card)
        
        # Contraintes
        constraint_header = QLabel("🔗 Contraintes")
        constraint_header.setFont(QFont("Arial", 16, QFont.Bold))
        constraint_header.setStyleSheet("color: #FCCB79; margin-top: 10px;")
        input_layout.addWidget(constraint_header)
        
        self.constraints_layout = QVBoxLayout()
        self.constraints_layout.setSpacing(12)
        input_layout.addLayout(self.constraints_layout)
        
        self.add_constraint("2", "1", "<=", "18")
        self.add_constraint("1", "3", "<=", "12")
        
        # Bouton ajouter
        btn_add = QPushButton("➕ Ajouter Contrainte")
        btn_add.setMinimumHeight(45)
        btn_add.setFont(QFont("Arial", 12, QFont.Bold))
        btn_add.setCursor(Qt.PointingHandCursor)
        btn_add.setStyleSheet("""
            QPushButton {
                background: rgba(73, 115, 113, 0.6);
                color: #FCCB79;
                border: 2px dashed rgba(252, 203, 121, 0.4);
                border-radius: 12px;
            }
            QPushButton:hover {
                background: rgba(73, 115, 113, 0.8);
                border: 2px solid #FCCB79;
            }
        """)
        btn_add.clicked.connect(lambda: self.add_constraint())
        input_layout.addWidget(btn_add)
        
        # Non-négativité
        non_neg = QLabel("✓ x₁, x₂ ≥ 0")
        non_neg.setAlignment(Qt.AlignCenter)
        non_neg.setFont(QFont("Arial", 12, QFont.Bold))
        non_neg.setStyleSheet("""
            background: rgba(73, 115, 113, 0.3);
            padding: 12px;
            border-radius: 10px;
            color: #FCCB79;
            border: 1px solid rgba(252, 203, 121, 0.3);
        """)
        input_layout.addWidget(non_neg)
        
        # Bouton Résoudre
        btn_solve = QPushButton("🚀 RÉSOUDRE")
        btn_solve.setMinimumHeight(65)
        btn_solve.setFont(QFont("Arial", 18, QFont.Bold))
        btn_solve.setCursor(Qt.PointingHandCursor)
        btn_solve.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FCCB79, stop:1 #f9b74d);
                color: #0B3B36;
                border: none;
                border-radius: 18px;
                font-weight: bold;
                margin-top: 15px;
            }
            QPushButton:hover {
                background: #ffd699;
            }
            QPushButton:pressed {
                background: #e6b665;
            }
        """)
        btn_solve.clicked.connect(self.solve_problem)
        input_layout.addWidget(btn_solve)
        
        return container
    
    def create_ai_inputs(self):
        """Interface AI-Text"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: rgba(11, 59, 54, 0.6);
                border: 2px solid rgba(252, 203, 121, 0.3);
                border-radius: 20px;
                padding: 25px;
            }
        """)
        
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        
        # Titre
        ai_title = QLabel("🤖 Extraction Automatique par IA")
        ai_title.setAlignment(Qt.AlignCenter)
        ai_title.setFont(QFont("Arial", 16, QFont.Bold))
        ai_title.setStyleSheet("color: #FCCB79; margin-bottom: 10px;")
        layout.addWidget(ai_title)
        
        # Zone de texte
        text_label = QLabel("📄 Collez votre problème en texte libre:")
        text_label.setFont(QFont("Arial", 13, QFont.Bold))
        text_label.setStyleSheet("color: #FCCB79;")
        layout.addWidget(text_label)
        
        self.ai_text_input = QTextEdit()
        self.ai_text_input.setMinimumHeight(300)
        self.ai_text_input.setFont(QFont("Arial", 12))
        self.ai_text_input.setPlaceholderText(
            "Exemple: Une entreprise disposant de 10 000 m² de carton en réserve, "
            "fabrique et commercialise 2 types de boîtes en carton. La fabrication "
            "d'une boîte en carton de type 1 ou 2 requiert, respectivement, 1 et 2 m² "
            "de carton ainsi que 2 et 3 minutes de temps d'assemblage..."
        )
        layout.addWidget(self.ai_text_input)
        
        # Bouton extraire
        self.btn_extract = QPushButton("✨ EXTRAIRE AVEC GEMINI")
        self.btn_extract.setMinimumHeight(65)
        self.btn_extract.setFont(QFont("Arial", 18, QFont.Bold))
        self.btn_extract.setCursor(Qt.PointingHandCursor)
        self.btn_extract.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FCCB79, stop:1 #f9b74d);
                color: #0B3B36;
                border: none;
                border-radius: 18px;
                font-weight: bold;
                margin-top: 15px;
            }
            QPushButton:hover {
                background: #ffd699;
            }
            QPushButton:pressed {
                background: #e6b665;
            }
        """)
        self.btn_extract.clicked.connect(self.extract_with_ai)
        layout.addWidget(self.btn_extract)
        
        # Status label
        self.ai_status = QLabel("")
        self.ai_status.setAlignment(Qt.AlignCenter)
        self.ai_status.setFont(QFont("Arial", 11))
        self.ai_status.setStyleSheet("color: #FCCB79; padding: 10px;")
        self.ai_status.setWordWrap(True)
        layout.addWidget(self.ai_status)
        
        return container
    
    def switch_mode(self, mode):
        """Change entre mode standard et AI"""
        self.input_mode = mode
        
        if mode == 'standard':
            self.standard_container.setVisible(True)
            self.ai_container.setVisible(False)
            
            self.btn_standard.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FCCB79, stop:1 #f9b74d);
                    color: #0B3B36;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                }
                QPushButton:hover { background: #ffd699; }
            """)
            self.btn_ai.setStyleSheet("""
                QPushButton {
                    background: rgba(73, 115, 113, 0.5);
                    color: rgba(252, 203, 121, 0.6);
                    border: 2px solid rgba(73, 115, 113, 0.8);
                    border-radius: 15px;
                }
                QPushButton:hover {
                    background: rgba(73, 115, 113, 0.7);
                    color: #FCCB79;
                }
            """)
        else:  # ai
            self.standard_container.setVisible(False)
            self.ai_container.setVisible(True)
            
            self.btn_ai.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FCCB79, stop:1 #f9b74d);
                    color: #0B3B36;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                }
                QPushButton:hover { background: #ffd699; }
            """)
            self.btn_standard.setStyleSheet("""
                QPushButton {
                    background: rgba(73, 115, 113, 0.5);
                    color: rgba(252, 203, 121, 0.6);
                    border: 2px solid rgba(73, 115, 113, 0.8);
                    border-radius: 15px;
                }
                QPushButton:hover {
                    background: rgba(73, 115, 113, 0.7);
                    color: #FCCB79;
                }
            """)
    
    def configure_api_key(self):
        """Ouvre dialog pour configurer API key"""
        dialog = APIKeyDialog(self, self.llm_extractor.api_key)
        if dialog.exec() == QDialog.Accepted:
            key = dialog.get_key()
            if key:
                self.llm_extractor.set_api_key(key)
                QMessageBox.information(self, "Succès", "API Key configurée avec succès!")
                self.btn_api_key.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #4CAF50, stop:1 #45a049);
                        color: white;
                        border: none;
                        border-radius: 15px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background: #45a049;
                    }
                """)
    
    def extract_with_ai(self):
        """Extrait le problème avec Gemini"""
        text = self.ai_text_input.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "Attention", "Veuillez entrer un texte de problème")
            return
        
        if not self.llm_extractor.api_key:
            QMessageBox.warning(self, "Attention", "Veuillez configurer votre API Key Gemini d'abord")
            self.configure_api_key()
            return
        
        # Désactiver le bouton pendant l'extraction
        self.btn_extract.setEnabled(False)
        self.ai_status.setText("⏳ Extraction en cours avec Gemini...")
        self.ai_status.setStyleSheet("color: #FCCB79;")
        
        # Lancer extraction dans thread séparé
        self.llm_worker = LLMWorker(self.llm_extractor, text)
        self.llm_worker.finished.connect(self.on_extraction_success)
        self.llm_worker.error.connect(self.on_extraction_error)
        self.llm_worker.start()
    
    def on_extraction_success(self, result):
        """Callback succès extraction"""
        self.btn_extract.setEnabled(True)
        self.ai_status.setText("✅ Extraction réussie! Données chargées en mode Standard.")
        self.ai_status.setStyleSheet("color: #4CAF50;")
        
        # Remplir l'interface standard avec les données extraites
        self.objective_type = result['objective_type']
        self.set_objective_type(result['objective_type'])
        
        self.c1_input.setText(str(result['c'][0]))
        self.c2_input.setText(str(result['c'][1]))
        
        # Effacer contraintes existantes
        while self.constraints:
            constraint = self.constraints.pop()
            constraint['widget'].deleteLater()
        
        # Ajouter nouvelles contraintes
        for constraint_data in result['constraints']:
            self.add_constraint(
                str(constraint_data['a']),
                str(constraint_data['b']),
                constraint_data['op'],
                str(constraint_data['c'])
            )
        
        # Passer en mode standard pour voir les résultats
        self.switch_mode('standard')
        
        QMessageBox.information(
            self,
            "Succès",
            f"Problème extrait avec succès!\n\n"
            f"Type: {result['objective_type'].upper()}\n"
            f"Fonction objectif: Z = {result['c'][0]}x₁ + {result['c'][1]}x₂\n"
            f"Contraintes: {len(result['constraints'])}\n\n"
            f"Vous pouvez maintenant résoudre le problème."
        )
    
    def on_extraction_error(self, error_msg):
        """Callback erreur extraction"""
        self.btn_extract.setEnabled(True)
        self.ai_status.setText(f"❌ Erreur: {error_msg}")
        self.ai_status.setStyleSheet("color: #ff6b6b;")
        QMessageBox.critical(self, "Erreur d'extraction", error_msg)
    
    def create_result_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0B3B36);
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("📊 RÉSULTAT & VISUALISATION")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 26, QFont.Bold))
        header.setStyleSheet("""
            color: #FCCB79;
            padding: 15px;
            background: rgba(11, 59, 54, 0.4);
            border-radius: 15px;
            border: 2px solid rgba(252, 203, 121, 0.3);
        """)
        layout.addWidget(header)
        
        # Zone de contenu
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.result_content = QWidget()
        self.result_layout = QVBoxLayout(self.result_content)
        self.result_layout.setSpacing(20)
        
        # Message initial
        initial_msg = QLabel("👈 Configurez votre problème\net cliquez sur RÉSOUDRE")
        initial_msg.setAlignment(Qt.AlignCenter)
        initial_msg.setFont(QFont("Arial", 16))
        initial_msg.setStyleSheet("""
            color: rgba(252, 203, 121, 0.5);
            padding: 100px;
            background: rgba(73, 115, 113, 0.1);
            border: 3px dashed rgba(252, 203, 121, 0.3);
            border-radius: 20px;
        """)
        self.result_layout.addWidget(initial_msg)
        self.result_layout.addStretch()
        
        scroll.setWidget(self.result_content)
        layout.addWidget(scroll)
        
        # Bouton export
        self.btn_export = QPushButton("📥 EXPORTER PDF")
        self.btn_export.setMinimumHeight(55)
        self.btn_export.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: rgba(73, 115, 113, 0.3);
                color: rgba(252, 203, 121, 0.4);
                border: 2px solid rgba(73, 115, 113, 0.5);
                border-radius: 12px;
            }
            QPushButton:enabled {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #497371, stop:1 #5a8785);
                color: #FCCB79;
                border: 2px solid #FCCB79;
            }
            QPushButton:enabled:hover {
                background: #5a8785;
            }
        """)
        self.btn_export.clicked.connect(self.export_pdf)
        layout.addWidget(self.btn_export)
        
        return panel
    
    def set_objective_type(self, obj_type):
        self.objective_type = obj_type
        if obj_type == 'max':
            self.btn_max.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FCCB79, stop:1 #f9b74d);
                    color: #0B3B36;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                }
                QPushButton:hover { background: #ffd699; }
            """)
            self.btn_min.setStyleSheet("""
                QPushButton {
                    background: rgba(73, 115, 113, 0.5);
                    color: rgba(252, 203, 121, 0.6);
                    border: 2px solid rgba(73, 115, 113, 0.8);
                    border-radius: 15px;
                }
                QPushButton:hover {
                    background: rgba(73, 115, 113, 0.7);
                    color: #FCCB79;
                }
            """)
        else:
            self.btn_min.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FCCB79, stop:1 #f9b74d);
                    color: #0B3B36;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                }
                QPushButton:hover { background: #ffd699; }
            """)
            self.btn_max.setStyleSheet("""
                QPushButton {
                    background: rgba(73, 115, 113, 0.5);
                    color: rgba(252, 203, 121, 0.6);
                    border: 2px solid rgba(73, 115, 113, 0.8);
                    border-radius: 15px;
                }
                QPushButton:hover {
                    background: rgba(73, 115, 113, 0.7);
                    color: #FCCB79;
                }
            """)
    
    def add_constraint(self, a="", b="", op="<=", c=""):
        constraint_widget = QFrame()
        constraint_widget.setStyleSheet("""
            QFrame {
                background: rgba(73, 115, 113, 0.15);
                border-radius: 12px;
                padding: 10px;
                border: 1px solid rgba(252, 203, 121, 0.2);
            }
        """)
        constraint_layout = QHBoxLayout(constraint_widget)
        constraint_layout.setSpacing(8)
        
        a_input = QLineEdit(a)
        a_input.setFixedSize(80, 45)
        a_input.setAlignment(Qt.AlignCenter)
        a_input.setFont(QFont("Arial", 13, QFont.Bold))
        
        label1 = QLabel("x₁ +")
        label1.setFont(QFont("Arial", 13))
        label1.setStyleSheet("color: #497371;")
        
        b_input = QLineEdit(b)
        b_input.setFixedSize(80, 45)
        b_input.setAlignment(Qt.AlignCenter)
        b_input.setFont(QFont("Arial", 13, QFont.Bold))
        
        label2 = QLabel("x₂")
        label2.setFont(QFont("Arial", 13))
        label2.setStyleSheet("color: #497371;")
        
        op_combo = QComboBox()
        op_combo.addItems(["≤", "≥", "="])
        symbol_map = {"<=": "≤", ">=": "≥", "=": "="}
        op_combo.setCurrentText(symbol_map.get(op, "≤"))
        op_combo.setFixedSize(70, 45)
        op_combo.setFont(QFont("Arial", 14, QFont.Bold))
        
        c_input = QLineEdit(c)
        c_input.setFixedSize(90, 45)
        c_input.setAlignment(Qt.AlignCenter)
        c_input.setFont(QFont("Arial", 13, QFont.Bold))
        
        btn_remove = QPushButton("🗑")
        btn_remove.setFixedSize(45, 45)
        btn_remove.setFont(QFont("Arial", 16))
        btn_remove.setCursor(Qt.PointingHandCursor)
        btn_remove.setStyleSheet("""
            QPushButton {
                background: rgba(255, 100, 100, 0.3);
                color: #ff6b6b;
                border: 2px solid rgba(255, 107, 107, 0.5);
                border-radius: 10px;
            }
            QPushButton:hover {
                background: rgba(255, 100, 100, 0.5);
            }
        """)
        btn_remove.clicked.connect(lambda: self.remove_constraint(constraint_widget))
        
        constraint_layout.addWidget(a_input)
        constraint_layout.addWidget(label1)
        constraint_layout.addWidget(b_input)
        constraint_layout.addWidget(label2)
        constraint_layout.addWidget(op_combo)
        constraint_layout.addWidget(c_input)
        constraint_layout.addWidget(btn_remove)
        
        self.constraints_layout.addWidget(constraint_widget)
        self.constraints.append({
            'widget': constraint_widget,
            'a': a_input,
            'b': b_input,
            'op': op_combo,
            'c': c_input
        })
    
    def remove_constraint(self, widget):
        if len(self.constraints) > 1:
            self.constraints = [c for c in self.constraints if c['widget'] != widget]
            widget.deleteLater()
    
    def solve_problem(self):
        try:
            c = [float(self.c1_input.text()), float(self.c2_input.text())]
            A = []
            b = []
            operators = []
            
            reverse_map = {"≤": "<=", "≥": ">=", "=": "="}
            
            for constraint in self.constraints:
                A.append([float(constraint['a'].text()), float(constraint['b'].text())])
                b.append(float(constraint['c'].text()))
                symbol = constraint['op'].currentText()
                operators.append(reverse_map.get(symbol, "<="))
            
            if not validate_inputs(c, A, b):
                QMessageBox.warning(self, "Erreur", "Vérifiez vos entrées")
                return
 
            self.result = solve_linear_program(c, A, b, operators, self.objective_type) 
            print("DEBUG solve_linear_program_result=",self.result)
            self.display_results(self.result, c, A, b, operators)
             
        except ValueError:
            QMessageBox.critical(self, "Erreur", "Entrez des nombres valides")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur: {str(e)}")
    
  

    def display_results(self, result, c, A, b, operators):
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        status = result.get('status', 'optimal')
        solution_type = result.get('solution_type', 'unique')
        region_bounded = bool(result.get('region_bounded', True))
        optimal_points = result.get('optimal_points', [])
        z_value = result.get('z')
        x_sol = result.get('x') or [None, None]
        objective_type = result.get('objective_type', getattr(self, 'objective_type', 'max'))
        recession_direction = result.get('recession_direction')

        has_valid_solution = bool(result.get('success') and isinstance(x_sol, (list, tuple)) and len(x_sol) >= 2 and x_sol[0] is not None and x_sol[1] is not None)


        # Carte Solution
        solution_card = QFrame()
        solution_card.setStyleSheet("""
            QFrame {
                background: rgba(11, 59, 54, 0.7);
                border: 2px solid rgba(252, 203, 121, 0.5);
                border-radius: 20px;
                padding: 25px;
            }
        """)
        solution_layout = QVBoxLayout(solution_card)

        if status == 'infeasible':
            title = QLabel("❌ RÉGION ADMISSIBLE VIDE")
            title.setAlignment(Qt.AlignCenter)
            title.setFont(QFont("Arial", 20, QFont.Bold))
            title.setStyleSheet("color: #ff6b6b; margin-bottom: 15px;")
            solution_layout.addWidget(title)
            msg = QLabel("Les contraintes sont incompatibles.")
            msg.setAlignment(Qt.AlignCenter)
            msg.setFont(QFont("Arial", 13))
            msg.setWordWrap(True)
            msg.setStyleSheet("color: #FCCB79;")
            solution_layout.addWidget(msg)

        elif status == 'unbounded':
            title = QLabel("❌ SOLUTION NON BORNÉE")
            title.setAlignment(Qt.AlignCenter)
            title.setFont(QFont("Arial", 20, QFont.Bold))
            title.setStyleSheet("color: #FFA500; margin-bottom: 15px;")
            solution_layout.addWidget(title)
            msg = QLabel("La fonction objectif n'admet pas de solution finie (tend à ±∞).")
            msg.setAlignment(Qt.AlignCenter)
            msg.setFont(QFont("Arial", 13))
            msg.setWordWrap(True)
            msg.setStyleSheet("color: #FCCB79;")
            solution_layout.addWidget(msg)

        else:
            
            if has_valid_solution:
                # CAS 1: région non bornée + direction de récession
                if not region_bounded and recession_direction is not None:
                    title = QLabel("✅ INFINITÉ DE SOLUTIONS\n(sur une direction non bornée)")
                    title.setAlignment(Qt.AlignCenter)
                    title.setFont(QFont("Arial", 20, QFont.Bold))
                    title.setStyleSheet("color: #32CD32; margin-bottom: 15px;")
                    solution_layout.addWidget(title)
                
                    info = QLabel("La région est non bornée.\nLa solution optimale s'étend à l'infini dans une direction.")
                    info.setAlignment(Qt.AlignCenter)
                    info.setFont(QFont("Arial", 11))
                    info.setWordWrap(True)
                    info.setStyleSheet("color: rgba(252,203,121,0.9); margin-bottom: 10px;")
                    solution_layout.addWidget(info)
            
                # CAS 2: région bornée + plusieurs points optimaux)
                elif solution_type == 'infinite_edge' and optimal_points and len(optimal_points) >= 2:
                    title = QLabel("✅ INFINITÉ DE SOLUTIONS")
                    title.setAlignment(Qt.AlignCenter)
                    title.setFont(QFont("Arial", 20, QFont.Bold))
                    title.setStyleSheet("color: #FFD700; margin-bottom: 15px;")
                    solution_layout.addWidget(title)
 
                    info = QLabel(f"Tous les points sur le segment reliant {len(optimal_points)} sommets sont optimaux.")
                    info.setAlignment(Qt.AlignCenter)
                    info.setFont(QFont("Arial", 11))
                    info.setWordWrap(True)
                    info.setStyleSheet("color: rgba(252,203,121,0.9); margin-bottom: 10px;")
                    solution_layout.addWidget(info)
            
                # CAS 3: Solution unique
                elif solution_type == 'unique':
                    title = QLabel("✅ SOLUTION OPTIMALE UNIQUE")
                    title.setAlignment(Qt.AlignCenter)
                    title.setFont(QFont("Arial", 20, QFont.Bold))
                    title.setStyleSheet("color: #4CAF50; margin-bottom: 15px;")
                    solution_layout.addWidget(title)
            
                else:
                    title = QLabel("✅ SOLUTION OPTIMALE")
                    title.setAlignment(Qt.AlignCenter)
                    title.setFont(QFont("Arial", 20, QFont.Bold))
                    title.setStyleSheet("color: #4CAF50; margin-bottom: 15px;")
                    solution_layout.addWidget(title)
            else:
                # Pas de solution valide mais status=optimal
                title = QLabel("❌ ERREUR")
                title.setAlignment(Qt.AlignCenter)
                title.setFont(QFont("Arial", 20, QFont.Bold))
                title.setStyleSheet("color: #ff6b6b; margin-bottom: 15px;")
                solution_layout.addWidget(title)
                err = QLabel(result.get('message', 'Une erreur est survenue.'))
                err.setAlignment(Qt.AlignCenter)
                err.setFont(QFont("Arial", 13))
                err.setWordWrap(True)
                err.setStyleSheet("color: #FCCB79;")
                solution_layout.addWidget(err)

            # Valeurs de la solution
            if has_valid_solution:
                values_layout = QHBoxLayout()
                values_layout.setSpacing(15)
                for i, val in enumerate(x_sol):
                    try:
                        text = f"x{i+1} = {float(val):.4f}"
                    except Exception:
                        text = f"x{i+1} = {val}"
                    value_badge = QLabel(text)
                    value_badge.setAlignment(Qt.AlignCenter)
                    value_badge.setFont(QFont("Arial", 16, QFont.Bold))
                    value_badge.setStyleSheet("color: #FCCB79; padding: 6px 12px; border: 2px solid rgba(252,203,121,0.4); border-radius: 10px;")
                    values_layout.addWidget(value_badge)
                solution_layout.addLayout(values_layout)

                if z_value is not None:
                    try:
                        ztext = f"Z = {float(z_value):.4f}"
                    except Exception:
                        ztext = f"Z = {z_value}"
                    z_label = QLabel(ztext)
                    z_label.setAlignment(Qt.AlignCenter)
                    z_label.setFont(QFont("Arial", 22, QFont.Bold))
                    z_label.setStyleSheet("color: white; padding: 10px; border-radius: 10px; background: #4CAF50;")
                    solution_layout.addWidget(z_label)

                # Note sur la région
                if not region_bounded and recession_direction is not None:
                    note_text = "⚠️ Région non bornée - Solution sur une direction infinie"
                    note_color = "#32CD32"
                elif not region_bounded:
                    note_text = "⚠️ Région non bornée"
                    note_color = "#FFA500"
                elif solution_type == 'infinite_edge' and optimal_points and len(optimal_points) >= 2:
                    note_text = f"✓ Région bornée - {len(optimal_points)} sommets optimaux"
                    note_color = "#FFD700"
                else:
                    note_text = "✓ Région bornée - Solution unique"
                    note_color = "rgba(252,203,121,0.8)"
            
                note = QLabel(note_text)
                note.setAlignment(Qt.AlignCenter)
                note.setFont(QFont("Arial", 11, QFont.Bold))
                note.setStyleSheet(f"color: {note_color}; margin-top: 8px;")
                solution_layout.addWidget(note)

        self.result_layout.addWidget(solution_card)

        # ========= Graph =========
        graph_card = QFrame()
        graph_card.setStyleSheet("""
            QFrame {
                background: rgba(11, 59, 54, 0.5);
                border: 2px solid rgba(252, 203, 121, 0.3);
                border-radius: 20px;
                padding: 20px;
            }
        """)
        graph_layout = QVBoxLayout(graph_card)

        # Titre du graphique adapté
        if status == 'infeasible':
            graph_title = QLabel("📈 Contraintes Incompatibles (Région vide)")
            graph_title.setStyleSheet("color: #ff6b6b;")
        elif status == 'unbounded':
            graph_title = QLabel("📈 Région Non Bornée - Pas de Solution Finie")
            graph_title.setStyleSheet("color: #FFA500;")
        elif not region_bounded and recession_direction is not None:
            graph_title = QLabel("📈 Solutions Infinies - Région Non Bornée")
            graph_title.setStyleSheet("color: #32CD32;")
        elif solution_type == 'infinite_edge' and optimal_points and len(optimal_points) >= 2:
            graph_title = QLabel("📈 Solutions Infinies - Région Bornée")
            graph_title.setStyleSheet("color: #FFD700;")
        elif not region_bounded:
            graph_title = QLabel("📈 Solution Optimale - Région Non Bornée")
            graph_title.setStyleSheet("color: #4CAF50;")
        else:
            graph_title = QLabel("📈 Région Bornée - Zone Faisable")
            graph_title.setStyleSheet("color: #4CAF50;")

        graph_title.setFont(QFont("Arial", 16, QFont.Bold)) 
        title_layout = QHBoxLayout()
        title_layout.addStretch(1)
        title_layout.addWidget(graph_title)
        title_layout.addStretch(1) 
        graph_layout.addLayout(title_layout)

        try:
            operators_for_plot = operators
            optimal_point = x_sol if has_valid_solution else None

            fig = create_plot(
                A, b, optimal_point, c, objective_type,
                operators_for_plot,
                optimal_points=optimal_points,
                region_bounded=region_bounded,
                status=status,
                solver_result=result
            )

            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(450, 350)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            graph_layout.addWidget(canvas)
            canvas.draw()
            canvas.updateGeometry()
            self.current_fig = fig
        except Exception as e:
            error_label = QLabel(f"⚠️ Erreur graphique:\n{str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setFont(QFont("Arial", 12))
            error_label.setStyleSheet("color: #ff6b6b; padding: 20px;")
            error_label.setWordWrap(True)
            graph_layout.addWidget(error_label) 

        self.result_layout.addWidget(graph_card)
        self.result_layout.addStretch()

        self.btn_export.setEnabled(has_valid_solution)
    
    def export_pdf(self):
        if self.result and self.current_fig:
            try:
                c = [float(self.c1_input.text()), float(self.c2_input.text())]
                A = []
                b = []
                operators = []
                
                reverse_map = {"≤": "<=", "≥": ">=", "=": "="}
                
                for constraint in self.constraints:
                    A.append([float(constraint['a'].text()), float(constraint['b'].text())])
                    b.append(float(constraint['c'].text()))
                    operators.append(reverse_map.get(constraint['op'].currentText(), "<="))
                
                # Générer le PDF avec le module pdf_export
                pdf_path = generate_pdf(self.result, c, A, b, operators, 
                                       self.objective_type, self.current_fig)
                
                if os.path.exists(pdf_path):
                    os.startfile(pdf_path)
                    
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur PDF: {str(e)}")