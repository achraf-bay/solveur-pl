# Solveur de Problèmes Linéaires

Application de bureau moderne pour résoudre des problèmes de programmation linéaire avec extraction automatique via IA (Gemini).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##  Table des matières

- [Caractéristiques](##-caractéristiques)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Technologies](#-technologies)
- [Exemples](#-exemples)
- [Contributions](#-contributions)
- [Licence](#-licence)

##  Caractéristiques

###  Deux modes d'entrée

#### **Mode Standard**
- Interface manuelle intuitive
- Saisie de la fonction objectif (maximisation/minimisation)
- Gestion dynamique des contraintes
- Ajout/suppression de contraintes en temps réel

#### **Mode AI-TEXT** 
- **Extraction automatique via Gemini AI**
- Collez votre problème en texte libre
- L'IA extrait automatiquement :
  - Type d'optimisation (max/min)
  - Coefficients de la fonction objectif
  - Toutes les contraintes

###  Visualisation

- **Graphiques interactifs** : Zone faisable, contraintes, point optimal
- **Affichage des résultats** : Solution détaillée avec valeurs de x₁, x₂ et Z
- **Export PDF** : Rapport complet avec graphique

###  Interface moderne

- Design élégant avec dégradés
- Thème sombre optimisé
- Animations et transitions fluides
- Responsive et intuitive

##  Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Dépendances principales

```
PySide6>=6.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
google-generativeai>=0.3.0
python-dotenv>=0.19.0
fpdf>=1.7.2
```

## ⚙️ Configuration

### 1. Configurer l'API Gemini

#### Option A : Via l'interface (Recommandé)

1. Lancez l'application
2. Cliquez sur le bouton ** API KEY**
3. Entrez votre clé API Gemini
4. Le bouton devient vert quand configuré 

#### Option B : Via fichier `.env`

Créez un fichier `.env` à la racine du projet :

```env
GEMINI_API_KEY=votre_clé_api_ici
```

### 2. Obtenir une clé API Gemini

1. Visitez : [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Connectez-vous avec votre compte Google
3. Créez une nouvelle clé API
4. Copiez la clé (format : `AIzaSy...`)

##  Utilisation

### Lancer l'application

```bash
python main.py
```

### Mode Standard

1. Choisissez **MAXIMISER** ou **MINIMISER**
2. Entrez les coefficients de la fonction objectif (c₁, c₂)
3. Configurez les contraintes :
   - `a·x₁ + b·x₂ ≤ c`
   - Ajoutez/supprimez des contraintes avec les boutons
4. Cliquez sur ** RÉSOUDRE**
5. Visualisez les résultats et le graphique
6. **Exportez en PDF** si désiré

### Mode AI-TEXT 

1. Cliquez sur ** AI-TEXT**
2. Collez votre problème en texte libre :

```
Une entreprise disposant de 10 000 m² de carton en réserve, 
fabrique et commercialise 2 types de boîtes en carton. 
La fabrication d'une boîte de type 1 ou 2 requiert, 
respectivement, 1 et 2 m² de carton ainsi que 2 et 3 minutes 
de temps d'assemblage. Seules 200 heures de travail sont 
disponibles pendant la semaine à venir...
```

3. Cliquez sur ** EXTRAIRE AVEC GEMINI**
4. L'IA remplit automatiquement les champs
5. Vérifiez et ajustez si nécessaire
6. Cliquez sur ** RÉSOUDRE**

##  Structure du projet

```
PROJET-1/
├── main.py                     # Point d'entrée de l'application
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
├── pdf_export.py               # Export PDF
├── test_models.py              # tester les models 
│
├── core/                       # Logique métier
│   ├── optimizer.py           # Algorithmes d'optimisation
│   └── plotting.py            # Génération de graphiques
│
├── LLM_GEMINI/                # Module d'extraction IA
│   └── llm_extractor.py       # Interface avec Gemini API
│
├── ui/                        # Interface utilisateur
│   └── main_window.py         # Fenêtre principale
│
└── utils/                     # Utilitaires
    └── validators.py          # Validation des entrées
```

##  Technologies

### Frontend
- **PySide6** : Interface graphique Qt
- **Matplotlib** : Visualisation des graphiques

### Backend
- **SciPy** : Résolution de problèmes linéaires
- **NumPy** : Calculs numériques

### IA
- **Google Gemini 1.5 Flash** : Extraction automatique de texte

### Export
- **FPDF** : Génération de PDF

##  Exemples

### Exemple 1 : Problème de production

**Énoncé :**
```
Maximiser Z = 3x₁ + 2x₂
Sous contraintes :
  2x₁ + x₂ ≤ 18
  x₁ + 3x₂ ≤ 12
  x₁, x₂ ≥ 0
```

**Résultat :**
- x₁ = 7.2
- x₂ = 1.6
- Z = 24.8

### Exemple 2 : Avec AI-TEXT

**Input texte libre :**
```
Une entreprise fabrique deux produits A et B. Le produit A 
rapporte 50€ et le produit B 40€. La fabrication nécessite 
pour A : 2h de main d'œuvre et 3kg de matière première. 
Pour B : 1h et 2kg. On dispose de 100h et 120kg.
```

**L'IA extrait automatiquement :**
- Maximiser Z = 50x₁ + 40x₂
- 2x₁ + x₂ ≤ 100
- 3x₁ + 2x₂ ≤ 120

##  Signaler un bug

Ouvrez une [issue](https://github.com/achraf-bay/solveur-pl/issues) avec :
- Description du problème
- Étapes pour reproduire
- Comportement attendu vs observé
- Captures d'écran si applicable

##  TODO / Roadmap

- [ ] Support de plus de 2 variables
- [ ] Historique des problèmes résolus
- [ ] Export en Excel
- [ ] Mode hors ligne (sans IA)
- [ ] Support multilingue
- [ ] Méthode du simplexe avec étapes détaillées

##  Remerciements

- Google Gemini pour l'API d'extraction de texte
- Communauté Qt/PySide6
- Bibliothèque SciPy pour les algorithmes d'optimisation

 **Si ce projet vous a aidé, n'oubliez pas de lui donner une étoile !**

---

**Fait avec et Python**
