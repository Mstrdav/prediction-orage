# 🏆 Data Battle IA PAU 2026 – Projet Météorage

## 👥 Équipe
- Nom de l’équipe : Les Vieux Briscards
- Membres :
  - Colin Geindre
  - Alexandre Hayoun
  - Tao Levray

## 🎯 Problématique
Le projet vise à prédire avec précision la fin des épisodes orageux (alertes Météorage) à l'aide de l'Intelligence Artificielle et de techniques de data science. L'objectif est de modéliser temporellement et spatialement la dynamique des cellules orageuses pour optimiser les alertes de sécurité pour un ensemble de sites ciblés (ici, des aéroports).

## 💡 Solution proposée
Après une phase préliminaire d'exploration des données, puis une phase de comparaison de différents modèles, nous avons opté pour un processus de Hawkes, modèle adapté aux événements auto-excitants (généralement utilisés pour modéliser l'activité sismique ou les transactions financières).
Dans une phase complémentaire, nous avons chercher à améliorer le modèle en y ajoutant des composantes spatiales et temporelles complexes. Puis nous avons développé une architecture "Mixture of Experts" (MoE) spatialisé, qui active un de 3 modèles spécialisés en fonction de la direction générale de l'orage, principalement.
Nous avons renoncé à spécialiser les modèles par aéroport, car le gain était très marginal, et le coup en temps de calcul et donc en énergie pas rentable.
Une interface web interactive permet d'explorer visuellement le déroulé des sessions orageuses et l'évolution des prédictions.

## ⚙️ Stack technique
- **Langages** : Python, HTML, JavaScript, LaTeX
- **Frameworks** : PyTorch, Scikit-Learn, Pandas, Numpy
- **IA (Architecture Finale)** : Neural Hawkes Process, Spatial Mixture of Experts (MoE)
- **Outils** : Pew, Git, MikTeX, Jupyter Notebook, VSCodium, Antigravity
- **Outils d'IA Générative** : Auto-completion et génération de code (Gemini 3.1 Pro), relecture et feedback (Gemini 3 Flash)

## 🚀 Installation & exécution

### Prérequis
- Python 3.9+
- Pip ou Conda (recommandé)

### Installation
1. Cloner le dépôt :
```bash
git clone <url-du-dépôt>
cd prediction-orage
```
2. Créer un environnement virtuel (optionnel mais recommandé) :
```bash
python -m venv .venv
source .venv/bin/activate  # Sous Linux/Mac
# Ou sous Windows : .venv\Scripts\activate
```
3. Installer les dépendances :
```bash
pip install -r requirements.txt
```
4. **Données** :
Placer les données et notamment le fichier d'entraînement (`segment_alerts_all_airports_train.csv`) dans le dossier `data/`. (Dans le cas ou git ne laisse pas passer le fichier d'entrainement, trop gros)

### Exécution

- **Rapport** :
Les rapports sont disponibles au lien suivant : [EDA](https://mstrdav.github.io/prediction-orage/eda_report.pdf), [Comparaison de modèles](https://mstrdav.github.io/prediction-orage/model_report.pdf), [Amélioration du Neural Hawkes](https://mstrdav.github.io/prediction-orage/phase2bis_report.pdf).
- **Entraînements complets** :
Les entraînements des divers modèles Hawkes et spatiaux peuvent être rejoués à travers les scripts situés sous `src/` (ex: `python src/spatial_moe_model.py`).
- **Démonstration Web** :
La plateforme de démo interactive, rejouant certaines sessions orageuses et les prédictions associées, est accessible au lien suivant : [https://mstrdav.github.io/prediction-orage/](https://mstrdav.github.io/prediction-orage/).
