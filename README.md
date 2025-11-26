# SoundFL / FedDalf – Federated Domain Adaptation & Lifelong Learning for Audio

## 🇫🇷 Présentation
Notebook Colab démontrant FedDalf (fédération + adaptation de domaine + apprentissage continu) pour la classification audio. Il simule plusieurs clients via Flower et entraîne un CNN Keras sur des caractéristiques audio (ex. UrbanSound8K) stockées en `.npy` dans Google Drive.

## Contenu du dépôt
- `SoundFL_All.ipynb` : pipeline complet (dépendances, préparation des données, filtrage, modèle, stratégie Flower, simulation).

## Prérequis
- Compte Google + runtime Colab (GPU recommandé).
- Données `.npy` par client dans Google Drive (`/content/drive/MyDrive/...`) avec features et labels alignés. Exemple de dossiers client : `/content/drive/MyDrive/numpyDataset`, `/content/drive/MyDrive/urbansound8k`.
- Python 3.11+/TensorFlow 2.x côté Colab. Dépendances principales : `flwr[simulation]`, `tensorflow`, `imgaug==0.4.0`, `numpy==1.26.4`, `cryptography==44.0.3`, `matplotlib`, `smote_variants`, `tfds-nightly`, `scipy`.

## Guide rapide
1) Ouvrir `SoundFL_All.ipynb` dans Colab.
2) Exécuter la cellule d’installation (désinstalle/installe des versions épinglées pour éviter les conflits). Si un redémarrage est demandé, redémarrer puis relancer la cellule.
3) Monter Google Drive (cellule `drive.mount`).
4) Vérifier/adapter `client_folders` pour pointer vers vos dossiers `.npy`.
5) Lancer la préparation des données : split train/test, reshape en `(16, 8, 1)`, sélection des classes cibles.
6) Ajuster les hyperparamètres dans la section Configuration (ex. `NUM_CLIENTS`, `NUM_ROUNDS`, `FRACTION_CLIENTS`, `indexed_slices`).
7) Exécuter la cellule de simulation Flower (`fl.simulation.start_simulation`) pour démarrer FedAvg personnalisé + journalisation.

## Points clés du notebook
- Filtrage des clients contenant certaines classes (`indexed_slices=[1,3,6,8]`) et rééquilibrage.
- Gestion des labels manquants et pseudo-étiquetage (`disturb_labels`, `map_predict`, `update_y_train`).
- Modèle CNN compact (2 conv + dense 1024) sur entrées 16x8.
- Stratégie Flower dérivée de FedAvg : sélection/trace des clients, agrégation pondérée, sauvegarde des métriques locales/globales round par round.
- Historique et modèles sauvegardés dans `/content/drive/MyDrive/FEDADL/history/evaluation/`.

## Personnalisation
- Chemins données : modifier `client_folders` et `initial_path_all_users`.
- Classes cibles : changer `indexed_slices`.
- Modèle : ajuster `input_dim`, couches, `base_learning_rate`.
- Simulation : `NUM_CLIENTS`, `FRACTION_CLIENTS`, `NUM_ROUNDS`, `EPOCHS`, `batch_size` dans `fit_config`.

## Résultats attendus
- Fichiers texte par client avec courbes train/val, métriques locales/globales.
- Modèle fédéré sauvegardé en fin de training (si autorisations d’écriture dans Drive).

## À faire / idées
- Ajouter une section d’évaluation centrale sur un jeu global.
- Intégrer une vraie boucle FedDalf (adaptation domaine + apprentissage continu) ou visualisation des distributions.
- Publier un script de préparation des features audio (MFCC/log-mels) pour générer les `.npy`.

---

## 🇬🇧 Overview
Colab notebook showcasing FedDalf (federated learning + domain adaptation + lifelong learning) for audio classification. It simulates multiple clients with Flower and trains a Keras CNN on per-client `.npy` features stored in Google Drive.

## Repository Content
- `SoundFL_All.ipynb`: end-to-end pipeline (deps, data prep, filtering, model, Flower strategy, simulation).

## Requirements
- Google account + Colab runtime (GPU recommended).
- Per-client `.npy` feature/label files in Google Drive (e.g., `/content/drive/MyDrive/numpyDataset`, `/content/drive/MyDrive/urbansound8k`).
- Core libs: `flwr[simulation]`, `tensorflow`, `imgaug==0.4.0`, `numpy==1.26.4`, `cryptography==44.0.3`, `matplotlib`, `smote_variants`, `tfds-nightly`, `scipy`.

## Quickstart
1) Open `SoundFL_All.ipynb` in Colab.  
2) Run the dependency cell (it pins versions; restart runtime if prompted and rerun).  
3) Mount Drive (`drive.mount`).  
4) Point `client_folders` to your `.npy` data.  
5) Run data prep: train/test split, reshape to `(16, 8, 1)`, select target classes.  
6) Tune hyperparameters in the Configuration section (`NUM_CLIENTS`, `NUM_ROUNDS`, `FRACTION_CLIENTS`, `indexed_slices`).  
7) Run the Flower simulation (`fl.simulation.start_simulation`) to start training + logging.

## Notebook Highlights
- Client filtering on selected classes (`indexed_slices=[1,3,6,8]`); distribution checks.
- Missing-label handling and pseudo-labeling helpers (`disturb_labels`, `map_predict`, `update_y_train`).
- Compact CNN (2 conv + dense) for 16x8 inputs.
- Custom FedAvg-based strategy: client selection tracing, weighted aggregation, per-round metric logging.
- Outputs saved under `/content/drive/MyDrive/FEDADL/history/evaluation/`.

## Customization
- Data paths: edit `client_folders`, `initial_path_all_users`.
- Target classes: change `indexed_slices`.
- Model: tweak `input_dim`, layers, `base_learning_rate`.
- Simulation: set `NUM_CLIENTS`, `FRACTION_CLIENTS`, `NUM_ROUNDS`, `EPOCHS`, `batch_size` in `fit_config`.

## Expected Outputs
- Text logs per client with train/val history and global metrics.
- Final federated model saved to Drive (if permissions allow).

## Next ideas
- Add central/global evaluation set.
- Flesh out full FedDalf domain-adaptation loop or add distribution visualizations.
- Provide an audio feature extraction script (MFCC/log-mel) to generate the `.npy` inputs.