# Rapport de Projet MLOps - Partie 2
## Architecture MLOps et Choix Techniques

### 1. Architecture Globale
L'architecture de notre pipeline MLOps est conçue pour être modulaire, évolutive et automatisée. Elle repose sur deux piliers principaux : la conteneurisation avec **Docker** et l'intégration/déploiement continu (CI/CD) avec **GitHub Actions**.

#### Composants
*   **API (FastAPI)** : Le cœur de l'application, exposant le modèle de prédiction via une interface REST.
*   **Base de Données (MongoDB)** : Utilisée pour stocker les logs de prédiction et assurer la persistance des données.
*   **Interface d'Administration (Mongo Express)** : Permet une visualisation simple des données en développement.

### 2. Choix Techniques

#### Docker & Docker Compose
Nous avons choisi Docker pour garantir la reproductibilité de l'environnement d'exécution.
*   **Polyvalence** : Le `docker-compose.yml` orchestre plusieurs services (API, DB, UI) dans un réseau isolé (`sentiment-network`), facilitant le déploiement local et la gestion des dépendances.
*   **Persistance** : L'utilisation de volumes Docker (`models-data`, `logs-data`) assure que les modèles entraînés et les logs ne sont pas perdus lors du redémarrage des conteneurs.

#### GitHub Actions (CI/CD)
GitHub Actions a été retenu pour son intégration native avec le dépôt de code.
*   **Workflow en Cascade** : Nous avons implémenté une chaîne de dépendance stricte :
    1.  `test.yml` : Vérifie la qualié du code (Linting) et sa justesse (Tests unitaires).
    2.  `evaluate.yml` : Ne se lance que si les tests passent. Il valide la performance du modèle.
    3.  `build.yml` : Ne construit et publie l'image Docker que si l'évaluation est satisfaisante.

### 3. Description du Workflow Automatisé

Le pipeline automatisé suit les étapes suivantes à chaque `push` sur la branche principale :

1.  **Test & Linting (`test.yml`)** :
    *   Installation de l'environnement Python.
    *   Vérification du style de code avec `flake8`, `black` et `isort`.
    *   Exécution des tests unitaires via `pytest` avec rapport de couverture.

2.  **Évaluation du Modèle (`evaluate.yml`)** :
    *   *Condition* : Déclenché uniquement après le succès de `test.yml`.
    *   Exécution d'un script d'évaluation sur un jeu de données de test.
    *   Comparaison de la métrique (Accuracy) avec un seuil défini (e.g., 0.50).
    *   Si la performance est insuffisante, le pipeline s'arrête ici (`FAIL`).

3.  **Construction et Déploiement (`build.yml`)** :
    *   *Condition* : Déclenché uniquement après le succès de `evaluate.yml`.
    *   Construction de l'image Docker optimisée.
    *   Publication automatique sur Docker Hub avec les tags `latest` et le SHA du commit pour la traçabilité.

---

**Auteurs :**
*   [Votre Nom]
*   [Nom du Binôme si applicable]

**Lien du Répository GitHub :**
[Insérer le lien ici]
