# Examen BentoML : API de prédiction d'admission

Service BentoML qui prédit la chance d'admission d'un étudiant dans une université (régression linéaire
sur GRE, TOEFL, note de l'université, SOP, LOR, CGPA et expérience de recherche). L'API est sécurisée par JWT :

| Endpoint | Méthode | Corps JSON | Réponse |
|---|---|---|---|
| `/login` | POST | `{"username": "user123", "password": "password123"}` | `{"token": "<JWT>", "token_type": "bearer"}`, ou 401 `{"detail": "Invalid credentials"}` |
| `/predict` | POST | `{"gre_score": 320, "toefl_score": 110, "university_rating": 4, "sop": 4.0, "lor": 4.5, "cgpa": 9.0, "research": 1}` + en-tête `Authorization: Bearer <JWT>` | `{"chance_of_admit": 0.8}`, 401 sans jeton valide (absent, invalide ou expiré), 400 si les données sont invalides |

Les jetons sont signés en HS256 et valables 30 minutes. La clé de signature est lue dans la variable
d'environnement `JWT_SECRET_KEY` ; sans elle, une clé de démonstration est utilisée par le service et par les tests.

## Lancer le service et les tests (rendu)

Prérequis : Docker. Les commandes sont à exécuter dans l'ordre, depuis le dossier qui contient
`admission_prediction_service.tar`, `requirements.txt` et `tests/`.

1. Charger l'image Docker :

```bash
docker load -i admission_prediction_service.tar
```

2. Lancer le service sur le port 3000 :

```bash
docker run -d --rm --name admission_service -p 3000:3000 admission_prediction_service:latest
```

3. Lancer les tests (ils attendent que le service soit prêt, puis l'interrogent sur http://localhost:3000).
   Les tests tournent dans un conteneur Python 3.11, quelle que soit la version de Python de la machine :

```bash
docker run --rm --network host -v "$(pwd)":/work -w /work python:3.11-slim \
  sh -c "pip install -q -r requirements.txt && pytest -v -p no:cacheprovider tests/"
```

4. Arrêter le service :

```bash
docker stop admission_service
```

Les tests (`tests/test_service.py`) vérifient :

- l'authentification JWT : refus d'un jeton absent, mal formé, signé avec une autre clé ou expiré, et acceptation d'un jeton valide ;
- `/login` : jeton JWT valide (utilisateur, date d'expiration) pour des identifiants corrects, 401 pour des identifiants incorrects ;
- `/predict` : 401 sans jeton valide, prédiction comprise entre 0 et 1 pour des données correctes, prédiction plus
  élevée pour un meilleur profil, et erreur pour des données invalides (champ manquant, mauvais type, valeur hors bornes).

## Reconstruire le projet depuis les sources

Code source : https://github.com/TARIQ-OUHSINE/examen_bentoml. Avec Python 3.11 et Docker, depuis la racine du dépôt :

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
wget https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv -O data/raw/admission.csv
python src/prepare_data.py
python src/train_model.py
bentoml models list
bentoml serve src.service:AdmissionPredictionService --port 3000
bentoml build
bentoml containerize admission_prediction_service:latest --image-tag admission_prediction_service:latest
docker save admission_prediction_service:latest -o admission_prediction_service.tar
```

- `src/prepare_data.py` nettoie les données (suppression de `Serial No.`, des espaces dans les noms de colonnes,
  des doublons et des valeurs manquantes) et les découpe en 80 % d'entraînement et 20 % de test dans `data/processed/`.
- `src/train_model.py` entraîne une régression linéaire précédée d'une normalisation (`StandardScaler`), affiche
  R², RMSE et MAE sur le jeu de test, puis enregistre le modèle dans le Model Store de BentoML (`admission_lr`).
- `src/service.py` définit le service `admission_prediction_service` et son middleware d'authentification JWT.
- `bentofile.yaml` décrit le bento : service, code inclus, modèle et dépendances (image `python:3.11-slim`).

`pydantic` est fixé à 2.12.5 : la version 2.12.1 citée dans l'énoncé dépend de `pydantic-core` 2.41.3,
retiré de PyPI (fichier corrompu), et ne peut plus être installée.
