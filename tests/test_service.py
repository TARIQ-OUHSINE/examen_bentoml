"""
Tests du service de prédiction d'admission (le service doit être lancé, par défaut sur http://localhost:3000).

Variables d'environnement :
    SERVICE_URL    : adresse du service (défaut : http://localhost:3000) ;
    JWT_SECRET_KEY : clé de signature des jetons, identique à celle du service (défaut : celle du service).
"""

import os
import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import requests

BASE_URL = os.environ.get("SERVICE_URL", "http://localhost:3000")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "admission-service-demo-key")
JWT_ALGORITHM = "HS256"

VALID_USER = {"username": "user123", "password": "password123"}
VALID_INPUT = {
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.0,
    "lor": 4.5,
    "cgpa": 9.0,
    "research": 1,
}
TIMEOUT = 10


def make_token(secret=JWT_SECRET_KEY, expires_in=timedelta(minutes=5), username="user123"):
    now = datetime.now(timezone.utc)
    payload = {"sub": username, "iat": now, "exp": now + expires_in}
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def post(route, payload, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.post(f"{BASE_URL}{route}", json=payload, headers=headers, timeout=TIMEOUT)


@pytest.fixture(scope="session", autouse=True)
def service_ready():
    """Attend que le service réponde (60 secondes au maximum)."""
    for _ in range(60):
        try:
            if requests.get(f"{BASE_URL}/readyz", timeout=2).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    pytest.fail(f"Le service {BASE_URL} n'est pas disponible")


@pytest.fixture
def token():
    response = post("/login", VALID_USER)
    assert response.status_code == 200
    return response.json()["token"]


# --------------------------------------------------------------------------- #
# Authentification JWT
# --------------------------------------------------------------------------- #
def test_jwt_missing_token_is_rejected():
    response = post("/predict", VALID_INPUT)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid authorization header"


def test_jwt_malformed_token_is_rejected():
    response = post("/predict", VALID_INPUT, token="not-a-jwt")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token"


def test_jwt_token_signed_with_wrong_key_is_rejected():
    response = post("/predict", VALID_INPUT, token=make_token(secret="wrong-key"))
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token"


def test_jwt_expired_token_is_rejected():
    response = post("/predict", VALID_INPUT, token=make_token(expires_in=timedelta(minutes=-1)))
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"


def test_jwt_valid_token_is_accepted():
    response = post("/predict", VALID_INPUT, token=make_token())
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# API de connexion
# --------------------------------------------------------------------------- #
def test_login_returns_valid_jwt_for_correct_credentials():
    response = post("/login", VALID_USER)
    assert response.status_code == 200
    payload = jwt.decode(response.json()["token"], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == VALID_USER["username"]
    assert payload["exp"] > datetime.now(timezone.utc).timestamp()


@pytest.mark.parametrize("credentials", [
    {"username": "user123", "password": "wrong"},
    {"username": "unknown", "password": "password123"},
])
def test_login_returns_401_for_incorrect_credentials(credentials):
    response = post("/login", credentials)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


# --------------------------------------------------------------------------- #
# API de prédiction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_token", [None, "invalid.token.value"])
def test_predict_returns_401_without_valid_token(bad_token):
    response = post("/predict", VALID_INPUT, token=bad_token)
    assert response.status_code == 401


def test_predict_returns_valid_prediction(token):
    response = post("/predict", VALID_INPUT, token=token)
    assert response.status_code == 200
    chance = response.json()["chance_of_admit"]
    assert isinstance(chance, float)
    assert 0.0 <= chance <= 1.0


def test_predict_is_higher_for_a_stronger_profile(token):
    weak = {**VALID_INPUT, "gre_score": 295, "toefl_score": 95, "cgpa": 7.0, "research": 0}
    strong = {**VALID_INPUT, "gre_score": 335, "toefl_score": 118, "cgpa": 9.7}
    weak_chance = post("/predict", weak, token=token).json()["chance_of_admit"]
    strong_chance = post("/predict", strong, token=token).json()["chance_of_admit"]
    assert strong_chance > weak_chance


@pytest.mark.parametrize("invalid_input", [
    {key: value for key, value in VALID_INPUT.items() if key != "cgpa"},   # champ manquant
    {**VALID_INPUT, "gre_score": "high"},                                   # mauvais type
    {**VALID_INPUT, "gre_score": 400},                                      # hors bornes
    {**VALID_INPUT, "research": 2},                                         # hors bornes
])
def test_predict_returns_error_for_invalid_input(token, invalid_input):
    response = post("/predict", invalid_input, token=token)
    assert response.status_code in (400, 422)
