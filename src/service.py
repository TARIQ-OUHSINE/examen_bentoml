"""
Service BentoML de prédiction de la chance d'admission d'un étudiant.

Endpoints (requêtes HTTP POST, corps JSON) :
    /login   : {"username", "password"} -> {"token": <JWT>} ; HTTP 401 si les identifiants sont faux.
    /predict : variables de l'étudiant -> {"chance_of_admit": <float>} ; nécessite l'en-tête
               "Authorization: Bearer <JWT>", sinon HTTP 401.

La clé de signature des jetons est lue dans la variable d'environnement JWT_SECRET_KEY
(une valeur par défaut de démonstration est utilisée si elle n'est pas définie).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bentoml
import jwt
import pandas as pd
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "admission-service-demo-key")
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME = timedelta(minutes=30)

# Comptes autorisés à utiliser l'API (démonstration)
USERS = {"user123": "password123"}

PROTECTED_ROUTES = {"/predict"}

# Ordre et noms des variables utilisés à l'entraînement du modèle
FEATURES = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]


def create_token(username: str) -> str:
    """Crée un jeton JWT signé, valable TOKEN_LIFETIME."""
    now = datetime.now(timezone.utc)
    payload = {"sub": username, "iat": now, "exp": now + TOKEN_LIFETIME}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Refuse (HTTP 401) les requêtes vers les routes protégées sans jeton JWT valide."""

    async def dispatch(self, request, call_next):
        if request.url.path in PROTECTED_ROUTES:
            authorization = request.headers.get("Authorization", "")
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                return JSONResponse({"detail": "Missing or invalid authorization header"}, status_code=401)
            try:
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse({"detail": "Token has expired"}, status_code=401)
            except jwt.InvalidTokenError:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)
            request.state.user = payload.get("sub")
        return await call_next(request)


@bentoml.service(name="admission_prediction_service")
class AdmissionPredictionService:
    model_ref = bentoml.models.BentoModel("admission_lr:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.model_ref)

    @bentoml.api(route="/login")
    def login(self, username: str, password: str, ctx: bentoml.Context) -> dict:
        """Renvoie un jeton JWT si les identifiants sont corrects."""
        if USERS.get(username) != password:
            ctx.response.status_code = 401
            return {"detail": "Invalid credentials"}
        return {"token": create_token(username), "token_type": "bearer"}

    @bentoml.api(route="/predict")
    def predict(
        self,
        gre_score: Annotated[int, Field(ge=0, le=340, description="Score GRE (sur 340)")],
        toefl_score: Annotated[int, Field(ge=0, le=120, description="Score TOEFL (sur 120)")],
        university_rating: Annotated[int, Field(ge=1, le=5, description="Note de l'université (1 à 5)")],
        sop: Annotated[float, Field(ge=1, le=5, description="Statement of Purpose (1 à 5)")],
        lor: Annotated[float, Field(ge=1, le=5, description="Letter of Recommendation (1 à 5)")],
        cgpa: Annotated[float, Field(ge=0, le=10, description="Moyenne cumulée (sur 10)")],
        research: Annotated[int, Field(ge=0, le=1, description="Expérience de recherche (0 ou 1)")],
    ) -> dict:
        """Prédit la chance d'admission (entre 0 et 1) d'un étudiant."""
        features = pd.DataFrame(
            [[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]],
            columns=FEATURES,
        )
        prediction = float(self.model.predict(features)[0])
        # La régression linéaire n'est pas bornée : la chance d'admission est ramenée dans [0, 1]
        return {"chance_of_admit": round(min(max(prediction, 0.0), 1.0), 4)}


AdmissionPredictionService.add_asgi_middleware(JWTAuthMiddleware)
