import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from model_utils import load_model, make_inference


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    day_of_week: str = Field(min_length=1)
    month: str = Field(min_length=1)
    weather_main: str = Field(min_length=1)
    temperature_c: float = Field(gt=-273.15, allow_inf_nan=False)
    rain_1h: float = Field(ge=0, allow_inf_nan=False)
    clouds_all: float = Field(ge=0, le=100, allow_inf_nan=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("The environment variable MODEL_PATH is empty!")
    app.state.model = load_model(model_path)
    yield


app = FastAPI(title="Hourly traffic inference service", lifespan=lifespan)
bearer_scheme = HTTPBearer(auto_error=False)


async def is_token_correct(token: str) -> bool:
    return token == "00000"


async def check_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme),
) -> None:
    if credentials is None:
        detail = "Not authenticated"
    elif not await is_token_correct(credentials.credentials):
        detail = "Invalid authentication credentials"
    else:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions", dependencies=[Depends(check_token)])
def predictions(instance: Instance) -> dict[str, float]:
    return make_inference(app.state.model, instance.model_dump())
