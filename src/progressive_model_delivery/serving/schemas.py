"""pydantic request and response contracts of the router and the shadow worker.

The feature bounds mirror :data:`progressive_model_delivery.models.features.NUMERIC_FEATURES`;
``tests/unit/test_serving_units.py`` checks that the two definitions agree. Unknown fields are
refused, floats must be finite, and subject ids are bounded and restricted to a safe alphabet.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "FeedbackRequest",
    "FeedbackResponse",
    "PredictRequest",
    "PredictResponse",
    "RequestFeatures",
    "ShadowJob",
]

Region = Literal["north", "south", "east", "west"]
_HEX32 = r"^[0-9a-f]{32}$"


class RequestFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0, le=500)
    support_tickets: int = Field(ge=0, le=20)
    num_products: int = Field(ge=1, le=6)
    discount_rate: float = Field(ge=0, le=0.5)
    usage_trend: float = Field(ge=-1, le=1)
    region: Region


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:@-]+$")
    features: RequestFeatures


class PredictResponse(BaseModel):
    request_id: str
    score: float
    label: int
    model_version: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=_HEX32)
    label: Literal[0, 1]


class FeedbackResponse(BaseModel):
    status: Literal["accepted"]


class ShadowJob(BaseModel):
    """What the router sends to the shadow worker: no subject id, no credentials."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: str = Field(pattern=_HEX32)
    rollout_id: str = Field(pattern=_HEX32)
    stage_index: int = Field(ge=0, le=16)
    release_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_hash: str = Field(pattern=_HEX32)
    ts: float = Field(ge=0)
    features: RequestFeatures
