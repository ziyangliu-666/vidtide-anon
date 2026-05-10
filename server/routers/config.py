"""Config API router: read and update pipeline.yaml."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yaml

router = APIRouter(tags=["config"])

_CONFIG_PATH = "config/pipeline.yaml"


class ConfigUpdate(BaseModel):
    yaml: str


@router.get("/config")
def get_config():
    """Return the current pipeline config as YAML string and parsed dict."""
    try:
        with open(_CONFIG_PATH) as f:
            raw = f.read()
        return {"yaml": raw, "parsed": yaml.safe_load(raw)}
    except FileNotFoundError:
        return {"yaml": "# No config file found", "parsed": {}}


@router.put("/config")
def put_config(body: ConfigUpdate):
    """Validate and write new pipeline config."""
    # Parse YAML
    try:
        parsed = yaml.safe_load(body.yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Config must be a YAML mapping")

    # Require top-level keys
    for key in ("crawl", "filter"):
        if key not in parsed:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required top-level key: '{key}'",
            )

    # Write atomically-ish
    with open(_CONFIG_PATH, "w") as f:
        f.write(body.yaml)

    return {"success": True}
