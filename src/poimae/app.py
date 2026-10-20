"""FastAPI app - everything runs on localhost, nothing is uploaded anywhere.

Endpoints
---------
GET  /api/health                liveness + which municipalities are loaded
GET  /api/municipalities        list of bundled rule files (with provenance)
GET  /api/municipalities/{id}   one file's summary (categories, counts, source)
GET  /api/lookup                ?municipality=&q=       text lookup
POST /api/photo                 multipart "image"       photo -> shortlist
POST /api/week                  {municipality, schedule, today?} -> this week
GET  /api/ambiguity             the cross-city evidence report

The photo endpoint is the only one that touches image bytes.  It decodes them in
memory, computes features, and drops them - no write, no cache, no logger that
would capture the payload.
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .advisor import advise_photo, advise_text
from .calendar import ScheduleError, week_plan
from .crosscity import ambiguity_report
from .features import ImageError
from .municipalities import MunicipalityError, Registry

STATIC_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent.parent / "static"

#: 8 MB is far above any phone photo we expect and far below "an accident".
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = FastAPI(
    title="PoiMae (ポイ前)",
    version="1.0.0",
    description="捨てる前に、ひと呼吸。写真で候補を絞り、自治体の公式辞典で区分を決めるローカルアプリ。",
)

_registry: Registry | None = None


def registry() -> Registry:
    """Load the rule files once, on first use."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def _municipality(municipality_id: str):
    try:
        return registry().get(municipality_id)
    except MunicipalityError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class WeekRequest(BaseModel):
    municipality: str = Field(..., description="municipality id, e.g. sapporo")
    schedule: dict[str, list[int]] = Field(
        default_factory=dict,
        description="category label -> ISO weekdays (1=Mon .. 7=Sun)",
    )
    today: str | None = Field(
        default=None, description="ISO date override, for tests and demos"
    )


@app.get("/api/health")
def health() -> dict:
    reg = registry()
    return {
        "status": "ok",
        "municipalities": reg.ids(),
        "version": app.version,
        "network": "disabled - this app makes no outbound requests",
    }


@app.get("/api/municipalities")
def list_municipalities() -> dict:
    reg = registry()
    return {"municipalities": [m.to_summary() for m in reg.all()]}


@app.get("/api/municipalities/{municipality_id}")
def get_municipality(municipality_id: str) -> dict:
    return _municipality(municipality_id).to_summary()


@app.get("/api/lookup")
def lookup(
    municipality: str = Query(..., description="municipality id"),
    q: str = Query(..., min_length=1, max_length=80, description="item name"),
) -> dict:
    _municipality(municipality)
    return advise_text(registry(), municipality, q)


@app.post("/api/photo")
async def photo(
    image: UploadFile = File(..., description="png/jpeg/webp/bmp"),
    municipality: str = Query(..., description="municipality id"),
) -> dict:
    _municipality(municipality)
    if image.content_type and image.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type {image.content_type!r}; allowed: {sorted(ALLOWED_MIME)}",
        )
    payload = await image.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    try:
        return advise_photo(registry(), municipality, payload)
    except ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/week")
def week(request: WeekRequest) -> dict:
    municipality = _municipality(request.municipality)
    today = date.today()
    if request.today:
        try:
            today = date.fromisoformat(request.today)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"bad date: {exc}") from exc
    try:
        return week_plan(
            request.schedule, today, allowed_categories=municipality.category_labels
        )
    except ScheduleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ambiguity")
def ambiguity() -> dict:
    return ambiguity_report(registry())


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# Mounted last so it never shadows /api/*.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(MunicipalityError)
def _municipality_handler(_request, exc: MunicipalityError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
