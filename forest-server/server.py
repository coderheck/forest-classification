# server script

import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import folium
from folium.plugins import HeatMap
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import math

DB_FILE = "forest.db"
API_KEY = os.environ.get("FOREST_API_KEY")

if not API_KEY:
    raise RuntimeError("Set the FOREST_API_KEY environment variable first.")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(records)"
            ).fetchall()
        }

        if not columns:
            conn.execute("""
                CREATE TABLE records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL,
                    recorded_at TEXT,
                    latitude REAL,
                    longitude REAL,
                    density TEXT,
                    confidence REAL
                )
            """)
            return

        # Migrate the old database if it still contains
        # device_id or telemetry, preserving useful records.
        required = {
            "id", "received_at", "recorded_at",
            "latitude", "longitude", "density", "confidence"
        }

        if columns != required:
            conn.execute("""
                CREATE TABLE records_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_at TEXT NOT NULL,
                    recorded_at TEXT,
                    latitude REAL,
                    longitude REAL,
                    density TEXT,
                    confidence REAL
                )
            """)

            def expr(column, fallback="NULL"):
                return column if column in columns else fallback

            received_expr = expr("received_at", "NULL")
            recorded_expr = expr("recorded_at")
            latitude_expr = expr("latitude")
            longitude_expr = expr("longitude")
            density_expr = expr("density")
            confidence_expr = expr("confidence")
            id_expr = expr("id", "NULL")

            conn.execute(f"""
                INSERT INTO records_new (
                    id, received_at, recorded_at,
                    latitude, longitude, density, confidence
                )
                SELECT
                    {id_expr},
                    COALESCE({received_expr}, ?),
                    {recorded_expr},
                    {latitude_expr},
                    {longitude_expr},
                    {density_expr},
                    {confidence_expr}
                FROM records
            """, (now_utc(),))

            conn.execute("DROP TABLE records")
            conn.execute(
                "ALTER TABLE records_new RENAME TO records"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Forest Telemetry API", lifespan=lifespan)


class Record(BaseModel):
    recorded_at: datetime | None = None
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


def check_key(x_api_key: str | None):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "service": "forest-telemetry"}


# ESP32: save a GPS fix without a forest classification.
@app.post("/api/records", status_code=201)
def add_record(
    record: Record,
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    received_at = now_utc()
    recorded_at = (
        record.recorded_at.isoformat()
        if record.recorded_at is not None
        else None
    )

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("""
            INSERT INTO records (
                received_at, recorded_at,
                latitude, longitude, density, confidence
            ) VALUES (?, ?, ?, ?, NULL, NULL)
        """, (
            received_at,
            recorded_at,
            record.latitude,
            record.longitude,
        ))

        record_id = cursor.lastrowid

    return {
        "status": "stored",
        "id": record_id,
        "received_at": received_at,
    }


# Laptop: associate a YOLO result with the latest unclassified GPS fix.
@app.post("/api/annotate-latest", status_code=201)
def annotate_latest(
    density: Literal["dense", "medium", "sparse", "unknown"],
    confidence: float = Query(ge=0, le=1),
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    timestamp = now_utc()

    with sqlite3.connect(DB_FILE) as conn:
        gps = conn.execute("""
            SELECT id, recorded_at
            FROM records
            WHERE density IS NULL
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        if gps is None:
            raise HTTPException(
                status_code=404,
                detail="No unclassified GPS fix is available",
            )

        gps_id, gps_timestamp = gps

        cursor = conn.execute("""
            UPDATE records
            SET density = ?,
                confidence = ?,
                recorded_at = COALESCE(recorded_at, ?)
            WHERE id = ?
              AND density IS NULL
        """, (
            density,
            confidence,
            timestamp,
            gps_id,
        ))

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=409,
                detail="GPS record was already classified",
            )

    return {
        "status": "stored",
        "id": gps_id,
        "recorded_at": gps_timestamp or timestamp,
        "density": density,
        "confidence": confidence,
    }


@app.get("/api/records")
def list_records(
    limit: int = Query(default=100, ge=1, le=1000),
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, received_at, recorded_at,
                   latitude, longitude, density, confidence
            FROM records
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]



# Forest density score for each YOLO class.
DENSITY_SCORES = {
    "sparse": 0,
    "medium": 50,
    "dense": 100,
}

# Web Mercator projection, in meters.
EARTH_RADIUS = 6378137.0


def project_coordinates(latitude, longitude):
    latitude = max(-85.05112878, min(85.05112878, latitude))

    x = EARTH_RADIUS * math.radians(longitude)
    y = EARTH_RADIUS * math.log(
        math.tan(math.pi / 4 + math.radians(latitude) / 2)
    )

    return x, y


def unproject_coordinates(x, y):
    longitude = math.degrees(x / EARTH_RADIUS)
    latitude = math.degrees(
        2 * math.atan(math.exp(y / EARTH_RADIUS)) - math.pi / 2
    )

    return latitude, longitude


def score_color(score):
    """Green = dense, yellow = medium, red = sparse."""
    if score <= 50:
        ratio = score / 50
        start = (215, 25, 28)       # Red: sparse
        end = (255, 255, 0)       # Yellow: medium
    else:
        ratio = (score - 50) / 50
        start = (255, 255, 0)     # Yellow: medium
        end = (44, 162, 95)          # Green: dense

    rgb = tuple(
        round(a + (b - a) * ratio)
        for a, b in zip(start, end)
    )

    return "#{:02x}{:02x}{:02x}".format(*rgb)


@app.get("/heatmap", response_class=HTMLResponse)
def forest_heatmap(
    cell_size: float = Query(default=20, ge=5, le=200), # min cạnh 5m (25m^2), max 200m (40000m^2 = 4ha)
    fill_gaps: bool = True,
    smoothing_radius: int = Query(default=1, ge=0, le=3),
    start: str | None = None,
    end: str | None = None,
):
    # Only classes with a defined numerical density score contribute.
    query = """
        SELECT latitude, longitude, density, confidence,
               recorded_at, received_at
        FROM records
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND density IN ('sparse', 'medium', 'dense')
    """
    params = []

    if start:
        query += " AND COALESCE(recorded_at, received_at) >= ?"
        params.append(start)

    if end:
        query += " AND COALESCE(recorded_at, received_at) <= ?"
        params.append(end)

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return HTMLResponse(
            "<h2>No scored forest observations match these filters.</h2>"
        )

    # Aggregate observations into fixed-size projected grid cells.
    cells = {}

    for row in rows:
        x, y = project_coordinates(row["latitude"], row["longitude"])

        ix = math.floor(x / cell_size)
        iy = math.floor(y / cell_size)
        key = (ix, iy)

        score = DENSITY_SCORES[row["density"]]
        confidence = row["confidence"]

        # Avoid a zero weight when confidence is zero or missing.
        weight = max(0.05, confidence if confidence is not None else 0.5)

        if key not in cells:
            cells[key] = {
                "weighted_score": 0.0,
                "total_weight": 0.0,
                "count": 0,
            }

        cells[key]["weighted_score"] += score * weight
        cells[key]["total_weight"] += weight
        cells[key]["count"] += 1

    for cell in cells.values():
        cell["score"] = (
            cell["weighted_score"] / cell["total_weight"]
        )

    # Estimate only unobserved cells near measured cells.
    # Observed cells are never overwritten by estimates.
    estimated = {}

    if fill_gaps and smoothing_radius > 0:
        for ix, iy in cells:
            for dx in range(-smoothing_radius, smoothing_radius + 1):
                for dy in range(-smoothing_radius, smoothing_radius + 1):
                    if dx == 0 and dy == 0:
                        continue

                    target = (ix + dx, iy + dy)

                    if target in cells:
                        continue

                    distance_squared = dx * dx + dy * dy
                    if distance_squared > smoothing_radius * smoothing_radius:
                        continue

                    source = cells[(ix, iy)]
                    weight = source["total_weight"] / distance_squared

                    if target not in estimated:
                        estimated[target] = {
                            "weighted_score": 0.0,
                            "total_weight": 0.0,
                        }

                    estimated[target]["weighted_score"] += (
                        source["score"] * weight
                    )
                    estimated[target]["total_weight"] += weight

        for cell in estimated.values():
            cell["score"] = (
                cell["weighted_score"] / cell["total_weight"]
            )

    # Center the map on actual observations, not estimated cells.
    center = [
        sum(row["latitude"] for row in rows) / len(rows),
        sum(row["longitude"] for row in rows) / len(rows),
    ]

    m = folium.Map(location=center, zoom_start=17)

    layer = folium.FeatureGroup(
        name=f"Forest density grid ({cell_size:g} m cells)"
    )

    # Draw measured cells.
    for (ix, iy), cell in cells.items():
        south, west = unproject_coordinates(
            ix * cell_size, iy * cell_size
        )
        north, east = unproject_coordinates(
            (ix + 1) * cell_size, (iy + 1) * cell_size
        )

        color = score_color(cell["score"])

        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.70,
            tooltip=(
                f"Measured cell | Score: {cell['score']:.1f}/100"
                f" | Observations: {cell['count']}"
            ),
            popup=(
                f"Status: Measured<br>"
                f"Density score: {cell['score']:.1f}/100<br>"
                f"Observations: {cell['count']}<br>"
                f"Grid size: {cell_size:g} m"
            ),
        ).add_to(layer)

    # Draw estimated cells with lower opacity so they are distinguishable.
    for (ix, iy), cell in estimated.items():
        south, west = unproject_coordinates(
            ix * cell_size, iy * cell_size
        )
        north, east = unproject_coordinates(
            (ix + 1) * cell_size, (iy + 1) * cell_size
        )

        color = score_color(cell["score"])

        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.25,
            dash_array="4",
            tooltip=f"Estimated cell | Score: {cell['score']:.1f}/100",
            popup=(
                f"Status: Estimated from nearby measured cells<br>"
                f"Estimated density score: {cell['score']:.1f}/100<br>"
                f"Grid size: {cell_size:g} m"
            ),
        ).add_to(layer)

    layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = """
    <div style="
        position: fixed; bottom: 25px; left: 25px; z-index: 9999;
        background: white; padding: 10px 14px; border: 1px solid #888;
        border-radius: 5px; font-size: 13px;">
        <b>Forest density score / Độ phủ rừng</b><br>
        <span style="color:#d7191c">■</span> 0 - Sparse / Thưa<br>
        <span style="color:#ffff00">■</span> 50 - Medium / Vừa<br>
        <span style="color:#2ca25f">■</span> 100 - Dense / Dày<br>
        Solid cells: observed / Ô nét liền: đã đo<br>
        Transparent dashed cells: estimated / Ô nét đứt: ước tính từ các ô xung quanh
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend))

    return HTMLResponse(content=m.get_root().render())
