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
import json

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


@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Forest Monitoring</title>
            <style>
                body, p {
                    margin: 0;
                    background: powderblue;
                    color: #000000;
                    font-family: Georgia, sans-serif;
                }
            </style>
        </head>
        <body>
            <h1>hello forest!</h1>
            <p>
                <a href="/docs">all endpoints / hướng dẫn tất cả truy vấn</a> <br>
                <a href="/heatmap">forest heatmap / bản đồ nhiệt rừng</a> <br>
                <a href="/heatmap?cell_size=10">10m grid, with gap fill / ô cạnh 10m, dự đoán ô trống từ các ô xung quanh</a> <br>
                <a href="/heatmap?cell_size=10&fill_gaps=false">10m grid, no gap fill / ô 10m, không dự đoán ô trống</a> <br>
            </p>
            <br>
        </body>
    </html>
    """


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


# =========================
# CARBON ESTIMATION
# =========================

TREE_TYPES = {
    "acacia": "Acacia",
    "tea": "Tea plant",
    "fokienia": "Fokienia",
    "birch": "Birch",
}

# Editable defaults.
# Units: tonnes of CO2 equivalent per hectare per year.
# Replace these zeroes with researched coefficients later.
CARBON_COEFFICIENTS = {
    "acacia": {
        "sparse": 0.0,
        "medium": 0.0,
        "dense": 0.0,
    },
    "tea": {
        "sparse": 0.0,
        "medium": 0.0,
        "dense": 0.0,
    },
    "fokienia": {
        "sparse": 0.0,
        "medium": 0.0,
        "dense": 0.0,
    },
    "birch": {
        "sparse": 0.0,
        "medium": 0.0,
        "dense": 0.0,
    },
}


def interpolate_carbon_coefficient(score, coefficients):
    """
    Interpolate the coefficient using the existing density score.

    0   = sparse
    50  = medium
    100 = dense
    """

    sparse = coefficients["sparse"]
    medium = coefficients["medium"]
    dense = coefficients["dense"]

    if score <= 50:
        ratio = score / 50
        return sparse + (medium - sparse) * ratio

    ratio = (score - 50) / 50
    return medium + (dense - medium) * ratio


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

    carbon_cells = []

    # Draw measured cells.
    for (ix, iy), cell in cells.items():
        south, west = unproject_coordinates(
            ix * cell_size, iy * cell_size
        )
        north, east = unproject_coordinates(
            (ix + 1) * cell_size, (iy + 1) * cell_size
        )

        color = score_color(cell["score"])

        rectangle = folium.Rectangle(
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
        ).add_to(layer)

        carbon_cells.append({
            "layer": rectangle.get_name(),
            "score": cell["score"],
            "count": cell["count"],
            "status": "Measured",
            "area": cell_size * cell_size,
        })

    # Draw estimated cells with lower opacity so they are distinguishable.
    for (ix, iy), cell in estimated.items():
        south, west = unproject_coordinates(
            ix * cell_size, iy * cell_size
        )
        north, east = unproject_coordinates(
            (ix + 1) * cell_size, (iy + 1) * cell_size
        )

        color = score_color(cell["score"])

        rectangle = folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.25,
            dash_array="4",
            tooltip=f"Estimated cell | Score: {cell['score']:.1f}/100",
        ).add_to(layer)

        carbon_cells.append({
            "layer": rectangle.get_name(),
            "score": cell["score"],
            "count": 0,
            "status": "Estimated",
            "area": cell_size * cell_size,
        })

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

    # =========================
    # CARBON ESTIMATOR INTERFACE
    # =========================

    coefficient_inputs = ""

    for tree_key, tree_name in TREE_TYPES.items():
        coefficient_inputs += f"""
            <div class="coefficient-group" id="coeff-group-{tree_key}"
                style="display:none;">

        """

        for density_name in ("sparse", "medium", "dense"):
            value = CARBON_COEFFICIENTS[tree_key][density_name]

            coefficient_inputs += f"""
                <label class="coefficient-row">
                    <span>{density_name.title()}</span>
                    <input
                        type="number"
                        min="0"
                        step="0.01"
                        class="carbon-coefficient"
                        id="coeff-{tree_key}-{density_name}"
                        value="{value}"
                    >
                </label>
            """

        # Close each tree group's div inside the loop.
        coefficient_inputs += "</div>"

    carbon_panel = f"""
    <div id="carbon-panel">
        <h3>Ước tính lượng tín chỉ carbon</h3>

        <label for="tree-type">Loại cây trồng</label>
        <select id="tree-type">
            <option value="acacia">Cây keo</option>
            <option value="tea">Cây chè</option>
            <option value="fokienia">Cây pơ mu</option>
            <option value="birch">Cây bạch dương</option>
        </select>

        <details open>
            <summary>Hệ số tín chỉ carbon</summary>

            <p class="carbon-note">
                Editable values in tCO2e/ha/year.
            </p>

            {coefficient_inputs}
        </details>

        <div class="carbon-results">
            <h4>Mức tín chỉ carbon trên diện tích đo thực:</h4>
            <div class="carbon-value" id="measured-total">
                0.000 tCO2e/year
            </div>

            <h4>Mức tín chỉ carbon trên diện tích được ước tính:</h4>
            <div class="carbon-value" id="estimated-total">
                0.000 tCO2e/year
            </div>

            <p id="carbon-area"></p>
            <p id="carbon-cell-count"></p>
        </div>

        <p class="carbon-note">
            Đây chỉ là kết quả ước tính đánh giá ban đầu, không phải số liệu chính thức.
        </p>
    </div>

    <style>
        #carbon-panel {{
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 9999;
            width: 290px;
            max-height: 85vh;
            overflow-y: auto;
            box-sizing: border-box;
            background: white;
            padding: 14px;
            border: 1px solid #888;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
            font-family: Arial, sans-serif;
            font-size: 13px;
        }}

        #carbon-panel h3 {{
            margin: 0 0 12px 0;
        }}

        #carbon-panel h4 {{
            margin: 12px 0 5px 0;
        }}

        #carbon-panel select,
        #carbon-panel input {{
            box-sizing: border-box;
            width: 100%;
            padding: 6px;
            margin: 5px 0 10px 0;
        }}

        .coefficient-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }}

        .coefficient-row span {{
            flex: 1;
        }}

        .coefficient-row input {{
            width: 95px !important;
        }}

        .carbon-results {{
            border-top: 1px solid #ccc;
            margin-top: 10px;
            padding-top: 5px;
        }}

        .carbon-value {{
            font-size: 18px;
            font-weight: bold;
            color: #217a45;
        }}

        .carbon-note {{
            color: #666;
            font-size: 11px;
            line-height: 1.4;
        }}
    </style>
    """

    m.get_root().html.add_child(
        folium.Element(carbon_panel)
    )

    # =========================
    # CARBON CALCULATION JAVASCRIPT
    # =========================

    carbon_data_json = json.dumps(carbon_cells)

    carbon_script = r"""
    const carbonCells = __CARBON_CELLS__;

    const treeSelector = document.getElementById("tree-type");

    const densityLevels = ["sparse", "medium", "dense"];

    // Restore previously edited coefficients from this browser.
    const storageKey = "forest_carbon_coefficients";

    try {
        const savedCoefficients = JSON.parse(
            localStorage.getItem(storageKey) || "{}"
        );

        document.querySelectorAll(".carbon-coefficient").forEach(input => {
            if (savedCoefficients[input.id] !== undefined) {
                input.value = savedCoefficients[input.id];
            }
        });
    } catch (error) {
        console.warn("Could not restore carbon coefficients:", error);
    }

    function getCoefficients(tree) {
        return {
            sparse: Number(
                document.getElementById(`coeff-${tree}-sparse`).value
            ) || 0,

            medium: Number(
                document.getElementById(`coeff-${tree}-medium`).value
            ) || 0,

            dense: Number(
                document.getElementById(`coeff-${tree}-dense`).value
            ) || 0
        };
    }

    function interpolateCoefficient(score, coefficients) {
        if (score <= 50) {
            const ratio = score / 50;

            return coefficients.sparse +
                (coefficients.medium - coefficients.sparse) * ratio;
        }

        const ratio = (score - 50) / 50;

        return coefficients.medium +
            (coefficients.dense - coefficients.medium) * ratio;
    }

    function updateCoefficientEditor() {
        const selectedTree = treeSelector.value;

        document.querySelectorAll(".coefficient-group").forEach(group => {
            group.style.display = "none";
        });

        document.getElementById(
            `coeff-group-${selectedTree}`
        ).style.display = "block";
    }

    function calculateCarbon() {
        const selectedTree = treeSelector.value;
        const coefficients = getCoefficients(selectedTree);

        let measuredTotal = 0;
        let estimatedTotal = 0;

        let measuredArea = 0;
        let estimatedArea = 0;

        let measuredCount = 0;
        let estimatedCount = 0;

        carbonCells.forEach(cell => {
            const coefficient = interpolateCoefficient(
                cell.score,
                coefficients
            );

            // Convert square metres to hectares.
            const areaInHectares = cell.area / 10000;

            // Annual estimated CO2 removal for this cell.
            const carbon = coefficient * areaInHectares;

            if (cell.status === "Measured") {
                measuredTotal += carbon;
                measuredArea += cell.area;
                measuredCount++;
            } else {
                estimatedTotal += carbon;
                estimatedArea += cell.area;
                estimatedCount++;
            }

            // Update the individual grid-cell popup.
            const layer = window[cell.layer];

            if (layer) {
                const densityLabel =
                    cell.score < 25 ? "Sparse" :
                    cell.score < 75 ? "Medium" : "Dense";

                const popup = `
                    <b>${cell.status} cell</b><br>
                    Tree type: ${selectedTree}<br>
                    Density: ${densityLabel}<br>
                    Density score: ${cell.score.toFixed(1)}/100<br>
                    Area: ${cell.area.toFixed(1)} m²<br>
                    Coefficient: ${coefficient.toFixed(3)}
                    tCO2e/ha/year<br>
                    <b>Estimated annual carbon removal:
                    ${carbon.toFixed(5)} tCO2e/year</b>
                `;

                layer.bindPopup(popup);
            }
        });

        document.getElementById("measured-total").textContent =
            measuredTotal.toFixed(3) + " tCO2e/year";

        document.getElementById("estimated-total").textContent =
            estimatedTotal.toFixed(3) + " tCO2e/year";

        document.getElementById("carbon-area").textContent =
            "Diện tích được đo: " + measuredArea.toFixed(1) +
            " m² | Diện tích được ước tính: " + estimatedArea.toFixed(1) + " m²";

        document.getElementById("carbon-cell-count").textContent =
            "Ô được đo: " + measuredCount +
            " | Ô được ước tính: " + estimatedCount;
    }

    // Recalculate when the tree type changes.
    treeSelector.addEventListener("change", () => {
        updateCoefficientEditor();
        calculateCarbon();
    });

    // Recalculate and save whenever a coefficient changes.
    document.querySelectorAll(".carbon-coefficient").forEach(input => {
        input.addEventListener("input", () => {
            const values = {};

            document.querySelectorAll(".carbon-coefficient").forEach(item => {
                values[item.id] = item.value;
            });

            localStorage.setItem(storageKey, JSON.stringify(values));

            calculateCarbon();
        });
    });

    // Initial calculation.
    updateCoefficientEditor();
    calculateCarbon();
    """.replace(
        "__CARBON_CELLS__",
        carbon_data_json
    )

    m.get_root().script.add_child(folium.Element(carbon_script))

    m.get_root().html.add_child(folium.Element(legend))

    return HTMLResponse(content=m.get_root().render())
