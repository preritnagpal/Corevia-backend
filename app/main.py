from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
from dotenv import load_dotenv
import ee
import os
from datetime import datetime, timedelta
from bson.errors import InvalidId
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google.oauth2 import service_account


# --------------------------------------------------
# INIT APP FIRST
# --------------------------------------------------
app = FastAPI(title="Factory Environmental Intelligence")

# --------------------------------------------------
# CORS (MUST BE AFTER app = FastAPI)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # 👈 OPTIONS allowed
    allow_headers=["*"],
)

# --------------------------------------------------
# ENV + EARTH ENGINE
# --------------------------------------------------
load_dotenv()


def init_earth_engine():
    try:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("GEE_PROJECT_ID")

        if not creds_path or not project_id:
            print("❌ EE env vars missing")
            return

        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )

        ee.Initialize(credentials, project=project_id)
        print("✅ Earth Engine initialized with service account")

    except Exception as e:
        print("❌ Earth Engine init failed:", e)


@app.on_event("startup")
def on_startup():
    init_earth_engine()

# --------------------------------------------------
# DB
# --------------------------------------------------
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["hackathonDB"]

users = db["users"]
daily_metrics = db["factory_daily_metrics"]
alerts = db["factory_alerts"]
factory_impact_col = db["factory_impact_daily"]

# --------------------------------------------------
# Satellite fetchers (rest code below)
# --------------------------------------------------
from app.satellites.gases_final import (
    fetch_all_gases,
    fetch_thermal_safe
)




# --------------------------------------------------
# MODELS
# --------------------------------------------------
class IngestRequest(BaseModel):
    factoryId: str

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
SAFE_LIMITS = {
    "no2": 0.00004,
    "so2": 0.00002,
    "co": 0.05,
    "pm25": 60
}
ALERT_SOURCES = {"ai", "government", "manual"}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def normalize(value, limit):
    if value is None or value <= 0:
        return None
    return min(value / limit, 2)

def compute_gas_score(gases):
    score = 0

    val = normalize(gases.get("no2"), SAFE_LIMITS["no2"])
    if val is not None:
        score += 0.35 * val

    val = normalize(gases.get("so2"), SAFE_LIMITS["so2"])
    if val is not None:
        score += 0.25 * val

    val = normalize(gases.get("co"), SAFE_LIMITS["co"])
    if val is not None:
        score += 0.20 * val

    val = normalize(gases.get("pm25"), SAFE_LIMITS["pm25"])
    if val is not None:
        score += 0.20 * val

    return score


def compute_eri(gases, thermal):
    gas_score = compute_gas_score(gases) or 0

    thermal_penalty = 10 if (
        thermal and
        thermal.get("night") and
        (not thermal.get("day") or thermal.get("night") > thermal.get("day"))
    ) else 0

    eri = gas_score * 60 + thermal_penalty
    eri = max(eri, 5)
    eri = min(eri, 100)

    if eri <= 25:
        category = "Low"
    elif eri <= 50:
        category = "Moderate"
    elif eri <= 75:
        category = "High"
    else:
        category = "Critical"

    return round(eri, 2), category


def compute_trend(factory_id, field):
    records = list(
        daily_metrics.find(
            {
                "factoryId": factory_id,
                field: {"$ne": None}
            },
            {field: 1, "date": 1, "_id": 0}
        ).sort("date", -1).limit(6)
    )

    values = []
    for r in records:
        val = r
        for k in field.split("."):
            val = val.get(k) if isinstance(val, dict) else None
        if isinstance(val, (int, float)):
            values.append(val)

    # Minimum data check
    if len(values) < 6:
        return {
            "status": "insufficient_data",
            "change_percent": None
        }

    recent = sum(values[:3]) / 3
    previous = sum(values[3:6]) / 3

    if previous == 0:
        return {
            "status": "insufficient_data",
            "change_percent": None
        }

    change = round(((recent - previous) / previous) * 100, 2)

    if change > 10:
        status = "rising"
    elif change < -10:
        status = "improving"
    else:
        status = "stable"

    # 🔥 OPTIONAL AI ALERT (SAFE)
    if status == "rising" and abs(change) > 15:
        create_alert(
            factory_id=factory_id,
            alert_type=f"{field.upper()}_SPIKE",
            severity="medium" if abs(change) < 30 else "high",
            message=f"{field.upper()} spike detected compared to recent average",
            value=change,
            baseline="7d_avg",
            source="ai"
        )

    return {
        "status": status,
        "change_percent": change
    }



def forecast_eri(current_eri, trend):
    if trend["status"] == "insufficient_data":
        return {
            "forecast": None,
            "confidence": "low",
            "message": "Not enough historical data"
        }

    delta = trend["change_percent"] / 7
    predicted = max(0, min(current_eri + delta * 3, 100))

    return {
        "forecast": round(predicted, 2),
        "confidence": "medium" if abs(delta) < 5 else "high",
        "window": "72h"
    }

def smooth_series(values, window=3):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i+1]
        smoothed.append(round(sum(chunk) / len(chunk), 2))
    return smoothed

def confidence_score(trend):
    if trend["status"] == "insufficient_data":
        return "low"

    change = abs(trend["change_percent"])

    if change < 5:
        return "medium"
    elif change < 15:
        return "high"
    else:
        return "very_high"

def trend_explanation(metric, trend):
    if trend["status"] == "insufficient_data":
        return f"Not enough historical data to determine {metric} trend."

    if trend["status"] == "rising":
        return f"{metric.upper()} levels have increased compared to the previous week."

    if trend["status"] == "improving":
        return f"{metric.upper()} levels have reduced compared to the previous week."

    return f"{metric.upper()} levels remain stable over the last two weeks."

def clean_value(val):
    if val is None:
        return None
    if isinstance(val, (int, float)) and val <= 0:
        return None
    return val


def compute_region_avg(factory_id, field, days=7):
    records = list(
        daily_metrics.find(
            {"factoryId": factory_id},
            {field: 1, "_id": 0}
        ).sort("date", -1).limit(days)
    )

    values = []
    for r in records:
        v = r
        for k in field.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            values.append(v)

    if not values:
        return None

    return sum(values) / len(values)


def classify_impact(percent):
    if percent is None:
        return "unknown"
    if abs(percent) < 10:
        return "negligible"
    if percent < -10:
        return "below_average"
    if percent < 20:
        return "moderate"
    return "high"

def compute_coverage(factory_id, days=7):
    total = days

    valid = daily_metrics.count_documents({
        "factoryId": factory_id,
        "eri": {"$ne": None}
    })

    if total == 0:
        return {
            "coverage_percent": 0,
            "confidence": "none"
        }

    coverage = (valid / total) * 100

    if coverage >= 80:
        confidence = "high"
    elif coverage >= 50:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "coverage_percent": round(coverage, 1),
        "confidence": confidence
    }

def create_alert(
    factory_id,
    alert_type,
    severity,
    message,
    value=None,
    baseline=None,
    source="ai",
    alert_date=None        # 👈 ADD THIS
):
    date_str = alert_date or datetime.utcnow().strftime("%Y-%m-%d")

    exists = alerts.find_one({
        "factoryId": factory_id,
        "type": alert_type,
        "date": date_str
    })
    if exists:
        return

    alerts.insert_one({
        "factoryId": factory_id,
        "date": date_str,
        "type": alert_type,
        "severity": severity,
        "message": message,
        "value": value,
        "baseline": baseline,
        "source": source,
        "read": False,
        "dismissedInDropdown": False,
        "deleted": False,
        "createdAt": datetime.utcnow()
    })



def alert_exists(factory_id, alert_type, date):
    return alerts.find_one({
        "factoryId": factory_id,
        "type": alert_type,
        "date": date
    }) is not None

def serialize_alert(alert):
    return {
        "_id": str(alert["_id"]),
        "factoryId": str(alert["factoryId"]),
        "date": alert.get("date"),
        "type": alert.get("type"),
        "severity": alert.get("severity"),
        "message": alert.get("message"),
        "value": alert.get("value"),
        "baseline": alert.get("baseline"),
        "read": alert.get("read", False),
        "createdAt": alert.get("createdAt"),
    }

def generate_daily_impact(factory_id, date_str):
    # 🔒 DUPLICATE BLOCK
    if factory_impact_col.find_one({
        "factoryId": factory_id,
        "date": date_str
    }):
        return

    record = daily_metrics.find_one({
        "factoryId": factory_id,
        "date": date_str
    })

    if not record:
        return

    gases = record.get("gases", {})
    impact = {}

    for gas in ["no2", "so2", "co", "pm25"]:
        factory_val = clean_value(gases.get(gas))
        region_avg = clean_value(
            compute_region_avg(factory_id, f"gases.{gas}")
        )

        if factory_val is None or region_avg is None or region_avg == 0:
            impact_percent = None
        else:
            impact_percent = round(
                ((factory_val - region_avg) / region_avg) * 100,
                2
            )

        impact[gas] = {
            "factory": factory_val,
            "region_avg": region_avg,
            "impact_percent": impact_percent,
            "category": classify_impact(impact_percent)
        }

    factory_impact_col.insert_one({
        "factoryId": factory_id,
        "date": date_str,

        "impact": impact,

        "emissionEstimate": {
            "dailyKgEstimate": record.get("emissionEstimate", {}).get("dailyKgEstimate"),      # 🔮 future model
            "method": "satellite+model"
        },

        "generatedAt": datetime.utcnow()
    })


def ingest_for_date(factory_id, lat, lon, date_obj, factory_type="generic"):
    date_str = date_obj.strftime("%Y-%m-%d")

    # 🔒 HARD DUPLICATE BLOCK
    if daily_metrics.find_one({
        "factoryId": factory_id,
        "date": date_str
    }):
        return "skipped"

    # ================= GASES (ALREADY SAFE FLOATS) =================
    gases = fetch_all_gases(lat, lon, date_obj)   # MUST return floats only

    # ================= THERMAL (FETCH ONCE) =================
    try:
        thermal_raw = fetch_thermal_safe(lat, lon, date_obj)

        # normalize → python dict
        if hasattr(thermal_raw, "getInfo"):
            thermal_raw = thermal_raw.getInfo()

        thermal = {
            "day": thermal_raw.get("day", 0),
            "night": thermal_raw.get("night", 0)
        }
    except Exception:
        thermal = {"day": 0, "night": 0}

    # ================= EMISSION ESTIMATE =================
    daily_kg_estimate = estimate_daily_emission_kg(gases, factory_type)

    # ================= ERI =================
    eri, category = compute_eri(gases, thermal)

    # ================= FINAL INSERT =================
    daily_metrics.insert_one({
        "factoryId": factory_id,
        "date": date_str,

        "gases": gases,
        "thermal": thermal,

        "eri": eri,
        "category": category,

        "emissionEstimate": {
            "dailyKgEstimate": daily_kg_estimate,
            "method": "satellite+model"
        },

        "satelliteMeta": {
            "satellite": "Sentinel-5P",
            "passType": "night",
            "sensingWindow": "last_3_days",
            "processedAt": datetime.utcnow()
        },

        "createdAt": datetime.utcnow()
    })

    # 🔥 AUTO IMPACT
    generate_daily_impact(factory_id, date_str)

    # 🔔 ALERT (ONCE PER DAY)
    if eri >= 75:
        create_alert(
            factory_id=factory_id,
            alert_type="ERI_CRITICAL",
            severity="critical",
            message=f"Environmental Risk Index critical ({eri})",
            value=eri,
            baseline=75,
            source="system",
            alert_date=date_str
        )

    return "inserted"



def estimate_daily_emission_kg(gases: dict, industry: str):
    """
    Estimate daily emission load (kg/day) using satellite gas columns
    Industry adjusted multipliers (production-ready heuristic model)
    """

    # ---------------- BASE CONVERSION FACTORS ----------------
    # satellite column → kg/day (approx regional dispersion model)
    FACTORS = {
        "no2": 52000,     # kg per mol/m² column (scaled)
        "so2": 64000,
        "co": 28000,
        "pm25": 1.2       # µg/m³ → kg/day estimate
    }

    # ---------------- INDUSTRY MULTIPLIERS ----------------
    INDUSTRY_MULTIPLIER = {
        "cement": 1.6,
        "steel": 1.9,
        "power": 2.2,
        "chemical": 2.0,
        "fertilizer": 1.8,
        "refinery": 2.4,
        "textile": 1.2,
        "generic": 1.0
    }

    industry = industry.lower()
    multiplier = INDUSTRY_MULTIPLIER.get(industry, 1.0)

    total_kg = 0.0

    # ---------------- GAS CONTRIBUTION ----------------
    for gas, raw in gases.items():
        if raw is None or raw <= 0:
            continue

        factor = FACTORS.get(gas)
        if not factor:
            continue

        total_kg += raw * factor

    # ---------------- FINAL INDUSTRY WEIGHT ----------------
    total_kg *= multiplier

    # ---------------- SAFETY BOUNDS ----------------
    total_kg = max(total_kg, 0)
    total_kg = min(total_kg, 500000)   # upper cap for sanity

    return round(total_kg, 3)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "backend running"}

@app.post("/satellite/daily-ingest")
def daily_ingest(payload: IngestRequest):
    try:
        factory_id = ObjectId(payload.factoryId)
    except:
        raise HTTPException(400, "Invalid factoryId")


    user = users.find_one({"_id": factory_id})
    if not user:
        raise HTTPException(404, "Factory not found")

    try:
        lat = float(user["data"]["latitude"])
        lon = float(user["data"]["longitude"])
    except:
        raise HTTPException(400, "Invalid lat/lon")

    today = datetime.utcnow()

    factory_type = user["data"].get("factoryType", "generic")

    result = ingest_for_date(
    factory_id,
    lat,
    lon,
    today,
    factory_type
    )


    if result == "skipped":
        existing = daily_metrics.find_one(
            {"factoryId": factory_id, "date": today.strftime("%Y-%m-%d")}
        )
        return {
            "success": True,
            "skipped": True,
            "date": existing["date"],
            "eri": existing["eri"],
            "category": existing["category"]
        }

    return {
        "success": True,
        "skipped": False,
        "date": today.strftime("%Y-%m-%d")
    }


@app.get("/factory/history")
def factory_history(factoryId: str, range: str = "7d"):
    factory_id = ObjectId(factoryId)

    days_map = {"1d": 1, "7d": 7, "30d": 30, "1y": 365}
    since = (datetime.utcnow() - timedelta(days=days_map[range])).strftime("%Y-%m-%d")

    records = list(
        daily_metrics.find(
            {"factoryId": factory_id, "date": {"$gte": since}},
            {"_id": 0}
        ).sort("date", 1)
    )

    eri_values = [r["eri"] for r in records if isinstance(r.get("eri"), (int, float))]
    eri_smooth = smooth_series(eri_values)

    return {
        "factoryId": factoryId,
        "range": range,
        "count": len(records),
        "data": records,
        "eri_smooth": eri_smooth
    }

@app.post("/satellite/backfill")
def backfill(payload: dict):
    factory_id = ObjectId(payload["factoryId"])
    days = payload.get("days", 7)

    user = users.find_one({"_id": factory_id})
    if not user:
        raise HTTPException(404, "Factory not found")

    lat = float(user["data"]["latitude"])
    lon = float(user["data"]["longitude"])

    inserted, skipped = 0, 0

    for i in range(days):
        date_obj = datetime.utcnow() - timedelta(days=i)
        result = ingest_for_date(factory_id, lat, lon, date_obj)

        if result == "inserted":
            inserted += 1
        else:
            skipped += 1

    return {
        "success": True,
        "inserted": inserted,
        "skipped": skipped
    }



@app.post("/satellite/trend")
def trend(payload: dict):
    factory_id = ObjectId(payload["factoryId"])

    eri_trend = compute_trend(factory_id, "eri")
    coverage = compute_coverage(factory_id, days=7)

    return {
        "success": True,

        "eri": {
            **eri_trend,
            "confidence": confidence_score(eri_trend),
            "explanation": trend_explanation("eri", eri_trend)
        },

        "no2": compute_trend(factory_id, "gases.no2"),
        "so2": compute_trend(factory_id, "gases.so2"),
        "co": compute_trend(factory_id, "gases.co"),

        # ✅ META BLOCK
        "meta": {
            "data_quality": "satellite_qa_filtered",
            "coverage_7d": coverage
        }
    }


@app.post("/satellite/impact")
def factory_impact(payload: dict):
    factory_id = ObjectId(payload["factoryId"])

    latest = daily_metrics.find_one(
        {"factoryId": factory_id},
        sort=[("date", -1)]
    )

    if not latest:
        raise HTTPException(status_code=404, detail="No data available")

    gases = latest.get("gases", {})
    impact_result = {}

    for gas in ["no2", "so2", "co", "pm25"]:
        raw_factory_val = gases.get(gas)
        raw_region_avg = compute_region_avg(factory_id, f"gases.{gas}")

        factory_val = clean_value(raw_factory_val)
        region_avg = clean_value(raw_region_avg)

        if factory_val is None or region_avg is None or region_avg == 0:
            impact_percent = None
        else:
            impact_percent = ((factory_val - region_avg) / region_avg) * 100
            impact_percent = round(impact_percent, 2)

        impact_result[gas] = {
            "factory": factory_val,
            "region_avg": region_avg,
            "impact_percent": impact_percent,
            "category": classify_impact(impact_percent)
        }

    return {
        "success": True,
        "date": latest["date"],
        "impact": impact_result
    }












@app.post("/satellite/forecast")
def forecast(payload: dict):
    factory_id = ObjectId(payload["factoryId"])

    latest = daily_metrics.find_one(
        {"factoryId": factory_id},
        sort=[("date", -1)]
    )

    if not latest:
        raise HTTPException(status_code=404, detail="No data")

    trend = compute_trend(factory_id, "eri")
    forecast_data = forecast_eri(latest["eri"], trend)

    coverage = compute_coverage(factory_id, days=7)

    return {
        "success": True,
        "current_eri": latest["eri"],
        "trend": trend,
        "forecast": forecast_data,
        "coverage": coverage
    }


@app.post("/government/alert")
def government_alert(payload: dict):
    try:
        factory_id = ObjectId(payload["factoryId"])
    except:
        raise HTTPException(status_code=400, detail="Invalid factoryId")

    message = payload.get("message")
    severity = payload.get("severity", "high")

    if not message:
        raise HTTPException(status_code=400, detail="Message required")

    create_alert(
        factory_id=factory_id,
        alert_type="GOVT_NOTICE",
        severity=severity,
        message=message,
        source="government"
    )

    return {
        "success": True,
        "message": "Government alert issued"
    }


@app.get("/alerts/unread")
def unread_alerts(factoryId: str):
    try:
        fid = ObjectId(factoryId)
    except:
        raise HTTPException(status_code=400, detail="Invalid factoryId")

    alerts = list(
        db.factory_alerts.find(
            {
                "factoryId": fid,
                "read": False,
                "deleted": False
            }
        ).sort("createdAt", -1)
    )

    for a in alerts:
        a["_id"] = str(a["_id"])
        a["factoryId"] = str(a["factoryId"])

    return {
        "count": len(alerts),
        "alerts": alerts
    }


@app.post("/alerts/mark-read")
async def mark_alert_read(data: dict):
    if "alertId" not in data:
        raise HTTPException(status_code=400, detail="alertId required")

    try:
        alert_id = ObjectId(data["alertId"])
    except:
        raise HTTPException(status_code=400, detail="Invalid alertId")

    result = db.factory_alerts.update_one(
        { "_id": alert_id },
        { "$set": { "read": True } }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        "success": True,
        "alertId": str(alert_id)
    }
@app.get("/alerts/unread-count")
def unread_count(factoryId: str):
    try:
        fid = ObjectId(factoryId)
    except:
        raise HTTPException(status_code=400, detail="Invalid factoryId")

    count = db.factory_alerts.count_documents({
        "factoryId": fid,
        "read": False,
        "deleted": False
    })

    return {
        "count": count
    }

@app.post("/alerts/dismiss-dropdown")
async def dismiss_from_dropdown(data: dict):
    alert_id = ObjectId(data["alertId"])

    db.factory_alerts.update_one(
        {"_id": alert_id},
        {"$set": {"dismissedInDropdown": True}}
    )

    return {"success": True}

@app.get("/alerts/dropdown")
def dropdown_alerts(factoryId: str):
    fid = ObjectId(factoryId)

    alerts = list(
        db.factory_alerts.find({
            "factoryId": fid,
            "read": False,
            "dismissedInDropdown": { "$ne": True },
            "deleted": False
        }).sort("createdAt", -1)
    )

    for a in alerts:
        a["_id"] = str(a["_id"])
        a["factoryId"] = str(a["factoryId"])

    return {
        "count": len(alerts),
        "alerts": alerts
    }

@app.post("/alerts/delete")
async def delete_alert(data: dict):
    if "alertId" not in data:
        raise HTTPException(status_code=400, detail="alertId required")

    try:
        alert_id = ObjectId(data["alertId"])
    except:
        raise HTTPException(status_code=400, detail="Invalid alertId")

    result = db.factory_alerts.update_one(
        { "_id": alert_id },
        { "$set": { "deleted": True } }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        "success": True,
        "alertId": str(alert_id)
    }

@app.post("/alerts/delete-bulk")
async def delete_bulk_alerts(data: dict):
    if "alertIds" not in data or not isinstance(data["alertIds"], list):
        raise HTTPException(status_code=400, detail="alertIds array required")

    try:
        ids = [ObjectId(i) for i in data["alertIds"]]
    except:
        raise HTTPException(status_code=400, detail="Invalid alertIds")

    db.factory_alerts.update_many(
        { "_id": { "$in": ids } },
        { "$set": { "deleted": True } }
    )

    return {
        "success": True,
        "deletedCount": len(ids)
    }

@app.get("/alerts/all")
def all_alerts(factoryId: str):
    fid = ObjectId(factoryId)

    alerts = list(
        db.factory_alerts.find({
            "factoryId": fid,
            "deleted": False   # 👈 IMPORTANT
        }).sort("createdAt", -1)
    )

    for a in alerts:
        a["_id"] = str(a["_id"])
        a["factoryId"] = str(a["factoryId"])
        a["read"] = bool(a.get("read", False))

    return {
        "alerts": alerts
    }




