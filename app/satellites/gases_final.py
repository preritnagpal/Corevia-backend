import ee
from datetime import datetime, timedelta

# =====================================================
# HELPERS
# =====================================================

def _reduce_mean(img, lon, lat, scale):
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry.Point([lon, lat]),
        scale=scale,
        maxPixels=1e9
    )

def _safe_get(region, key):
    return ee.Algorithms.If(region.contains(key), region.get(key), None)

def _to_float(val):
    try:
        if val is None:
            return 0.0
        if hasattr(val, "getInfo"):
            val = val.getInfo()
        return float(val) if val is not None else 0.0
    except:
        return 0.0

def _pick(daily, avg):
    return daily if daily and daily > 0 else avg


# =====================================================
# AVERAGE FALLBACKS (ONLY IF DAILY FAILS)
# =====================================================

def _avg_cams(lat, lon, band, date, days=5):
    start = date.advance(-days, "day")
    img = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .filterDate(start, date)
        .select(band)
        .mean()
    )
    stats = _reduce_mean(img, lon, lat, 25000)
    return _safe_get(stats, band)


def _avg_pm25_merra(lat, lon, date, days=7):
    start = date.advance(-days, "day")
    img = (
        ee.ImageCollection("NASA/GEOS-CF/v1/rpl")
        .filterDate(start, date)
        .select("PM25")
        .mean()
    )
    stats = _reduce_mean(img, lon, lat, 25000)
    return _safe_get(stats, "PM25")


# =====================================================
# MAIN — DAILY FIRST, AVERAGE IF ZERO
# =====================================================

def fetch_all_gases(lat, lon, date_obj=None):
    """
    ✔ Daily satellite data ONLY
    ✔ If daily = 0 → fallback average
    """

    if not date_obj:
        date_obj = datetime.utcnow()

    date = ee.Date(date_obj)
    next_day = date.advance(1, "day")

    # ---------- NO2 (DAILY) ----------
    no2_daily = _safe_get(
        _reduce_mean(
            ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
            .select("tropospheric_NO2_column_number_density")
            .filterDate(date, next_day)
            .first(),
            lon, lat, 7000
        ),
        "tropospheric_NO2_column_number_density"
    )
    no2_avg = _avg_cams(lat, lon, "no2", date)

    # ---------- SO2 (DAILY) ----------
    so2_daily = _safe_get(
        _reduce_mean(
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
            .select("SO2_column_number_density")
            .filterDate(date, next_day)
            .first(),
            lon, lat, 10000
        ),
        "SO2_column_number_density"
    )
    so2_avg = _avg_cams(lat, lon, "so2", date)

    # ---------- CO (DAILY) ----------
    co_daily = _safe_get(
        _reduce_mean(
            ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
            .select("CO_column_number_density")
            .filterDate(date, next_day)
            .first(),
            lon, lat, 7000
        ),
        "CO_column_number_density"
    )
    co_avg = _avg_cams(lat, lon, "co", date)

    # ---------- PM2.5 (DAILY via MODIS AOD) ----------
    aod_daily = _safe_get(
        _reduce_mean(
            ee.ImageCollection("MODIS/061/MCD19A2")
            .select("Optical_Depth_055")
            .filterDate(date, next_day)
            .first(),
            lon, lat, 1000
        ),
        "Optical_Depth_055"
    )
    pm25_daily = ee.Algorithms.If(aod_daily, ee.Number(aod_daily).multiply(50), None)
    pm25_avg = _avg_pm25_merra(lat, lon, date)

    return {
        "no2":  _pick(_to_float(no2_daily),  _to_float(no2_avg)),
        "so2":  _pick(_to_float(so2_daily),  _to_float(so2_avg)),
        "co":   _pick(_to_float(co_daily),   _to_float(co_avg)),
        "pm25": _pick(_to_float(pm25_daily), _to_float(pm25_avg)),
    }
def fetch_thermal_safe(lat, lon, date_obj=None):
    if not date_obj:
        date_obj = datetime.utcnow()

    date = ee.Date(date_obj)
    next_day = date.advance(1, "day")

    collection = (
        ee.ImageCollection("NOAA/VIIRS/001/VNP21A1D")
        .filterDate(date, next_day)
    )

    day = _safe_get(
        _reduce_mean(collection.select("LST_Day_1km").first(), lon, lat, 1000),
        "LST_Day_1km"
    )
    night = _safe_get(
        _reduce_mean(collection.select("LST_Night_1km").first(), lon, lat, 1000),
        "LST_Night_1km"
    )

    return {
        "day":   _to_float(ee.Number(day).multiply(0.02) if day else None),
        "night": _to_float(ee.Number(night).multiply(0.02) if night else None)
    }
