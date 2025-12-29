import ee
from datetime import datetime

# =====================================================
# COMMON HELPERS
# =====================================================

def _date_window(days):
    end = ee.Date(datetime.utcnow())
    start = end.advance(-days, "day")
    return start, end


def _reduce_mean(img, lon, lat, scale):
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry.Point([lon, lat]),
        scale=scale,
        maxPixels=1e9
    )


def _safe_get(region, key):
    return ee.Algorithms.If(
        region.contains(key),
        region.get(key),
        None
    )


def _to_float(val):
    """
    🔒 HARD SAFETY
    EE / None / garbage → Python float
    """
    try:
        if val is None:
            return 0.0
        if hasattr(val, "getInfo"):
            val = val.getInfo()
        return float(val) if val is not None else 0.0
    except:
        return 0.0


def _choose(primary, fallback):
    """
    🎯 GOLDEN RULE
    Satellite > Model
    """
    if primary and primary > 0:
        return primary
    if fallback and fallback > 0:
        return fallback
    return 0.0


# =====================================================
# FALLBACK SOURCES (CAMS / MERRA)
# =====================================================

def _fetch_cams(lat, lon, band, days=7):
    start, end = _date_window(days)
    img = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .filterDate(start, end)
        .select(band)
        .mean()
    )
    stats = _reduce_mean(img, lon, lat, 25000)
    return _safe_get(stats, band)


def _fetch_pm25_merra(lat, lon, days=10):
    start, end = _date_window(days)
    img = (
        ee.ImageCollection("NASA/GEOS-CF/v1/rpl")
        .filterDate(start, end)
        .select("PM25")
        .mean()
    )
    stats = _reduce_mean(img, lon, lat, 25000)
    return _safe_get(stats, "PM25")


# =====================================================
# FINAL GAS FETCHER (PRIMARY + FALLBACK)
# =====================================================

def fetch_all_gases(lat, lon, date_obj=None):
    """
    ✅ Sentinel + MODIS primary
    ✅ CAMS / MERRA fallback
    ✅ returns PURE PYTHON FLOATS
    """

    if not date_obj:
        date_obj = datetime.utcnow()

    date = ee.Date(date_obj)

    # ---------- NO2 (Sentinel + CAMS) ----------
    no2_sat = _safe_get(
        _reduce_mean(
            ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
            .select("tropospheric_NO2_column_number_density")
            .filterDate(date.advance(-3, "day"), date)
            .mean(),
            lon, lat, 7000
        ),
        "tropospheric_NO2_column_number_density"
    )
    no2_cams = _fetch_cams(lat, lon, "no2")

    # ---------- SO2 (Sentinel + CAMS) ----------
    so2_sat = _safe_get(
        _reduce_mean(
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
            .select("SO2_column_number_density")
            .filterDate(date.advance(-5, "day"), date)
            .mean(),
            lon, lat, 10000
        ),
        "SO2_column_number_density"
    )
    so2_cams = _fetch_cams(lat, lon, "so2")

    # ---------- CO (Sentinel + CAMS) ----------
    co_sat = _safe_get(
        _reduce_mean(
            ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
            .select("CO_column_number_density")
            .filterDate(date.advance(-5, "day"), date)
            .mean(),
            lon, lat, 7000
        ),
        "CO_column_number_density"
    )
    co_cams = _fetch_cams(lat, lon, "co")

    # ---------- PM2.5 (MODIS + MERRA) ----------
    aod = _safe_get(
        _reduce_mean(
            ee.ImageCollection("MODIS/061/MCD19A2")
            .select("Optical_Depth_055")
            .filterDate(date.advance(-7, "day"), date)
            .mean(),
            lon, lat, 1000
        ),
        "Optical_Depth_055"
    )
    pm25_sat = ee.Algorithms.If(aod, ee.Number(aod).multiply(50), None)
    pm25_merra = _fetch_pm25_merra(lat, lon)

    # ---------- FINAL (Satellite first) ----------
    return {
        "no2":  _choose(_to_float(no2_sat),  _to_float(no2_cams)),
        "so2":  _choose(_to_float(so2_sat),  _to_float(so2_cams)),
        "co":   _choose(_to_float(co_sat),   _to_float(co_cams)),
        "pm25": _choose(_to_float(pm25_sat), _to_float(pm25_merra)),
    }


# =====================================================
# THERMAL (VIIRS – SAFE)
# =====================================================

def fetch_thermal_safe(lat, lon):
    date = ee.Date(datetime.utcnow())

    collection = (
        ee.ImageCollection("NOAA/VIIRS/001/VNP21A1D")
        .filterDate(date.advance(-30, "day"), date)
    )

    day = _safe_get(
        _reduce_mean(collection.select("LST_Day_1km").mean(), lon, lat, 1000),
        "LST_Day_1km"
    )
    night = _safe_get(
        _reduce_mean(collection.select("LST_Night_1km").mean(), lon, lat, 1000),
        "LST_Night_1km"
    )

    return {
        "day":   _to_float(ee.Number(day).multiply(0.02) if day else None),
        "night": _to_float(ee.Number(night).multiply(0.02) if night else None)
    }
