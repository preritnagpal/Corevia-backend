import ee
from datetime import datetime

# =====================================================
# HELPERS
# =====================================================

def _reduce_mean(img, lon, lat, scale):
    try:
        return img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry.Point([lon, lat]),
            scale=scale,
            maxPixels=1e9
        )
    except:
        return None


def _safe_get(region, key):
    try:
        if region is None:
            return None
        return ee.Algorithms.If(region.contains(key), region.get(key), None)
    except:
        return None


def _to_float(val):
    try:
        if val is None:
            return 0.0
        if hasattr(val, "getInfo"):
            val = val.getInfo()
        return float(val) if val is not None else 0.0
    except:
        return 0.0


# =====================================================
# CORE FETCH (WITH DATE WINDOW)
# =====================================================

def _first_daily(collection, band, date, lon, lat, scale):

    try:

        start = date.advance(-3, "day")
        end = date.advance(1, "day")

        img = (
            ee.ImageCollection(collection)
            .select(band)
            .filterDate(start, end)
            .sort("system:time_start", False)
            .first()
        )

        region = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry.Point([lon, lat]),
            scale=scale,
            maxPixels=1e9
        )

        val = region.get(band)

        if hasattr(val, "getInfo"):
            val = val.getInfo()

        return float(val) if val is not None else 0.0

    except Exception as e:
        print("Satellite fetch error:", e)
        return 0.0


# =====================================================
# GAS FETCH
# =====================================================

def fetch_all_gases(lat, lon, date_obj=None):

    if not date_obj:
        date_obj = datetime.utcnow()

    date = ee.Date(date_obj)

    # ---------- NO2 ----------
    no2 = _first_daily(
        "COPERNICUS/S5P/NRTI/L3_NO2",
        "tropospheric_NO2_column_number_density",
        date, lon, lat, 7000
    )

    if no2 == 0:
        no2 = _first_daily(
            "ECMWF/CAMS/NRT",
            "no2",
            date, lon, lat, 25000
        )

    # ---------- SO2 ----------
    so2 = _first_daily(
        "COPERNICUS/S5P/OFFL/L3_SO2",
        "SO2_column_number_density",
        date, lon, lat, 10000
    )

    if so2 == 0:
        so2 = _first_daily(
            "ECMWF/CAMS/NRT",
            "so2",
            date, lon, lat, 25000
        )

    # ---------- CO ----------
    co = _first_daily(
        "COPERNICUS/S5P/NRTI/L3_CO",
        "CO_column_number_density",
        date, lon, lat, 7000
    )

    if co == 0:
        co = _first_daily(
            "ECMWF/CAMS/NRT",
            "co",
            date, lon, lat, 25000
        )

    # ---------- PM2.5 ----------
    aod = _first_daily(
        "MODIS/061/MCD19A2",
        "Optical_Depth_055",
        date, lon, lat, 1000
    )

    pm25 = aod * 50 if aod > 0 else 0

    if pm25 == 0:
        pm25 = _first_daily(
            "NASA/GEOS-CF/v1/rpl",
            "PM25",
            date, lon, lat, 25000
        )

    return {
        "no2": no2,
        "so2": so2,
        "co": co,
        "pm25": pm25
    }


# =====================================================
# THERMAL FETCH
# =====================================================

def fetch_thermal_safe(lat, lon, date_obj=None):

    if not date_obj:
        date_obj = datetime.utcnow()

    date = ee.Date(date_obj)

    start = date.advance(-3, "day")
    end = date.advance(1, "day")

    def daily_lst(band):

        img = (
            ee.ImageCollection("NOAA/VIIRS/001/VNP21A1D")
            .select(band)
            .filterDate(start, end)
            .sort("system:time_start", False)
            .first()
        )

        if img is None:
            return 0.0

        region = _reduce_mean(img, lon, lat, 1000)
        val = _safe_get(region, band)

        try:
            return _to_float(ee.Number(val).multiply(0.02))
        except:
            return 0.0

    return {
        "day": daily_lst("LST_Day_1km"),
        "night": daily_lst("LST_Night_1km")
    }

