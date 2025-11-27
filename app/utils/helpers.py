import re
import math
from math import radians, sin, cos, sqrt, atan2
from urllib.parse import urlparse, unquote
from app.schemas.property import AddressData
from app.core.address import STATE_ABBR, UNIT_MARKERS, DIRECTIONS
from typing import Mapping, Callable


def build_fget(source: Mapping) -> Callable:

    def fget(k):
        try:
            return float(source.get(k, float("nan")))
        except Exception:
            return float("nan")

    return fget


def _clean_unit_tokens(tokens: list[str]) -> list[str]:
    """
    Remove trailing unit markers (e.g., Unit, Apt, #3B) from the *street* portion only.
    Keep leading house number and directionals like N/S/E/W intact.
    """
    if len(tokens) < 4:
        return tokens[:]

    # Find the STATE + ZIP pair from the right
    idx_state = None
    for i in range(len(tokens) - 2, 0, -1):
        if tokens[i].upper() in STATE_ABBR and re.fullmatch(r"\d{5}(-\d{4})?", tokens[i + 1]):
            idx_state = i
            break
    if idx_state is None:
        return tokens[:]  # can't split reliably → return as is

    # Assume city is the single token immediately before STATE (works for most Zillow slugs)
    # (If you have lots of multi-token cities, we can extend this later.)
    city_tokens = tokens[idx_state - 1 : idx_state]
    addr_tokens = tokens[: idx_state - 1]  # street part
    state_zip = tokens[idx_state:]  # [STATE, ZIP]

    # Now strip trailing unit markers *at the end of the street tokens only*
    def is_short_unit_like(t: str) -> bool:
        # short tokens like "3", "3B", "#2" used as units; BUT do not treat a leading house number as unit
        return bool(re.fullmatch(r"[#]?\w{1,4}", t)) and any(ch.isdigit() for ch in t)

    i = len(addr_tokens) - 1
    drops = 0
    while i >= 0 and drops < 3:
        t = addr_tokens[i]
        tl = t.lower()
        if tl in UNIT_MARKERS or is_short_unit_like(t):
            # Protect the *first* tokens if they are house number / directional
            if i <= 1 and (t.isdigit() or t.upper() in DIRECTIONS):
                break
            addr_tokens.pop(i)
            drops += 1
            i -= 1
        else:
            break

    # Remove explicit unit markers anywhere inside the street tokens (rare) but
    # preserve leading house number and directional
    cleaned = []
    for j, tok in enumerate(addr_tokens):
        if j <= 1 and (tok.isdigit() or tok.upper() in DIRECTIONS):
            cleaned.append(tok)
            continue
        if tok.lower() in UNIT_MARKERS:
            continue
        cleaned.append(tok)

    return cleaned + city_tokens + state_zip


def extract_address_from_url(url: str) -> AddressData | None:
    """
    Extract address data suitable for ATTOM API from Zillow/Redfin URLs.
    Returns dict with keys: street_tokens, city, state, zipcode, address1, address2, address
    """
    url = unquote(url.strip())
    p = urlparse(url)
    host = p.netloc.lower()
    path = p.path.strip("/")

    def nice(tok: str) -> str:
        """Title-case token, preserving directions and state abbreviations"""
        u = tok.upper()
        if u in DIRECTIONS or u in STATE_ABBR or tok.isdigit():
            return u
        return tok.capitalize()

    def parse_tokens(tokens: list[str]) -> dict | None:
        """Parse address tokens into dict if valid format"""
        tokens = _clean_unit_tokens(tokens)

        # Expect ... City STATE ZIP
        if (
            len(tokens) >= 4
            and tokens[-2].upper() in STATE_ABBR
            and re.fullmatch(r"\d{5}(-\d{4})?", tokens[-1])
        ):
            state = tokens[-2].upper()
            zipcode = tokens[-1]
            city = tokens[-3].replace("-", " ").title()
            street_tokens = tokens[:-3]

            address1 = " ".join(nice(t) for t in street_tokens).strip().replace("  ", " ")
            address2 = f"{city} {state} {zipcode}"

            return AddressData(
                street_tokens=street_tokens,
                city=city,
                state=state,
                zipcode=zipcode,
                address1=address1,
                address2=address2,
                address=f"{address1}, {address2}",
            )
        return None

    # --- Zillow ---
    if "zillow" in host:
        m = re.search(r"homedetails/([^/]+)/", path)
        address_slug = m.group(1) if m else None
        if not address_slug:
            parts = path.split("/")
            if len(parts) >= 3:
                address_slug = parts[-2]

        if address_slug:
            tokens = [t for t in address_slug.split("-") if t]
            return parse_tokens(tokens)

    # --- Redfin ---
    elif "redfin" in host:
        parts = path.split("/")
        if len(parts) >= 4:
            state = parts[0].upper()
            city = parts[1].replace("-", " ").title()
            addr_slug = parts[2]
            tokens = [t for t in addr_slug.split("-") if t]
            # Append city/state for consistent parsing
            tokens = tokens + city.split() + [state]
            return parse_tokens(tokens)

    return None


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def format_property_data(data: dict) -> dict:
    """
    Return a copy of data with appropriate units, rounding, and readable formatting.
    """
    UNITS = {
        # Economics
        "median_income": "USD",
        "median_gross_rent": "USD",
        "median_home_value": "USD",
        "avg_price": "USD",
        "med_price": "USD",
        "price_pred": "USD",
        "rev_pred": "USD",
        # Demographics / counts
        "population": "count",
        "education_bachelors": "count",
        "race_white": "count",
        "race_black": "count",
        "race_asian": "count",
        "race_other": "count",
        "total_housing_units": "count",
        "labor_force": "count",
        "unemployed": "count",
        "occ_pred": "count",
        # Ages / years
        "median_age": "years",
        "median_year_built": "year",
        # Time / distance
        "commute_time_mean": "seconds",
        "dist_to_airport_km": "km",
        "dist_to_train_km": "km",
        "dist_to_park_km": "km",
        "dist_to_university_km": "km",
        "dist_to_bus_km": "km",
        "dist_to_city_center_km": "km",
        # Percentages (fractions 0–1)
        "percent_foreign_born": "%",
        "unemployment_rate": "%",
        "percent_owner_occupied": "%",
        "percent_public_transport": "%",
        "percent_work_from_home": "%",
        "vacancy_rate": "%",
        "poverty_rate": "%",
        "percent_over_65": "%",
        # Other metrics
        "gini_index": "index",
    }

    out = {}
    for key, value in data.items():
        if value is None:
            out[key] = "N/A"
            continue

        unit = UNITS.get(key)

        try:
            if unit == "USD":
                out[key] = f"${float(value):,.0f}"
            elif unit == "count":
                out[key] = f"{int(round(float(value))):,}"
            elif unit == "years":
                out[key] = f"{float(value):.1f} yrs"
            elif unit == "year":
                out[key] = f"{int(value)}"
            elif unit == "seconds":
                mins = float(value) / 60.0
                out[key] = f"{mins:.1f} min"
            elif unit == "km":
                out[key] = f"{float(value):.2f} km"
            elif unit == "%":
                out[key] = f"{float(value) * 100:.1f}%"
            elif unit == "index":
                out[key] = f"{float(value):.3f}"
            else:
                # for anything unclassified
                if isinstance(value, (int, float)):
                    out[key] = f"{value:.2f}".rstrip("0").rstrip(".")
                else:
                    out[key] = str(value)
        except Exception:
            out[key] = str(value)

    return out


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


COL_FIXES = columns = dict(
    zip(
        [
            "room_type_hotel_room",
            "room_type_private_room",
            "room_type_shared_room",
            "privacy_room_in",
        ],
        [
            "room_type_Hotel room",
            "room_type_Private room",
            "room_type_Shared room",
            "privacy_room in",
        ],
    )
)


def short_term_scenario_cleaning(df):
    df["state"] = "il"
    # df["zipcode"] = 60654.0
    df["city"] = "chicago-il"
    df["avg_price"] = 577.59
    df["med_price"] = 169.0
    if "dist_to_bus_dm" not in df:
        df["dist_to_bus_km"] = 5

    df.rename(
        columns=COL_FIXES,
        inplace=True,
    )
    df.index = df.index.astype(int)
    return df
