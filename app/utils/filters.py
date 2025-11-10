

def currency(v, digits=0):
    if v is None:
        return "N/A"
    try:
        return f"${float(v):,.{digits}f}"
    except Exception:
        return str(v)


def comma(v, digits=0):
    if v is None:
        return "N/A"
    try:
        x = float(v)
        if digits and (x % 1 != 0):
            return f"{x:,.{digits}f}"
        return f"{int(round(x)):,}"
    except Exception:
        return str(v)


def percent(v, digits=1, already_pct=False):
    if v is None:
        return "N/A"
    try:
        x = float(v)
        if not already_pct:
            x *= 100.0
        return f"{x:.{digits}f}%"
    except Exception:
        return str(v)
