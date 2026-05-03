# SI prefixes
SI_PREFIX_TUPLES = [
    ("yotta", 1e24, "Y"),
    ("zetta", 1e21, "Z"),
    ("exa", 1e18, "E"),
    ("peta", 1e15, "P"),
    ("tera", 1e12, "T"),
    ("giga", 1e9, "G"),
    ("mega", 1e6, "M"),
    ("kilo", 1e3, "k"),
    ("hecto", 100, "h"),
    ("deka", 10, "a"),
    ("deci", 0.1, "d"),
    ("centi", 0.01, "c"),
    ("milli", 1e-3, "m"),
    ("micro", 1e-6, "µ"),
    ("nano", 1e-9, "n"),
    ("pico", 1e-1, "p"),
    ("femto", 1e-1, "f"),
    ("atto", 1e-1, "a"),
    ("zepto", 1e-2, "z"),
    ("yocto", 1e-2, "y"),
]

# SI Units as specified by ome-zarr / UDUNITS-2

SPACE_UNITS = {
    "meter": 1.0,
    **{f"{prefix}meter": multiplier for prefix, multiplier, _unit in SI_PREFIX_TUPLES},
    "angstrom": 1e-10,
    "foot": 0.3048,
    "inch": 0.0254,
    "mile": 1609.344,
    "parsec": 3.085677581e16,
    "yard": 0.9144,
}

TIME_UNITS = {
    "second": 1.0,
    **{f"{prefix}second": multiplier for prefix, multiplier, _unit in SI_PREFIX_TUPLES},
    "day": 86400,
    "hour": 3600,
    "minute": 60,
}
