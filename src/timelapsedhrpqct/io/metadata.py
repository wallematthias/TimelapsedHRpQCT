"""Helpers for extracting and handling metadata from AIM files and other sources."""

from typing import Dict, Any, Union, List


def logtodict(log: str) -> Dict[str, Any]:
    """Convert a formatted AIM processing log string into a dictionary.

    Lines beginning with ``!`` are treated as comments and ignored.  Each
    non-comment line is split on double spaces, which avoids splitting dates
    and other free-text values.  Numeric values are converted to ``int`` or
    ``float`` when the string is purely numeric (allowing for scientific
    notation).  Lines with more than two fields are treated as lists of
    integers (e.g. dimension triples).
    """
    lines = log.split("\n")
    log_dict: Dict[str, Any] = {}

    for line in lines:
        if line.strip() and not line.startswith("!"):
            # first handle explicit colon separators
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                value = val.strip()
                num = value.replace('.', '', 1).replace('-', '', 1).replace('e+', '', 1)
                if num.isdigit():
                    value = float(value) if '.' in value else int(value)
                log_dict[key] = value
                continue

            parts = line.split("  ")  # two spaces to avoid issues with dates
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == 1:
                # fallback to splitting on any whitespace
                pieces = parts[0].split()
                if len(pieces) >= 2:
                    key = pieces[0]
                    remainder = pieces[1:]
                    if len(remainder) == 1:
                        # single trailing item: treat as scalar (possibly numeric)
                        val = remainder[0]
                        num = val.replace('.', '', 1).replace('-', '', 1).replace('e+', '', 1)
                        if num.isdigit():
                            val = float(val) if '.' in val else int(val)
                        value = val
                    else:
                        # multiple items: try to convert to ints list
                        try:
                            value = [int(p) for p in remainder]
                        except ValueError:
                            value = remainder
                    log_dict[key] = value
                continue
            if len(parts) == 2:  # normal key/value pair
                key, value = parts[0], parts[1]
                # convert numeric strings to numbers
                num = value.replace('.', '', 1).replace('-', '', 1).replace('e+', '', 1)
                if num.isdigit():
                    value = float(value) if '.' in value else int(value)
            elif len(parts) > 2:
                key = parts[0]
                # assume remaining parts are ints
                try:
                    value = [int(p) for p in parts[1:]]
                except ValueError:
                    value = parts[1:]
            else:
                # unexpected line type; ignore
                continue
            log_dict[key] = value
    return log_dict


def dicttolog(log_dict: Dict[str, Any]) -> str:
    """Convert a dictionary back into an AIM-style processing log string.

    Numeric values are right-justified, strings are left-justified, and lists
    are dumped as space-separated integers.  The header and split lines mirror
    the format used in the original tutorials.
    """
    log = "! Processing Log\n!\n!-------------------------------------------------------------------------------\n"
    split_line = "!-------------------------------------------------------------------------------\n"
    for key, value in log_dict.items():
        if isinstance(value, (int, float)):
            formatted_line = f"{key.ljust(30)}{value:>23}"
        elif isinstance(value, list):
            list_values = " ".join(f"{v:>10}" for v in value)
            formatted_line = f"{key.ljust(30)}{list_values}"
        else:
            formatted_line = f"{key.ljust(30)}{value}".ljust(80)
        log += formatted_line + "\n"
        if key in [
            "Orig-ISQ-Dim-um",
            "Index Measurement",
            "Default-Eval",
            "HU: mu water",
            "Standard data deviation",
        ]:
            log += split_line
    return log


# backwards-compatible alias for the old simple parser
parse_processing_log = logtodict
