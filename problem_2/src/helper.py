import unicodedata

def normalize_spaces(name):
    # Strip and replace multiple spaces with a single space.
    return " ".join(name.split())

def normalize_string(s):
    """Normalize a string: lower-case, strip, and remove diacritics."""
    if not isinstance(s, str):
        return s
    s = normalize_spaces(s)
    s = s.lower().strip()
    s = unicodedata.normalize('NFKD', s)
    # replace & for "and"
    s = s.replace("&", "and")
    # Normalize use of curly or straight single quotes
    s = s.replace("’", "'")
    return "".join(c for c in s if not unicodedata.combining(c))

