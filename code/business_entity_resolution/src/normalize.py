#!/usr/bin/env python3
"""
Stage 1: Text normalization for business entity resolution.

Handles:
- Unicode NFKD + accent stripping + casefolding
- Legal suffix canonicalization
- Name/address abbreviation expansion
- Postal code extraction (generic: IN 6-digit, US 5/5+4, FR 5-digit)
- Phonetic keys (simplified Double Metaphone)
- Character n-gram and sorted-token representations
"""

import re
import unicodedata
from collections import defaultdict


# ============================================================
# Legal suffixes: map variants -> canonical form
# ============================================================
LEGAL_SUFFIXES = {
    # English
    'corporation': 'corp', 'corp': 'corp',
    'incorporated': 'inc', 'inc': 'inc',
    'limited': 'ltd', 'ltd': 'ltd',
    'company': 'co', 'co': 'co',
    'private': 'pvt', 'pvt': 'pvt',
    'proprietary': 'pty', 'pty': 'pty',
    'llc': 'llc',
    'llp': 'llp',
    'plc': 'plc',
    'lp': 'lp',
    'gmbh': 'gmbh',
    'ag': 'ag',
    # French
    'sarl': 'sarl',
    'sas': 'sas',
    'sa': 'sa',
    'eurl': 'eurl',
    'sci': 'sci',
    'scp': 'scp',
    'snc': 'snc',
    'sasu': 'sasu',
    # Indian
    'nidhi': 'nidhi',
}

# Patterns for legal suffixes - will match at end of name
LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b(' + '|'.join(sorted(LEGAL_SUFFIXES.keys(), key=len, reverse=True)) + r')\b\.?',
    re.IGNORECASE
)

# ============================================================
# Address abbreviations
# ============================================================
ADDRESS_ABBREVS = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bld': 'boulevard', 'bd': 'boulevard',
    'dr': 'drive', 'ct': 'court', 'ln': 'lane', 'pl': 'place',
    'cir': 'circle', 'pkwy': 'parkway', 'hwy': 'highway',
    'apt': 'apartment', 'ste': 'suite', 'flr': 'floor', 'fl': 'floor',
    'bldg': 'building', 'rm': 'room',
    'nr': 'near', 'opp': 'opposite', 'adj': 'adjacent',
    'sec': 'sector', 'ph': 'phase', 'ext': 'extension',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    # French
    'rue': 'rue', 'av': 'avenue', 'bd': 'boulevard', 'bld': 'boulevard',
    'pl': 'place', 'imp': 'impasse', 'all': 'allee',
    'che': 'chemin', 'rte': 'route', 'crs': 'cours',
    'cedex': 'cedex',
}

# ============================================================
# Country normalization
# ============================================================
COUNTRY_ALIASES = {
    'us': 'US', 'usa': 'US', 'united states': 'US', 'united states of america': 'US',
    'u.s.': 'US', 'u.s.a.': 'US', 'america': 'US',
    'india': 'India', 'in': 'India', 'ind': 'India', 'bharat': 'India',
    'france': 'France', 'fr': 'France', 'fra': 'France',
}


def normalize_unicode(text):
    """NFKD normalize, strip accents, casefold."""
    if not text:
        return ""
    # NFKD decomposition
    text = unicodedata.normalize('NFKD', text)
    # Strip combining marks (accents)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Casefold
    text = text.casefold()
    return text


def normalize_punctuation(text):
    """Replace punctuation with spaces, collapse whitespace."""
    if not text:
        return ""
    # Replace & with 'and'
    text = text.replace('&', ' and ')
    # Replace common punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_legal_suffix(name_normalized):
    """Extract and canonicalize legal suffix from business name.
    
    Returns:
        (core_name, legal_suffix) where legal_suffix is canonical form or ''
    """
    if not name_normalized:
        return "", ""
    
    suffixes_found = []
    core = name_normalized
    
    # Find all legal suffixes
    for match in LEGAL_SUFFIX_PATTERN.finditer(name_normalized):
        word = match.group(1).lower().rstrip('.')
        if word in LEGAL_SUFFIXES:
            suffixes_found.append(LEGAL_SUFFIXES[word])
    
    # Remove legal suffixes from name
    core = LEGAL_SUFFIX_PATTERN.sub(' ', core).strip()
    core = re.sub(r'\s+', ' ', core)
    
    suffix = ' '.join(sorted(set(suffixes_found))) if suffixes_found else ''
    return core, suffix


def normalize_country(country_str):
    """Normalize country string to canonical label. Unknown values pass through."""
    if not country_str:
        return ""
    normalized = country_str.strip().lower()
    return COUNTRY_ALIASES.get(normalized, country_str.strip())


def extract_postal_code(address, country=""):
    """Extract postal code from address using generic patterns.
    
    Handles:
    - Indian PIN: 6 digits (may have spaces)
    - US ZIP: 5 digits or 5+4 format
    - French: 5 digits (starting with 0-9)
    - Generic: any 5-6 digit sequence
    
    Returns:
        (postal_code, address_without_postal) 
    """
    if not address:
        return "", address or ""
    
    # Indian PIN code: 6 digits
    pin_match = re.search(r'\b(\d{6})\b', address)
    # US ZIP: 5 digits, optionally followed by -4 digits
    zip_match = re.search(r'\b(\d{5})(?:-(\d{4}))?\b', address)
    
    if pin_match:
        code = pin_match.group(1)
        addr_clean = address[:pin_match.start()] + address[pin_match.end():]
        return code, re.sub(r'\s+', ' ', addr_clean).strip()
    elif zip_match:
        code = zip_match.group(1)
        addr_clean = address[:zip_match.start()] + address[zip_match.end():]
        return code, re.sub(r'\s+', ' ', addr_clean).strip()
    
    return "", address


def extract_house_number(address):
    """Extract house/building number from address."""
    if not address:
        return "", address
    
    # Match patterns like "123 Main St" or "No. 45" or "Plot 12" or "#5"
    patterns = [
        r'^(\d+[\-/]?\d*)\s+',           # Leading number
        r'\bno\s*\.?\s*(\d+)\b',          # No. 45
        r'\b(?:plot|kh|khasra)\s*(?:no\.?)?\s*[\-]?\s*(\d+[\-/]?\d*)\b',  # Plot/Kh numbers
        r'#(\d+)\b',                       # #5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, address, re.IGNORECASE)
        if match:
            return match.group(1), address
    
    return "", address


def extract_landmark(address):
    """Extract landmark phrases from address.
    
    Returns:
        (landmark, address_without_landmark)
    """
    if not address:
        return "", address
    
    landmark_patterns = [
        r'\b(?:near|nr|opp|opposite|behind|beside|adjacent to|next to|in front of|above|below)\s+[^,]+',
    ]
    
    landmarks = []
    addr_clean = address
    for pattern in landmark_patterns:
        match = re.search(pattern, addr_clean, re.IGNORECASE)
        if match:
            landmarks.append(match.group(0))
            addr_clean = addr_clean[:match.start()] + addr_clean[match.end():]
    
    landmark = ' '.join(landmarks)
    return landmark.strip(), re.sub(r'\s+', ' ', addr_clean).strip()


def char_ngrams(text, n=3):
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def sorted_tokens(text):
    """Get sorted, deduplicated tokens for transposition robustness."""
    if not text:
        return ""
    tokens = text.split()
    return ' '.join(sorted(set(tokens)))


def first_token(text):
    """Get the first meaningful token."""
    if not text:
        return ""
    tokens = text.split()
    return tokens[0] if tokens else ""


def simple_phonetic(word):
    """Simplified phonetic key (consonant skeleton + vowel reduction).
    
    This is a lightweight phonetic encoder that:
    1. Removes vowels except leading
    2. Reduces double consonants
    3. Maps similar consonants
    """
    if not word:
        return ""
    
    word = word.lower().strip()
    if not word:
        return ""
    
    # Consonant mapping for similar sounds
    mapping = {
        'b': 'b', 'p': 'b',
        'c': 'k', 'k': 'k', 'q': 'k',
        'd': 'd', 't': 't',
        'f': 'f', 'v': 'f', 'ph': 'f',
        'g': 'g', 'j': 'j',
        'l': 'l', 'r': 'r',
        'm': 'm', 'n': 'n',
        's': 's', 'z': 's', 'x': 's',
        'w': 'w', 'h': '',
        'y': 'y',
    }
    
    result = [word[0]]  # Keep first char
    for c in word[1:]:
        if c in 'aeiou':
            continue
        mapped = mapping.get(c, c)
        if mapped and (not result or result[-1] != mapped):
            result.append(mapped)
    
    return ''.join(result)


def phonetic_key(name):
    """Generate phonetic key for a business name."""
    if not name:
        return ""
    tokens = name.split()
    return ' '.join(simple_phonetic(t) for t in tokens if len(t) > 1)


def normalize_name(name):
    """Full name normalization pipeline.
    
    Returns dict with:
        - raw: original
        - normalized: cleaned full name
        - core: name without legal suffix
        - legal_form: canonical legal suffix
        - sorted_tokens: sorted token form
        - first_token: first token
        - phonetic: phonetic key
        - ngrams3: character 3-grams of core name
    """
    if not name:
        return {
            'raw': '', 'normalized': '', 'core': '', 'legal_form': '',
            'sorted_tokens': '', 'first_token': '', 'phonetic': '',
        }
    
    # Basic normalization
    norm = normalize_unicode(name)
    norm = normalize_punctuation(norm)
    
    # Extract legal suffix
    core, legal = extract_legal_suffix(norm)
    
    return {
        'raw': name,
        'normalized': norm,
        'core': core,
        'legal_form': legal,
        'sorted_tokens': sorted_tokens(core),
        'first_token': first_token(core),
        'phonetic': phonetic_key(core),
    }


def normalize_address(address, country=""):
    """Full address normalization pipeline.
    
    Returns dict with:
        - raw: original
        - normalized: cleaned full address
        - core: address without landmark and postal code
        - postal_code: extracted postal code
        - house_number: extracted house/building number
        - landmark: extracted landmark
        - tokens: sorted unique tokens
    """
    if not address:
        return {
            'raw': '', 'normalized': '', 'core': '', 'postal_code': '',
            'house_number': '', 'landmark': '', 'tokens': '',
        }
    
    # Basic normalization
    norm = normalize_unicode(address)
    norm = normalize_punctuation(norm)
    
    # Expand address abbreviations
    tokens = norm.split()
    expanded = []
    for t in tokens:
        expanded.append(ADDRESS_ABBREVS.get(t, t))
    norm = ' '.join(expanded)
    
    # Extract components
    postal, no_postal = extract_postal_code(norm, country)
    landmark, no_landmark = extract_landmark(no_postal)
    house_num, _ = extract_house_number(no_landmark)
    
    return {
        'raw': address,
        'normalized': norm,
        'core': no_landmark,
        'postal_code': postal,
        'house_number': house_num,
        'landmark': landmark,
        'tokens': sorted_tokens(no_landmark),
    }


def normalize_record(entity_id, name, address, country):
    """Normalize a complete business record.
    
    Returns a flat dict suitable for DataFrame construction.
    """
    country_norm = normalize_country(country)
    name_info = normalize_name(name)
    addr_info = normalize_address(address, country_norm)
    
    return {
        'entity_id': entity_id,
        'country': country_norm,
        'country_raw': country,
        # Name fields
        'name_raw': name_info['raw'],
        'name_norm': name_info['normalized'],
        'name_core': name_info['core'],
        'legal_form': name_info['legal_form'],
        'name_sorted': name_info['sorted_tokens'],
        'name_first_token': name_info['first_token'],
        'name_phonetic': name_info['phonetic'],
        # Address fields
        'addr_raw': addr_info['raw'],
        'addr_norm': addr_info['normalized'],
        'addr_core': addr_info['core'],
        'postal_code': addr_info['postal_code'],
        'house_number': addr_info['house_number'],
        'landmark': addr_info['landmark'],
        'addr_tokens': addr_info['tokens'],
    }


# ============================================================
# Unit tests
# ============================================================
def test_normalization():
    """Test normalization functions."""
    # Unicode / accent stripping
    assert normalize_unicode("Café") == "cafe"
    assert normalize_unicode("Société Générale") == "societe generale"
    assert normalize_unicode("München") == "munchen"
    
    # Punctuation
    assert normalize_punctuation("A & B") == "A and B"
    assert normalize_punctuation("A.B.C.") == "A B C"
    
    # Legal suffix extraction
    core, legal = extract_legal_suffix("acme corporation")
    assert core == "acme"
    assert legal == "corp"
    
    core, legal = extract_legal_suffix("xyz pvt ltd")
    assert "pvt" in legal and "ltd" in legal
    assert core == "xyz"
    
    core, legal = extract_legal_suffix("sarl des arts")
    assert legal == "sarl"
    
    # Country normalization
    assert normalize_country("US") == "US"
    assert normalize_country("USA") == "US"
    assert normalize_country("India") == "India"
    assert normalize_country("France") == "France"
    assert normalize_country("Germany") == "Germany"  # Unknown passes through
    
    # Postal code extraction
    code, _ = extract_postal_code("123 Main St, NY 10001")
    assert code == "10001"
    
    code, _ = extract_postal_code("Delhi 110001")
    assert code == "110001"
    
    code, _ = extract_postal_code("75001 Paris")
    assert code == "75001"
    
    # French examples
    name_info = normalize_name("Société Générale SA")
    assert name_info['legal_form'] == 'sa'
    assert 'generale' in name_info['core']
    
    addr_info = normalize_address("175 Boulevard du Président Roosevelt, 33000 Bordeaux")
    assert addr_info['postal_code'] == '33000'
    
    print("All normalization tests PASSED!")


if __name__ == "__main__":
    test_normalization()
