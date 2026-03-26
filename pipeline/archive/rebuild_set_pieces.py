"""
Rebuild all set piece taker columns in players_raw.csv from verified data.
Clean slate — all players reset to 0 first, then values applied by club+name match.
"""

import pandas as pd
import sys
import os
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")


_SPECIAL = str.maketrans({
    "Ø": "O", "ø": "o",   # Ødegaard -> Odegaard
    "ß": "ss",             # Groß -> Gross
    "Đ": "D", "đ": "d",
    "Ł": "L", "ł": "l",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
})


def strip_accents(s):
    """
    Normalize to ASCII-safe lowercase:
    1. Apply explicit substitutions for non-decomposable chars (Ø, ß, etc.)
    2. NFD decompose then drop combining diacritics (accents, umlauts, etc.)
    """
    s = str(s).translate(_SPECIAL)
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()


# Nickname / short token -> (club, player_id) for ambiguous or alias cases
NICKNAME_OVERRIDES = {
    ("Everton",    "beto"): 311,   # Norberto Bercique Gomes Betuncal
    ("Sunderland", "bi"):   800,   # Djiamgone Jocelin Ta Bi
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE_DIR, "data", "raw", "fpl_api", "players_raw.csv")

# ── Verified set piece data ────────────────────────────────────────────────────
# Format: club -> { "pen": [...], "fk": [...], "corn": [...] }
# Players listed in priority order (index 0 = order 1)

SET_PIECES = {
    "Arsenal": {
        "pen":  ["Saka", "Gyokeres", "Odegaard", "Trossard"],
        "fk":   ["Rice", "Saka"],
        "corn": ["Rice", "Saka", "Madueke", "Odegaard"],
    },
    "Aston Villa": {
        "pen":  ["Buendia", "Watkins"],
        "fk":   ["Rogers", "Digne", "Buendia"],
        "corn": ["Douglas Luiz", "Bailey", "Tielemans", "Sancho", "Cash"],
    },
    "Bournemouth": {
        "pen":  ["Kluivert", "Tavernier"],
        "fk":   ["Unal", "Tavernier", "Kluivert", "Brooks", "Scott"],
        "corn": ["Cook", "Tavernier", "Adli", "Scott"],
    },
    "Brentford": {
        "pen":  ["Thiago", "Schade", "Carvalho", "Jensen"],
        "fk":   ["Lewis-Potter", "Jensen", "Damsgaard"],
        "corn": ["Jensen", "Janelt", "Dango Ouattara"],
    },
    "Brighton": {
        "pen":  ["Milner", "Welbeck", "O'Riley"],
        "fk":   ["De Cuyper", "Ayari", "Dunk", "Welbeck", "Gomez"],
        "corn": ["Gross", "De Cuyper", "O'Riley"],
    },
    "Burnley": {
        "pen":  ["Flemming", "Barnes", "Bruun Larsen"],
        "fk":   ["Ward-Prowse", "Bruun Larsen", "Marcus Edwards", "Tchaouna", "Flemming"],
        "corn": ["Ward-Prowse", "Anthony", "Lucas Pires", "Mejbri"],
    },
    "Chelsea": {
        "pen":  ["Palmer", "Fernandez", "Estevao"],
        "fk":   ["Fernandez", "James", "Palmer", "Neto"],
        "corn": ["Neto", "James", "Fernandez"],
    },
    "Crystal Palace": {
        "pen":  ["Mateta", "Ismaila Sarr", "Devenny"],
        "fk":   ["Pino", "Devenny"],
        "corn": ["Hughes", "Pino", "Wharton", "Johnson"],
    },
    "Everton": {
        "pen":  ["Ndiaye", "Garner", "Beto"],
        "fk":   ["Garner"],
        "corn": ["Garner", "Dewsbury-Hall"],
    },
    "Fulham": {
        "pen":  ["Raul Jimenez", "Wilson"],
        "fk":   ["Wilson", "Raul Jimenez"],
        "corn": ["Wilson", "Iwobi", "Lukic"],
    },
    "Leeds": {
        "pen":  ["Calvert-Lewin", "Nmecha", "Piroe"],
        "fk":   ["Stach", "Longstaff"],
        "corn": ["Stach", "Longstaff", "Justin", "Gruev"],
    },
    "Liverpool": {
        "pen":  ["Salah", "Szoboszlai", "Gakpo", "Mac Allister"],
        "fk":   ["Szoboszlai", "Salah", "Wirtz"],
        "corn": ["Szoboszlai", "Salah", "Gakpo", "Wirtz"],
    },
    "Man City": {
        "pen":  ["Haaland", "Omar Marmoush", "Doku", "Matheus Nunes"],
        "fk":   ["Cherki", "Foden", "Omar Marmoush"],
        "corn": ["Bernardo Silva", "Foden", "Cherki", "Reijnders"],
    },
    "Man Utd": {
        "pen":  ["Bruno Fernandes"],
        "fk":   ["Bruno Fernandes", "Mbeumo"],
        "corn": ["Bruno Fernandes", "Mbeumo"],
    },
    "Newcastle": {
        "pen":  ["Gordon", "Bruno Guimaraes", "Woltemade"],
        "fk":   ["Schar", "Tonali", "Trippier", "Lewis Hall", "Bruno Guimaraes"],
        "corn": ["Tonali", "Lewis Hall", "Bruno Guimaraes", "Trippier", "Elanga"],
    },
    "Nott'm Forest": {
        "pen":  ["Wood", "Gibbs-White", "Igor Jesus", "Anderson", "Mcatee"],
        "fk":   ["Anderson", "Murillo", "Gibbs-White"],
        "corn": ["Anderson", "Hutchinson", "Bakwa", "Ndoye"],
    },
    "Sunderland": {
        "pen":  ["Diarra", "Le Fee"],
        "fk":   ["Xhaka", "Le Fee"],
        "corn": ["Xhaka", "Bi", "Hume", "Le Fee"],
    },
    "Spurs": {
        "pen":  ["Solanke", "Kudus", "Simons", "Richarlison"],
        "fk":   ["Porro", "Simons", "Kudus"],
        "corn": ["Simons", "Porro", "Kudus", "Tel"],
    },
    "West Ham": {
        "pen":  ["Bowen"],
        "fk":   ["Bowen"],
        "corn": ["Bowen", "Summerville", "Mateus Fernandes", "Scarles"],
    },
    "Wolves": {
        "pen":  ["Hwang Hee-Chan", "Arokodare"],
        "fk":   ["Hwang Hee-Chan", "Joao Gomes", "Bellegarde", "Mane"],
        "corn": ["Hugo Bueno", "Mane", "Bellegarde"],
    },
}

# ── Load CSV ───────────────────────────────────────────────────────────────────
players = pd.read_csv(PATH)

# ── Reset all set piece columns to 0 ──────────────────────────────────────────
for col in ["penalties_order", "direct_freekicks_order",
            "corners_and_indirect_freekicks_order",
            "penalty_taker_order", "corner_taker_order", "freekick_taker_order",
            "is_penalty_taker", "is_freekick_taker", "is_corner_taker",
            "is_set_piece_taker"]:
    if col in players.columns:
        players[col] = 0

# Add order columns if missing
for col in ["penalty_taker_order", "corner_taker_order", "freekick_taker_order"]:
    if col not in players.columns:
        players[col] = 0

# ── Name matching helper ───────────────────────────────────────────────────────

def find_player(club_df, name_token, club=None):
    """
    Match name_token against players in club_df.
    - Strips accents from both sides before comparing
    - For multi-word tokens, ALL words must appear in the full name
    - Falls back to nickname overrides for known aliases
    Returns the index list of matching rows.
    """
    tok_clean = strip_accents(name_token.strip())

    # Check nickname override first
    if club is not None:
        key = (club, tok_clean)
        if key in NICKNAME_OVERRIDES:
            pid = NICKNAME_OVERRIDES[key]
            match = club_df[club_df["id"] == pid]
            return match.index.tolist()

    full_name = (
        club_df["first_name"].fillna("") + " " + club_df["second_name"].fillna("")
    ).apply(strip_accents)

    words = tok_clean.split()
    if len(words) == 1:
        mask = full_name.str.contains(words[0], regex=False)
    else:
        # All words must be present somewhere in the full name
        mask = full_name.apply(lambda n: all(w in n for w in words))

    return club_df[mask].index.tolist()


# ── Apply set piece data ───────────────────────────────────────────────────────
unmatched = []   # (club, role, name_token)
ambiguous = []   # (club, role, name_token, count_found)

col_map = {
    "pen":  ("penalties_order",                    "penalty_taker_order",  "is_penalty_taker"),
    "fk":   ("direct_freekicks_order",             "freekick_taker_order", "is_freekick_taker"),
    "corn": ("corners_and_indirect_freekicks_order","corner_taker_order",   "is_corner_taker"),
}

for club, roles in SET_PIECES.items():
    club_df = players[players["team_name"] == club]

    for role, names in roles.items():
        raw_col, order_col, flag_col = col_map[role]

        for order, name_token in enumerate(names, start=1):
            matches = find_player(club_df, name_token, club=club)

            if len(matches) == 0:
                unmatched.append((club, role, name_token))
            elif len(matches) > 1:
                ambiguous.append((club, role, name_token, len(matches),
                                  [players.loc[i, "first_name"] + " " + players.loc[i, "second_name"]
                                   for i in matches]))
            else:
                idx = matches[0]
                players.loc[idx, raw_col]   = order
                players.loc[idx, order_col] = order

# ── Derive binary flag columns ─────────────────────────────────────────────────
players["is_penalty_taker"]  = (players["penalties_order"] >= 1).astype(int)
players["is_freekick_taker"] = (players["direct_freekicks_order"] >= 1).astype(int)
players["is_corner_taker"]   = (players["corners_and_indirect_freekicks_order"] >= 1).astype(int)
players["is_set_piece_taker"] = (
    (players["is_penalty_taker"] == 1) |
    (players["is_freekick_taker"] == 1) |
    (players["is_corner_taker"] == 1)
).astype(int)

# ── Save ───────────────────────────────────────────────────────────────────────
players.to_csv(PATH, index=False)
print("Saved.\n")

# ── Unmatched / ambiguous report ───────────────────────────────────────────────
if unmatched:
    print("=" * 60)
    print("UNMATCHED PLAYERS (need manual resolution):")
    for club, role, name in unmatched:
        print(f"  [{club}] {role.upper()}: '{name}'")
    print()

if ambiguous:
    print("=" * 60)
    print("AMBIGUOUS MATCHES (multiple players found):")
    for club, role, name, count, found in ambiguous:
        print(f"  [{club}] {role.upper()}: '{name}' matched {count} players: {found}")
    print()

# ── Validation table ───────────────────────────────────────────────────────────
print("=" * 70)
print("SET PIECE TAKERS BY CLUB")
print(f"{'Club':<16} {'Player':<32} {'PEN':>4} {'FK':>4} {'CORN':>5}")
print("-" * 70)

for club in sorted(SET_PIECES.keys()):
    club_players = players[
        (players["team_name"] == club) & (players["is_set_piece_taker"] == 1)
    ].sort_values("penalties_order")

    for _, r in club_players.iterrows():
        name = r["first_name"] + " " + r["second_name"]
        if len(name) > 31:
            name = name[:29] + ".."
        pen  = int(r["penalties_order"]) if r["penalties_order"] > 0 else "-"
        fk   = int(r["direct_freekicks_order"]) if r["direct_freekicks_order"] > 0 else "-"
        corn = int(r["corners_and_indirect_freekicks_order"]) if r["corners_and_indirect_freekicks_order"] > 0 else "-"
        print(f"  {club:<14} {name:<32} {str(pen):>4} {str(fk):>4} {str(corn):>5}")

    if club_players.empty:
        print(f"  {club:<14} *** NO SET PIECE TAKERS ***")

print()
print("=" * 70)
print("TOTALS")
print(f"  is_penalty_taker:   {players['is_penalty_taker'].sum()}")
print(f"  is_freekick_taker:  {players['is_freekick_taker'].sum()}")
print(f"  is_corner_taker:    {players['is_corner_taker'].sum()}")
print(f"  is_set_piece_taker: {players['is_set_piece_taker'].sum()}")
