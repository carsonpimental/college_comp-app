import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_path(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(BASE_DIR), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


BATTING_PATH = _resolve_data_path("pimental_batting.csv")
PITCHING_PATH = _resolve_data_path("pimental_pitching.csv")

PASTE_FIELDS = [
    "ab",
    "r",
    "h",
    "2b",
    "3b",
    "hr",
    "rbi",
    "sb",
    "cs",
    "bb",
    "so",
    "hbp",
    "sh",
    "sf",
    "ibb",
    "gdp",
    "tb",
    "pa",
    "xbh",
    "1b",
    "avg",
    "obp",
    "slg",
    "ops",
    "seca",
    "iso",
    "babip",
    "bb%",
    "so%",
    "so/bb",
    "ab/hr",
]

PITCHER_PASTE_FIELDS = [
    "era",
    "g",
    "gs",
    "cg",
    "sho",
    "gr",
    "gf",
    "sv",
    "ip",
    "h",
    "r",
    "er",
    "hr",
    "bb",
    "so",
    "wp",
    "bk",
    "hb",
    "whip",
    "h9",
    "hr9",
    "bb9",
    "so9",
    "ra9",
    "so/bb",
]

SIM_FIELDS_DISPLAY = [
    "AB",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "RBI",
    "TB",
    "SB",
    "BB",
    "K",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
    "BB%",
    "K%",
]

PITCHER_DISPLAY_RENAMES = {
    "g": "G",
    "ip": "IP",
    "era": "ERA",
    "h": "H",
    "hr": "HR",
    "bb": "BB",
    "so": "K",
    "hb": "HBP",
    "whip": "WHIP",
    "h9": "H/9",
    "hr9": "HR/9",
    "bb9": "BB/9",
    "so9": "K/9",
    "so/bb": "K/BB",
}

PITCHER_SIM_FIELDS_PASTE = [
    "era",
    "g",
    "ip",
    "h",
    "hr",
    "bb",
    "so",
    "hb",
    "whip",
    "h9",
    "hr9",
    "bb9",
    "so9",
    "so/bb",
]

PITCHER_SIM_FIELDS_DISPLAY = [PITCHER_DISPLAY_RENAMES[f] for f in PITCHER_SIM_FIELDS_PASTE]

PITCHER_DISPLAY_TO_PASTE = {PITCHER_DISPLAY_RENAMES[f]: f for f in PITCHER_SIM_FIELDS_PASTE}
PITCHER_DISPLAY_TO_CSV = {PITCHER_DISPLAY_RENAMES[f]: f for f in PITCHER_SIM_FIELDS_PASTE}

PITCHER_SIM_FIELDS_CSV = [PITCHER_DISPLAY_TO_CSV[field] for field in PITCHER_SIM_FIELDS_DISPLAY]

DISPLAY_TO_PASTE = {
    "AB": "ab",
    "H": "h",
    "1B": "1b",
    "2B": "2b",
    "3B": "3b",
    "HR": "hr",
    "RBI": "rbi",
    "TB": "tb",
    "SB": "sb",
    "BB": "bb",
    "K": "so",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "BB%": "bb%",
    "K%": "so%",
}

DISPLAY_TO_CSV = {
    "AB": "ab",
    "H": "h",
    "1B": "sgl",
    "2B": "dbl",
    "3B": "tpl",
    "HR": "hr",
    "RBI": "rbi",
    "TB": "tb",
    "SB": "sb",
    "BB": "bb",
    "K": "so",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "BB%": "bb%",
    "K%": "so%",
}

SIM_FIELDS_CSV = [DISPLAY_TO_CSV[field] for field in SIM_FIELDS_DISPLAY]


@dataclass(frozen=True)
class Match:
    idx: int
    name: str
    year: str
    team: str
    league: str
    distance: float
    score: float


def _to_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt_num(x, display_field: str | None = None) -> str:
    v = _to_float(x)
    if v is None or math.isnan(v) or math.isinf(v):
        return "—"

    if display_field in {"BB%", "K%"}:
        return f"{v:.2f}".rstrip("0").rstrip(".")

    if display_field in {"ERA", "WHIP", "H/9", "HR/9", "BB/9", "K/9", "RA9", "K/BB"}:
        return f"{v:.2f}".rstrip("0").rstrip(".")

    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.3f}"


@st.cache_data(show_spinner=False)
def load_batting(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            "Batting CSV not found. "
            + f"path={path} | BASE_DIR={BASE_DIR} | CWD={os.getcwd()}"
        )
        try:
            st.write("BASE_DIR contents", os.listdir(BASE_DIR))
        except Exception:
            pass
        try:
            st.write("CWD contents", os.listdir(os.getcwd()))
        except Exception:
            pass
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_pitching(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(
            "Pitching CSV not found. "
            + f"path={path} | BASE_DIR={BASE_DIR} | CWD={os.getcwd()}"
        )
        try:
            st.write("BASE_DIR contents", os.listdir(BASE_DIR))
        except Exception:
            pass
        try:
            st.write("CWD contents", os.listdir(os.getcwd()))
        except Exception:
            pass
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def compute_stds(df: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    numeric = df[list(fields)].map(_to_float)
    numeric = numeric.dropna(axis=0, how="any")
    arr = numeric.to_numpy(dtype=float)
    stds = arr.std(axis=0)
    stds[stds == 0] = 1.0
    return stds


@st.cache_data(show_spinner=False)
def count_complete_rows(df: pd.DataFrame, fields: tuple[str, ...]) -> int:
    return int(df[list(fields)].map(_to_float).dropna(axis=0, how="any").shape[0])


def parse_pasted_line(text: str, fields: list[str] | None = None) -> dict:
    use_fields = fields or PASTE_FIELDS
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        raise ValueError("Paste a Baseball Cube stat line.")

    # Pitching paste lines sometimes include an empty column (commonly GF) which gets dropped
    # when splitting on whitespace. If exactly one value is missing, assume GF=0.
    if use_fields == PITCHER_PASTE_FIELDS and len(tokens) == len(use_fields) - 1:
        gf_index = use_fields.index("gf")
        tokens = tokens[:gf_index] + ["0"] + tokens[gf_index:]

    if len(tokens) != len(use_fields):
        raise ValueError(f"Expected {len(use_fields)} values, found {len(tokens)}.")

    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError as e:
            raise ValueError(f"Invalid number: {t}") from e

    return dict(zip(use_fields, values))


def build_input_vector(
    paste_map: dict,
    sim_fields_display: list[str] = SIM_FIELDS_DISPLAY,
    display_to_paste: dict[str, str] = DISPLAY_TO_PASTE,
) -> np.ndarray:
    return np.array([paste_map[display_to_paste[f]] for f in sim_fields_display], dtype=float)


def build_row_vector(
    row: pd.Series,
    sim_fields_display: list[str] = SIM_FIELDS_DISPLAY,
    display_to_csv: dict[str, str] = DISPLAY_TO_CSV,
) -> np.ndarray | None:
    vals = []
    for display_field in sim_fields_display:
        csv_field = display_to_csv[display_field]
        v = _to_float(row.get(csv_field))
        if v is None:
            return None
        vals.append(v)
    return np.array(vals, dtype=float)


def standardized_distance(a: np.ndarray, b: np.ndarray, stds: np.ndarray) -> float:
    z = (a - b) / stds
    return float(np.sqrt(np.sum(z**2)))


def score_from_distance(d: float) -> float:
    # 0..1 where 1 is best
    return 1.0 / (1.0 + d)


def main():
    st.set_page_config(page_title="Comp Radar", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Roboto+Mono:wght@400;600;700&display=swap');

        html, body, [class*="css"], .stApp {
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          color: #f4f1fa;
        }

        .stApp {
          background: linear-gradient(135deg, #120a1f 0%, #0a0712 45%, #060409 100%);
        }

        .block-container {
          max-width: 1180px;
          padding-top: 40px !important;
          padding-bottom: 28px;
          padding-left: 18px;
          padding-right: 18px;
        }

        h1, h2, h3 {
          letter-spacing: -0.03em;
        }

        .app-title {
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 6px 0 18px;
          padding: 18px 28px;
          border-radius: 18px;
          background: rgba(139,92,246,0.10);
          border: 1px solid rgba(167,139,250,0.22);
          position: relative;
          overflow: hidden;
        }

        .app-title > div {
          flex: 0 1 auto;
          text-align: center;
          position: relative;
          z-index: 2;
        }

        .app-title h1 {
          font-size: 44px;
          margin: 0;
          font-weight: 800;
        }

        .app-subtitle {
          margin-top: 6px;
          font-size: 16px;
          font-weight: 700;
          color: rgba(196,166,255,0.85);
        }

        .app-desc {
          margin-top: 10px;
          font-size: 15px;
          color: #8a92ad;
          font-weight: 500;
          width: 100%;
          max-width: 560px;
          margin-left: auto;
          margin-right: auto;
          line-height: 1.35;
        }

        .banner-ball {
          position: absolute;
          right: -40px;
          top: 50%;
          transform: translateY(-50%);
          width: 280px;
          height: 280px;
          z-index: 1;
          pointer-events: none;
          opacity: 1;
        }

        .chip {
          display:inline-block;
          padding: 8px 12px;
          border-radius: 999px;
          background: rgba(167,139,250,0.12);
          border: 1px solid rgba(167,139,250,0.30);
          font-size: 12px;
          color: #c4a6ff;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }

        .perf-foot {
          display: flex;
          justify-content: flex-end;
          padding: 0 18px 14px;
        }

        .chip-foot {
          font-size: 11.5px;
          padding: 7px 10px;
        }

        .muted { color: #a89bc4; }

        .section-header {
          font-size: 24px;
          font-weight: 800;
          margin: 0;
          padding: 0;
        }

        .banner-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          padding: 16px 0;
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        #input-anchor + div[data-testid="stVerticalBlock"] .banner-head,
        #perf-anchor + div[data-testid="stVerticalBlock"] .banner-head {
          width: 100%;
          box-sizing: border-box;
        }

        .section-gap { height: 18px; }

        #input-anchor + div[data-testid="stVerticalBlock"] {
          background: linear-gradient(175deg, rgba(167,139,250,0.10), rgba(167,139,250,0.02));
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 22px;
          overflow: hidden;
          box-shadow: 0 30px 70px -35px rgba(0,0,0,0.90);
          padding: 18px 18px;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] .banner-head {
          margin: -18px -18px 18px;
        }

        .comparison-card {
          margin-top: 18px;
          border-radius: 22px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012));
          overflow: hidden;
          box-shadow: 0 30px 70px -35px rgba(0,0,0,0.90);
        }

        .comparison-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          padding: 16px 18px;
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        .comparison-head .lhs {
          display: flex;
          align-items: baseline;
          gap: 10px;
          flex-wrap: wrap;
        }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] .lhs {
          margin: 0;
          padding: 0;
        }

        .comparison-head .lhs .vs {
          color: #a89bc4;
          font-weight: 700;
        }

        .table-wrap {
          overflow-x: auto;
          padding: 14px 14px 18px;
        }

        table.comp {
          width: max-content;
          min-width: 100%;
          border-collapse: separate;
          border-spacing: 0;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 16px;
          color: #e2ddf0;
        }

        table.comp thead th {
          position: sticky;
          top: 0;
          background: rgba(12, 7, 20, 0.92);
          border-bottom: 1px solid rgba(255,255,255,0.08);
          padding: 10px 12px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          font-size: 13px;
          color: #a89bc4;
          white-space: nowrap;
          text-align: right;
        }

        table.comp thead th.player,
        table.comp tbody td.player {
          text-align: left;
        }

        table.comp tbody td {
          padding: 10px 12px;
          border-bottom: 1px solid rgba(255,255,255,0.06);
          white-space: nowrap;
          text-align: right;
        }

        table.comp tbody tr:last-child td { border-bottom: 0; }

        table.comp tbody td.player {
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          font-weight: 800;
          color: #f4f1fa;
        }

        table.comp thead th.player {
          position: sticky;
          left: 0;
          z-index: 3;
        }

        table.comp tbody td.player {
          position: sticky;
          left: 0;
          z-index: 2;
          background: rgba(12, 7, 20, 0.92);
        }

        table.comp tbody tr.you td {
          background: rgba(139,92,246,0.14);
        }

        table.comp tbody tr.you td.player {
          background: rgba(139,92,246,0.20);
          color: #c4a6ff;
        }

        #comp-anchor + div[data-testid="stVerticalBlock"] {
          margin-top: 18px;
          border-radius: 22px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012));
          overflow: hidden;
          box-shadow: 0 30px 70px -35px rgba(0,0,0,0.90);
          width: 100% !important;
          max-width: none !important;
          box-sizing: border-box;
        }

        #perf-anchor + div[data-testid="stVerticalBlock"] {
          margin-top: 18px;
          border-radius: 22px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012));
          overflow: hidden;
          box-shadow: 0 30px 70px -35px rgba(0,0,0,0.90);
          width: 100% !important;
          max-width: none !important;
          box-sizing: border-box;
        }

        .perf-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          padding: 16px 18px;
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        .perf-table-wrap { padding: 12px 18px 16px; }

        table.perf {
          width: 100%;
          border-collapse: separate;
          border-spacing: 0 12px;
        }

        table.perf thead th {
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 18px;
          color: #a89bc4;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          text-align: center;
          padding: 0 10px 6px;
          white-space: nowrap;
          border: 0;
        }

        table.perf thead th.num { text-align: center; }
        table.perf thead th.pct { text-align: center; color: #c4a6ff; }

        table.perf tbody tr {
          background: rgba(255,255,255,0.025);
        }

        table.perf tbody tr.top {
          background: linear-gradient(150deg, rgba(139,92,246,0.18), rgba(139,92,246,0.03));
        }

        table.perf tbody td {
          padding: 14px 10px;
          vertical-align: middle;
          border: 0;
          text-align: center;
          font-size: 18px;
        }

        table.perf tbody td:first-child {
          padding-left: 14px;
        }

        .perf-name {
          font-size: 18px;
          font-weight: 800;
          line-height: 1.15;
          text-align: center;
        }

        .perf-num {
          text-align: center;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 700;
          font-size: 18px;
          color: rgba(226, 221, 240, 0.86);
          white-space: nowrap;
        }

        .perf-pct {
          text-align: right;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 800;
          font-size: 18px;
          white-space: nowrap;
          color: #c4a6ff;
        }

        #comp-anchor + div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { gap: 0px; }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] {
          display: flex;
          flex-direction: row;
          align-items: baseline;
          gap: 6px;
          flex-wrap: nowrap;
          padding: 16px 18px;
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] > div {
          flex: 0 0 auto;
        }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] {
          width: 220px;
          max-width: 220px;
        }

        #comp-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          border-radius: 13px !important;
          box-shadow: none !important;
        }

        #comp-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] div[role="combobox"] {
          min-height: 26px !important;
          padding: 1px 8px !important;
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          font-size: 20px !important;
          font-weight: 800 !important;
          color: #f4f1fa !important;
        }

        #comp-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] svg {
          width: 18px !important;
          height: 18px !important;
        }

        

        

        .field-label {
          font-size: 14px;
          letter-spacing: 0.09em;
          text-transform: uppercase;
          font-weight: 600;
          color: #a89bc4;
          margin-top: 16px;
          margin-bottom: 10px;
        }

        .toggle-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 10px;
          margin-bottom: 18px;
        }

        .toggle-label {
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          font-size: 16px;
          font-weight: 600;
          color: #e2ddf0;
        }

        .toggle-track {
          width: 42px;
          height: 24px;
          border-radius: 999px;
          background: #0c0714;
          border: 1px solid rgba(255,255,255,0.10);
          position: relative;
          box-sizing: border-box;
        }

        .toggle-knob {
          width: 18px;
          height: 18px;
          border-radius: 999px;
          background: #5a5170;
          position: absolute;
          top: 50%;
          left: 2px;
          transform: translateY(-50%);
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stButton"] {
          margin-top: 16px;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
          font-size: 34px !important;
          padding: 18px 14px !important;
          letter-spacing: 0.10em !important;
          line-height: 1.1 !important;
          width: 100% !important;
          text-align: center !important;
        }

        .player-type-toggle {
          display: flex;
          gap: 8px;
          width: 100%;
        }

        .player-type-toggle div[data-testid="stButton"] {
          flex: 1 1 0%;
        }

        .player-type-toggle div[data-testid="stButton"] button {
          width: 100% !important;
          min-height: 44px !important;
          height: 44px !important;
          border-radius: 13px !important;
          padding: 13px 20px !important;
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          box-shadow: none !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
        }

        .player-type-toggle div[data-testid="stButton"] button p {
          margin: 0 !important;
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          font-size: 16px !important;
          font-weight: 700 !important;
          color: #a89bc4 !important;
          letter-spacing: 0 !important;
        }

        .player-type-toggle div[data-testid="stButton"].selected button {
          background: linear-gradient(150deg, #c4a6ff, #8b5cf6) !important;
          border: 0 !important;
          box-shadow: 0 12px 28px -10px rgba(139,92,246,0.6), inset 0 1px 0 rgba(255,255,255,0.45) !important;
        }

        .player-type-toggle div[data-testid="stButton"].selected button p {
          color: #1a0f30 !important;
          font-weight: 800 !important;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stToggle"] {
          display: flex;
          align-items: center;
          min-height: 44px;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stToggle"] {
          justify-content: flex-start;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stToggle"] [data-baseweb="switch"] {
          transform: scale(1.35);
          transform-origin: left center;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stToggle"] [data-baseweb="switch"] > div {
          height: 34px !important;
          width: 62px !important;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stToggle"] [data-baseweb="switch"] > div > div {
          height: 28px !important;
          width: 28px !important;
          top: 3px !important;
          left: 3px !important;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stToggle"] div[role="switch"] {
          transform: scale(1.35);
          transform-origin: left center;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] div[role="combobox"] {
          min-height: 44px !important;
          padding-top: 10px !important;
          padding-bottom: 10px !important;
        }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] {
          display: flex;
          flex-direction: row;
          align-items: baseline;
          gap: 8px;
          flex-wrap: nowrap;
          width: 100%;
          box-sizing: border-box;
          padding: 16px 18px;
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stMarkdown"] {
          margin: 0 !important;
        }

        #comp-inline-anchor + div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] {
          flex: 0 0 auto;
          min-width: 240px;
        }

        #input-anchor + div[data-testid="stVerticalBlock"] .stTextInput,
        #input-anchor + div[data-testid="stVerticalBlock"] .stTextArea,
        #input-anchor + div[data-testid="stVerticalBlock"] .stButton,
        #input-anchor ~ div[data-testid="stVerticalBlock"] .stSelectbox {
          margin: 0px !important;
          padding: 0px !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] .stTextInput { margin-bottom: 16px !important; }
        #input-anchor ~ div[data-testid="stVerticalBlock"] .stTextArea { margin-bottom: 14px !important; }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] {
          width: 100% !important;
          display: flex !important;
          align-items: center !important;
          min-height: 44px !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] {
          display: flex !important;
          width: 100% !important;
          gap: 8px !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] {
          flex: 1 1 0% !important;
          margin: 0 !important;
          cursor: pointer !important;
          width: 100% !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div {
          display: flex !important;
          justify-content: center !important;
          align-items: center !important;
          min-height: 44px !important;
          height: 44px !important;
          border-radius: 13px !important;
          padding: 13px 20px !important;
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          box-sizing: border-box !important;
          width: 100% !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div > div:first-of-type {
          display: none !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] p {
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          font-size: 16px !important;
          font-weight: 700 !important;
          color: #a89bc4 !important;
          margin: 0 !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div {
          background: linear-gradient(150deg, #c4a6ff, #8b5cf6) !important;
          border: 0 !important;
          box-shadow: 0 12px 28px -10px rgba(139,92,246,0.6), inset 0 1px 0 rgba(255,255,255,0.45) !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div p {
          color: #1a0f30 !important;
          font-weight: 800 !important;
        }

        #input-anchor ~ div[data-testid="stVerticalBlock"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] input {
          appearance: none !important;
          -webkit-appearance: none !important;
          position: absolute !important;
          width: 1px !important;
          height: 1px !important;
          overflow: hidden !important;
          clip: rect(0 0 0 0) !important;
        }

        /* Fallback/global st.radio pill styling (in case #input-anchor selectors don't match Streamlit's DOM) */
        .stApp div[data-testid="stRadio"] div[role="radiogroup"] {
          display: flex !important;
          width: 100% !important;
          gap: 8px !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] {
          flex: 1 1 0% !important;
          margin: 0 !important;
          cursor: pointer !important;
          width: 100% !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div {
          display: flex !important;
          justify-content: center !important;
          align-items: center !important;
          min-height: 44px !important;
          height: 44px !important;
          border-radius: 13px !important;
          padding: 13px 20px !important;
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          box-sizing: border-box !important;
          width: 100% !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div > div:first-of-type {
          display: none !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] p {
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          font-size: 16px !important;
          font-weight: 700 !important;
          color: #a89bc4 !important;
          margin: 0 !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div {
          background: linear-gradient(150deg, #c4a6ff, #8b5cf6) !important;
          border: 0 !important;
          box-shadow: 0 12px 28px -10px rgba(139,92,246,0.6), inset 0 1px 0 rgba(255,255,255,0.45) !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div p {
          color: #1a0f30 !important;
          font-weight: 800 !important;
        }

        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] input {
          appearance: none !important;
          -webkit-appearance: none !important;
          position: absolute !important;
          width: 1px !important;
          height: 1px !important;
          overflow: hidden !important;
          clip: rect(0 0 0 0) !important;
        }

        /* Streamlit Cloud (newer DOM): stRadioGroup / React Aria */
        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] {
          display: flex !important;
          width: 100% !important;
          gap: 8px !important;
          align-items: center !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] [role="radio"],
        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] label {
          flex: 1 1 0% !important;
          width: 100% !important;
          min-height: 44px !important;
          height: 44px !important;
          border-radius: 13px !important;
          padding: 13px 20px !important;
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          box-sizing: border-box !important;
          display: flex !important;
          justify-content: center !important;
          align-items: center !important;
          cursor: pointer !important;
          gap: 10px !important;
          user-select: none !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] [role="radio"] svg,
        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] label svg,
        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] input[type="radio"] {
          display: none !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] [role="radio"] *,
        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] label * {
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          font-size: 16px !important;
          font-weight: 700 !important;
          color: #a89bc4 !important;
          margin: 0 !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] [role="radio"][aria-checked="true"] {
          background: linear-gradient(150deg, #c4a6ff, #8b5cf6) !important;
          border: 0 !important;
          box-shadow: 0 12px 28px -10px rgba(139,92,246,0.6), inset 0 1px 0 rgba(255,255,255,0.45) !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] [role="radio"][aria-checked="true"] * {
          color: #1a0f30 !important;
          font-weight: 800 !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] label:has(input:checked) {
          background: linear-gradient(150deg, #c4a6ff, #8b5cf6) !important;
          border: 0 !important;
          box-shadow: 0 12px 28px -10px rgba(139,92,246,0.6), inset 0 1px 0 rgba(255,255,255,0.45) !important;
        }

        .stApp div[data-testid="stRadio"] div[data-testid="stRadioGroup"][role="radiogroup"] label:has(input:checked) * {
          color: #1a0f30 !important;
          font-weight: 800 !important;
        }

        .stTextInput label, .stTextArea label { display: none; }

        div[data-testid="stTextInput"] input {
          width: 100% !important;
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          border-radius: 13px !important;
          color: #e2ddf0 !important;
          font-family: Manrope, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
          font-size: 15px !important;
          font-weight: 600 !important;
          padding: 13px 14px !important;
          box-shadow: none !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
          color: #453a5c !important;
          font-weight: 400 !important;
        }

        div[data-testid="stTextArea"] textarea {
          width: 100% !important;
          height: 150px !important;
          background: #0c0714 !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
          border-radius: 13px !important;
          color: #e2ddf0 !important;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
          font-size: 16px !important;
          line-height: 1.6 !important;
          padding: 12px 14px !important;
          resize: none !important;
          box-shadow: none !important;
        }

        div[data-testid="stTextArea"] small {
          display: none !important;
        }

        div[data-testid="stTextArea"] textarea::placeholder {
          color: #453a5c !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
          border-color: rgba(167,139,250,0.35) !important;
          outline: none !important;
          box-shadow: 0 0 0 4px rgba(139,92,246,0.18) !important;
        }

        .stButton button {
          width: 100% !important;
          border-radius: 13px;
          padding: 14px;
          border: 0;
          font-weight: 800;
          font-size: 15.5px;
          color: #1a0f30;
          background: linear-gradient(150deg, #c4a6ff, #8b5cf6);
          box-shadow: 0 12px 28px -10px rgba(139,92,246,0.6), inset 0 1px 0 rgba(255,255,255,0.45);
          text-align: center !important;
        }

        .stButton button p {
          font-size: 24px !important;
          font-weight: 800 !important;
          color: #f4f1fa !important;
          letter-spacing: 0.06em !important;
          line-height: 1.15 !important;
          margin: 0 !important;
        }

        .stButton button:hover {
          filter: brightness(1.04);
          box-shadow: 0 18px 44px rgba(139,92,246,0.45);
        }

        .quick-stat {
          display: flex;
          justify-content: space-between;
          padding: 10px 0;
          border-top: 1px solid rgba(255,255,255,0.07);
          font-size: 14px;
        }

        .quick-stat .value {
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 700;
          color: #f4f1fa;
        }

        .results-header {
          display: grid;
          grid-template-columns: 30px minmax(0, 1fr) 70px 70px 70px 70px;
          align-items: baseline;
          margin-bottom: 14px;
        }

        .results-header h2 {
          margin: 0;
          font-size: inherit;
          font-weight: inherit;
          text-align: left;
          grid-column: 1 / 3;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .year-hdr {
          grid-column: 3;
        }

        .round-hdr { grid-column: 4; }
        .pick-hdr { grid-column: 5; }
        .score-hdr { grid-column: 6; }

        .match-score-label {
          font-size: 12px;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: #a89bc4;
          font-weight: 800;
          text-align: center;
        }

        .result-row {
          display: grid;
          grid-template-columns: 30px 1fr 70px 70px 70px 70px;
          align-items: center;
          gap: 16px;
          padding: 13px 16px;
          border-radius: 14px;
          background: rgba(255,255,255,0.025);
          border: 1px solid rgba(255,255,255,0.07);
        }

        .rows {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        

        .result-row.top {
          background: linear-gradient(150deg, rgba(139,92,246,0.18), rgba(139,92,246,0.03));
        }

        .rank {
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          color: #a89bc4;
          font-weight: 700;
          font-size: 13px;
        }

        .rank.top { color: #c4a6ff; }

        .player-name {
          font-size: 20px;
          font-weight: 800;
          line-height: 1.05;
        }

        .player-meta {
          font-size: 13px;
          color: #a89bc4;
          margin-top: 3px;
        }

        .year {
          text-align: center;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 700;
          color: #f4f1fa;
        }

        .draft {
          text-align: center;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 700;
          color: #f4f1fa;
        }

        .pct {
          text-align: right;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 800;
          font-size: 20px;
        }

        @media (max-width: 900px) {
          .block-container {
            max-width: 760px;
            padding-left: 14px;
            padding-right: 14px;
          }

          .perf-name { font-size: 18px; }
          .perf-pct { font-size: 18px; }
        }

        .comparison {
          margin-top: 14px;
          border-top: 1px solid rgba(255,255,255,0.07);
          padding-top: 12px;
        }

        .comparison h3 {
          margin: 0 0 10px;
          font-size: 16px;
          font-weight: 800;
        }

        .stat-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }

        .stat-cell {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 14px;
          padding: 10px 10px;
        }

        .stat-cell .k {
          font-size: 12px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #a89bc4;
          margin-bottom: 2px;
        }

        .stat-cell .v {
          font-size: 16px;
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-weight: 700;
          color: #f4f1fa;
        }

        [data-testid="column"] { padding-left: 0px !important; padding-right: 0px !important; }

        @media (max-width: 980px) {
          .result-row { grid-template-columns: 36px 1fr 60px 1fr 92px; }
          .pct { font-size: 18px; }
          .stat-grid { grid-template-columns: repeat(2, 1fr); }
        }

        /* Make Streamlit dataframes match theme */
        div[data-testid="stDataFrame"] {
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 18px;
          overflow: hidden;
        }

        div[data-testid="stDataFrame"] table {
          text-align: center !important;
        }

        div[data-testid="stDataFrame"] thead th,
        div[data-testid="stDataFrame"] tbody td {
          text-align: center !important;
        }

        div[data-testid="stDataFrame"] * {
          font-family: Roboto Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    batting_df = load_batting(BATTING_PATH)
    pitching_df = load_pitching(PITCHING_PATH)
    total_rows = int(len(batting_df) + len(pitching_df))

    st.markdown(
        '<div class="app-title">'
        '<svg class="banner-ball" viewBox="0 0 280 280" aria-hidden="true">'
        '<circle cx="140" cy="140" r="118" fill="none" stroke="rgba(167,139,250,0.16)" stroke-width="1" />'
        '<path d="M 70 78 C 108 112, 172 112, 210 78" fill="none" stroke="rgba(196,166,255,0.28)" stroke-width="2" stroke-dasharray="4 6" stroke-linecap="round" />'
        '<path d="M 70 202 C 108 168, 172 168, 210 202" fill="none" stroke="rgba(196,166,255,0.28)" stroke-width="2" stroke-dasharray="4 6" stroke-linecap="round" />'
        '</svg>'
        '<div><h1>College Baseball Comps</h1><div class="app-subtitle">Created by Carson Pimental</div>'
        f'<div class="app-desc">A similarity engine for college baseball hitters, matching any pasted D1 statistic line against {total_rows:,} season performances from 2015–2025 to surface the five strongest statistical comps.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    conferences_batting = (
        sorted(
            [
                str(x).strip()
                for x in batting_df.get("leaguename", pd.Series([], dtype=object)).dropna().unique()
                if str(x).strip()
            ]
        )
        if "leaguename" in batting_df.columns
        else []
    )

    conferences_pitching = (
        sorted(
            [
                str(x).strip()
                for x in pitching_df.get("leaguename", pd.Series([], dtype=object)).dropna().unique()
                if str(x).strip()
            ]
        )
        if "leaguename" in pitching_df.columns
        else []
    )

    st.markdown('<div id="input-anchor"></div>', unsafe_allow_html=True)
    input_block = st.container()
    with input_block:
        st.markdown(
            '<div class="banner-head"><div class="section-header">Insert Player & Stats</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="field-label">Player name</div>', unsafe_allow_html=True)
        name_l, name_r = st.columns([6, 6])
        with name_l:
            hitter_name = st.text_input("Hitter name", placeholder="e.g Jaxon Reeves", label_visibility="collapsed")
        with name_r:
            if "player_type" not in st.session_state:
                st.session_state["player_type"] = "Hitter"

            current_type = st.session_state.get("player_type", "Hitter")

            st.markdown(
                '<div class="player-type-toggle">',
                unsafe_allow_html=True,
            )

            t1, t2 = st.columns(2)
            with t1:
                st.markdown(
                    '<div class="' + ("selected" if current_type == "Hitter" else "") + '">',
                    unsafe_allow_html=True,
                )
                if st.button("Hitter", key="player_type_btn_hitter"):
                    st.session_state["player_type"] = "Hitter"
                st.markdown("</div>", unsafe_allow_html=True)
            with t2:
                st.markdown(
                    '<div class="' + ("selected" if current_type == "Pitcher" else "") + '">',
                    unsafe_allow_html=True,
                )
                if st.button("Pitcher", key="player_type_btn_pitcher"):
                    st.session_state["player_type"] = "Pitcher"
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            player_type = st.session_state.get("player_type", "Hitter")

        active_confs = conferences_batting if player_type == "Hitter" else conferences_pitching
        st.markdown('<div class="field-label">Paste season statline</div>', unsafe_allow_html=True)

        if player_type == "Hitter":
            placeholder = (
                "Paste Baseball Cube batting stats in this order:\n"
                + " ".join(PASTE_FIELDS)
                + "\n\n"
                + "Example:\n"
                + "231 73 74 10 0 21 60 1 0 36 36 25 7 0 0 6 147 299 31 43 .320 .462 .636 1.099 .476 .316 .305 12.04 12.04 1.00 11.00"
            )
        else:
            placeholder = (
                "Paste Baseball Cube pitching stats in this order:\n"
                + " ".join(PITCHER_PASTE_FIELDS)
                + "\n\n"
                + "Example:\n"
                + "3.45 18 18 0 0 0 0 0 104.1 89 45 40 8 32 112 2 0 6 1.16 7.68 0.69 2.76 9.66 3.88 3.50"
            )
        pasted = st.text_area("Paste stats", height=150, placeholder=placeholder)
        toggle_l, toggle_m, toggle_r = st.columns([6, 2, 6])
        with toggle_l:
            st.markdown(
                '<div class="toggle-row" style="justify-content:flex-start;"><div class="toggle-label">Compare Within Conference</div></div>',
                unsafe_allow_html=True,
            )
        with toggle_m:
            within_conf = st.toggle("Compare Within Conference", value=False, label_visibility="collapsed")
        with toggle_r:
            conf_choice = st.selectbox(
                "Conference",
                options=active_confs,
                index=0 if active_confs else None,
                disabled=not within_conf,
                label_visibility="collapsed",
                placeholder="Select conference" if hasattr(st, "selectbox") else None,
            )

        find = st.button("Find similar seasons", type="primary")

        paste_map = None
        if find:
            try:
                active_paste_fields = PASTE_FIELDS if player_type == "Hitter" else PITCHER_PASTE_FIELDS
                paste_map = parse_pasted_line(pasted, fields=active_paste_fields)
                st.session_state["last_input"] = paste_map
            except Exception as e:
                st.error(str(e))

    active_type = st.session_state.get("player_type", "Hitter")
    active_df = batting_df if active_type == "Hitter" else pitching_df
    active_sim_fields_display = SIM_FIELDS_DISPLAY if active_type == "Hitter" else PITCHER_SIM_FIELDS_DISPLAY
    active_display_to_paste = DISPLAY_TO_PASTE if active_type == "Hitter" else PITCHER_DISPLAY_TO_PASTE
    active_display_to_csv = DISPLAY_TO_CSV if active_type == "Hitter" else PITCHER_DISPLAY_TO_CSV
    active_sim_fields_csv = SIM_FIELDS_CSV if active_type == "Hitter" else PITCHER_SIM_FIELDS_CSV
    stds = compute_stds(active_df, tuple(active_sim_fields_csv))

    if find and paste_map:
        input_vec = build_input_vector(
            paste_map,
            sim_fields_display=active_sim_fields_display,
            display_to_paste=active_display_to_paste,
        )

        selected_conf = None
        if within_conf:
            selected_conf = (conf_choice or "").strip() if "conf_choice" in locals() else None

        matches: list[Match] = []
        for i, row in active_df.iterrows():
            if within_conf and selected_conf:
                if str(row.get("leaguename", "")).strip() != selected_conf:
                    continue
            vec = build_row_vector(
                row,
                sim_fields_display=active_sim_fields_display,
                display_to_csv=active_display_to_csv,
            )
            if vec is None:
                continue
            d = standardized_distance(input_vec, vec, stds)
            s = score_from_distance(d)
            name = f"{row.get('firstName','')} {row.get('lastName','')}".strip() or "Unknown"
            matches.append(
                Match(
                    idx=i,
                    name=name,
                    year=str(row.get("year", "")),
                    team=str(row.get("teamname", "")),
                    league=str(row.get("leaguename", "")),
                    distance=d,
                    score=s,
                )
            )

        matches.sort(key=lambda m: m.distance)
        st.session_state["top_matches"] = matches[:5]

    top = st.session_state.get("top_matches", [])

    st.markdown('<div id="perf-anchor"></div>', unsafe_allow_html=True)
    perf_block = st.container()
    with perf_block:
        if top:
            rows_html = ""
            for i, m in enumerate(top):
                row = active_df.loc[m.idx]
                position = str(row.get("draftPosit", "")).strip()
                school = str(row.get("teamname", m.team)).strip()
                draft_round = _fmt_num(row.get("draftRound", math.nan))
                pick = _fmt_num(row.get("overall", math.nan))
                pct = max(0.0, min(1.0, m.score))
                pct_label = f"{pct * 100:.1f}%"
                tr_class = "top" if i == 0 else ""

                rows_html += (
                    f'<tr class="{tr_class}">'
                    f'<td><div class="perf-name">{m.name}</div></td>'
                    f'<td class="perf-num">{position or "—"}</td>'
                    f'<td class="perf-num">{school or "—"}</td>'
                    f'<td class="perf-num">{m.year}</td>'
                    f'<td class="perf-num">{draft_round}</td>'
                    f'<td class="perf-num">{pick}</td>'
                    f'<td class="perf-pct">{pct_label}</td>'
                    f'</tr>'
                )

            st.markdown(
                f"""
                <div class="banner-head">
                  <div class="section-header">Strongest Comparable Performances</div>
                </div>
                <div class="perf-table-wrap">
                  <table class="perf">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th class="num">Position</th>
                        <th class="num">School</th>
                        <th class="num">Year</th>
                        <th class="num">Round</th>
                        <th class="num">Pick</th>
                        <th class="pct">%-similar</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows_html}
                    </tbody>
                  </table>
                </div>
                <div class="perf-foot"><span class="chip chip-foot">2015–2025 NCAA D1</span></div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="banner-head">
                  <div class="section-header">Strongest Comparable Performances</div>
                </div>
                <div class="perf-table-wrap"><div class="muted" style="font-size: 14px;">Paste a stat line and click <b>Find similar seasons</b> to see comparable seasons.</div></div>
                <div class="perf-foot"><span class="chip chip-foot">2015–2025 NCAA D1</span></div>
                """,
                unsafe_allow_html=True,
            )

    

    # Statline comparison card (YOU vs selected dropdown)
    top = st.session_state.get("top_matches", [])
    paste_map = st.session_state.get("last_input")

    if paste_map and top:
        you_label = (hitter_name or "Roch").strip() or "Roch"
        options = list(range(len(top)))
        default_idx = 0

        st.markdown('<div id="comp-anchor"></div>', unsafe_allow_html=True)
        comp_block = st.container()
        with comp_block:
            st.markdown('<div id="comp-inline-anchor"></div>', unsafe_allow_html=True)
            inline_block = st.container()
            with inline_block:
                st.markdown(
                    f'<div class="lhs"><span class="section-header">Statline Comparison:&nbsp;</span><span class="section-header">{you_label}</span> <span class="vs">vs. </span></div>',
                    unsafe_allow_html=True,
                )

                chosen_idx = st.selectbox(
                    "Comparable season",
                    options=options,
                    index=default_idx,
                    format_func=lambda i: top[i].name,
                    label_visibility="collapsed",
                )

            chosen = top[int(chosen_idx)]
            chosen_row = active_df.loc[chosen.idx]

            cols = ["Player"] + active_sim_fields_display
            you_vals = [you_label] + [
                _fmt_num(paste_map.get(active_display_to_paste[f], math.nan), f) for f in active_sim_fields_display
            ]
            top_vals = [chosen.name] + [
                _fmt_num(chosen_row.get(active_display_to_csv[f], math.nan), f) for f in active_sim_fields_display
            ]

            ths = ''.join([f'<th class="player">{cols[0]}</th>'] + [f'<th>{c}</th>' for c in cols[1:]])
            you_tds = ''.join([f'<td class="player">{you_vals[0]}</td>'] + [f'<td>{v}</td>' for v in you_vals[1:]])
            top_tds = ''.join([f'<td class="player">{top_vals[0]}</td>'] + [f'<td>{v}</td>' for v in top_vals[1:]])

            st.markdown(
                f"""
                <div class="table-wrap">
                  <table class="comp">
                    <thead><tr>{ths}</tr></thead>
                    <tbody>
                      <tr class="you">{you_tds}</tr>
                      <tr>{top_tds}</tr>
                    </tbody>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
