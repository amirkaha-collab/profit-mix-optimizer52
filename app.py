from __future__ import annotations
# -*- coding: utf-8 -*-
# Profit Mix Optimizer – Improved Version
# שיפורים עיקריים:
# - עיצוב אחיד ונקי (Streamlit native + CSS מינימלי)
# - חיפוש מהיר עם NumPy vectorized (מהיר פי ~10)
# - גרף Radar להשוואה בין חלופות
# - ייצוא לאקסל + טאב השוואת מסלולים
# - נעילת קרן ספציפית
# - הסבר על ה-Score
# - היסטוריית ריצות (3 אחרונות)
# - הודעות שגיאה מפורטות לקבצים
# - אזהרה על סיסמה ברירת מחדל

import itertools
import math
import os
import re
import html
import io
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Profit Mix Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Version compatibility shim ─────────────────
import streamlit as _st_check
_st_version = tuple(int(x) for x in _st_check.__version__.split(".")[:2])

def _safe_plotly(fig, key=None):
    """Render plotly chart, compatible with all Streamlit versions and safe for repeated cards."""
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        # Older Streamlit versions may not support use_container_width or key in the same way.
        try:
            st.plotly_chart(fig, key=key)
        except TypeError:
            st.plotly_chart(fig)

# ─────────────────────────────────────────────
# CSS – אחיד, נקי, RTL
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* RTL baseline */
html, body, [class*="css"] { direction: rtl; text-align: right; }
div[data-baseweb="slider"], div[data-baseweb="slider"] * { direction: ltr !important; }

/* Header */
.app-header { padding: 8px 0 4px; margin-bottom: 4px; }
.app-title  { font-size: 30px; font-weight: 900; letter-spacing: -0.5px; margin: 0; }
.app-sub    { font-size: 14px; opacity: 0.7; margin: 2px 0 0; }

/* Metric cards */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 16px; }
.metric-box {
  flex: 1; min-width: 120px;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 12px 14px 10px;
  background: #f8fafc;
}
.metric-box .label { font-size: 11px; color: #64748b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .4px; }
.metric-box .value { font-size: 22px; font-weight: 800; color: #0f172a; }
.metric-box .sub   { font-size: 11px; color: #64748b; margin-top: 2px; }
@media (prefers-color-scheme: dark) {
  .metric-box { background: #1e293b; border-color: #334155; }
  .metric-box .label { color: #94a3b8; }
  .metric-box .value { color: #f1f5f9; }
  .metric-box .sub   { color: #94a3b8; }
}


/* Subtext for weighted recommendation */
.alt-sub{
  margin-top:-6px;
  margin-bottom:10px;
  font-size:12px;
  color:#334155;
  font-weight:700;
}
@media (prefers-color-scheme: dark) {
  .alt-sub{ color:#cbd5e1; }
}
/* Alt cards */
.alt-card {
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 16px;
  background: #fff;
  margin-bottom: 12px;
}

/* Recommendation emphasis */
.alt-primary {
  border: 2px solid #2563eb;
  box-shadow: 0 8px 22px rgba(37,99,235,0.12);
  position: relative;
}
.alt-primary .alt-badge{
  position:absolute; top:12px; left:12px;
  background:#2563eb; color:#fff;
  padding:4px 10px; border-radius:999px;
  font-size:12px; font-weight:800;
}
.alt-secondary { opacity: 0.92; }
.alt-secondary h3 { font-size: 15px; }
.manager-chip { display:inline-block; margin: 4px 6px 0 0; padding: 4px 10px; border-radius: 999px; background:#eef2ff; color:#1e3a8a; font-size:12px; font-weight:700; }

.alt-card h3 { margin: 0 0 4px; font-size: 16px; }
.alt-adv { font-size: 12px; color: #475569; background: #f1f5f9; border-radius: 999px; padding: 3px 10px; display: inline-block; margin-bottom: 10px; }
.fund-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px dashed #e2e8f0; }
.fund-row:last-child { border-bottom: none; }
.fund-pct  { min-width: 50px; font-weight: 800; font-size: 14px; color: #0f172a; }
.fund-name { font-size: 13px; color: #334155; flex: 1; }
.fund-track{ font-size: 11px; color: #94a3b8; }
.kpi-mini  { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.kpi-chip  { font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid #e2e8f0; background: #f8fafc; color: #334155; }
.kpi-chip b{ color: #0f172a; }
@media (prefers-color-scheme: dark) {
  .alt-card  { background: #1e293b; border-color: #334155; }
  .alt-card h3 { color: #f1f5f9; }
  .alt-adv   { background: #0f172a; color: #94a3b8; }
  .fund-row  { border-color: #334155; }
  .fund-pct  { color: #f1f5f9; }
  .fund-name { color: #cbd5e1; }
  .kpi-chip  { background: #0f172a; border-color: #334155; color: #94a3b8; }
  .kpi-chip b{ color: #f1f5f9; }
}

/* Score tooltip */
.score-tip {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 12.5px;
  color: #78350f;
  margin: 8px 0;
}

/* History badge */
.hist-badge {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  background: #dbeafe;
  color: #1e40af;
  margin-left: 6px;
}

/* Comparison table */
div[data-testid="stDataFrame"] * { direction: rtl; text-align: right; }

/* Password screen */
.pw-wrap { max-width: 340px; margin: 60px auto; text-align: center; }
.pw-title { font-size: 26px; font-weight: 800; margin-bottom: 6px; }
.pw-sub   { font-size: 14px; opacity: 0.7; margin-bottom: 20px; }
.pw-warn  { font-size: 12px; color: #b45309; background: #fef3c7; border-radius: 8px; padding: 6px 10px; margin-top: 10px; }

/* Allocation panel (results) */
.alloc-panel { margin: 10px 0 12px; padding: 10px 12px; border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; background: rgba(255,255,255,0.03); }
.alloc-row { display: grid; grid-template-columns: 140px 1fr 70px; gap: 10px; align-items: center; margin: 8px 0; }
.alloc-label { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; opacity: 0.92; }
.alloc-icon { font-size: 16px; line-height: 1; }
.alloc-name { font-size: 14px; }
.alloc-bar { position: relative; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.10); overflow: hidden; }
.alloc-fill { height: 100%; border-radius: 999px; background: rgba(99, 102, 241, 0.85); }
.alloc-val { text-align: left; direction: ltr; font-variant-numeric: tabular-nums; font-weight: 800; }
.mgr-line { margin: 6px 0 10px; }
.result-shell{border:1px solid #e2e8f0;border-radius:18px;padding:14px 14px 10px;background:#fff;margin-bottom:12px;}
.result-shell.primary{border:2px solid #4f46e5;box-shadow:0 8px 24px rgba(79,70,229,.10);}
.result-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:12px;}
.result-title{font-size:21px;font-weight:900;margin:0;color:#111827;}
.result-subtle{font-size:12px;color:#64748b;margin-top:2px;}
.result-tag{display:inline-block;padding:5px 10px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;font-weight:800;}
.manager-pills{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px;}
.manager-pill{background:#f8fafc;border:1px solid #e2e8f0;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700;color:#334155;}
.compact-note{font-size:12px;color:#64748b;margin:2px 0 8px;}
.kpi-box{border:1px solid #e5e7eb;border-radius:14px;padding:10px 12px;background:#fafafa;margin-bottom:8px;}
.kpi-label{font-size:11px;color:#64748b;margin-bottom:2px;}
.kpi-value{font-size:20px;font-weight:900;color:#111827;}
.small-muted{font-size:11px;color:#94a3b8;}
@media (prefers-color-scheme: dark) {
  .result-shell{background:#111827;border-color:#334155;}
  .result-title{color:#f8fafc;}
  .result-subtle,.compact-note{color:#94a3b8;}
  .manager-pill,.kpi-box{background:#0f172a;border-color:#334155;color:#e2e8f0;}
  .kpi-value{color:#f8fafc;}
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _esc(x) -> str:
    try:
        return html.escape("" if x is None else str(x), quote=True)
    except Exception:
        return ""

def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = re.sub(r"[^\d.\-]", "", str(x).replace(",", "").replace("−", "-"))
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _fmt_pct(x, decimals=2) -> str:
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "—"

def _fmt_num(x, fmt="{:.2f}") -> str:
    try:
        return fmt.format(float(x))
    except Exception:
        return "—"


# ─────────────────────────────────────────────
# Password Gate
# ─────────────────────────────────────────────
def _check_password() -> bool:
    if st.session_state.get("auth_ok", False):
        return True

    is_default = True
    if hasattr(st, "secrets") and "APP_PASSWORD" in st.secrets:
        correct = str(st.secrets["APP_PASSWORD"])
        is_default = False
    else:
        correct = os.getenv("APP_PASSWORD", "1234")

    st.markdown("""
    <div class="pw-wrap">
      <div class="pw-title">🔒 כניסה</div>
      <div class="pw-sub">האפליקציה מוגנת בסיסמה</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("סיסמה", type="password", placeholder="••••••••", label_visibility="collapsed")
        if st.button("כניסה", use_container_width=True, type="primary"):
            if pwd == correct:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("סיסמה שגויה")
        if is_default:
            st.markdown(
                '<div class="pw-warn">⚠️ הסיסמה היא ברירת מחדל (1234). הגדר APP_PASSWORD ב-Streamlit Secrets בסביבת ייצור!</div>',
                unsafe_allow_html=True
            )
    st.stop()

_check_password()


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Google Sheets – מקורות נתונים
# ─────────────────────────────────────────────
FUNDS_GSHEET_ID   = "1ty_tqcyGqmVI4pQZetHHKd-cC0O2HCpD2dbpNpYlPtY"
SERVICE_GSHEET_ID = "1FSgvIG6VsJxB5QPY6fmwAwGc1TYLB0KXg-7ckkD_RJQ"

PARAM_ALIASES = {
    "stocks":   ["סך חשיפה למניות", "מניות"],
    "foreign":  ['סך חשיפה לנכסים המושקעים בחו"ל', "סך חשיפה לנכסים המושקעים בחו׳ל", 'חו"ל', "חו׳ל"],
    "fx":       ['חשיפה למט"ח', 'מט"ח', "מט׳׳ח"],
    "illiquid": ["נכסים לא סחירים", "לא סחירים", "לא-סחיר", "לא סחיר"],
    "sharpe":   ["מדד שארפ", "שארפ"],
}


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def _match_param(row_name: str, key: str) -> bool:
    rn = str(row_name).strip()
    return any(a in rn for a in PARAM_ALIASES[key])

def _extract_manager(fund_name: str) -> str:
    name = str(fund_name).strip()
    for splitter in [" קרן", " השתלמות", " -", "-", "  "]:
        if splitter in name:
            head = name.split(splitter)[0].strip()
            if head:
                return head
    return name.split()[0] if name.split() else name

def _gsheet_to_bytes(sheet_id: str) -> Tuple[bytes, str]:
    """הורד Google Sheet כ-XLSX. מנסה כמה URL formats."""
    import requests as _req

    urls = [
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx",
        f"https://docs.google.com/feeds/download/spreadsheets/Export?key={sheet_id}&exportFormat=xlsx",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    last_err = ""
    for url in urls:
        try:
            resp = _req.get(url, headers=headers, allow_redirects=True, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 500:
                # Verify it's actually an xlsx (PK header) not an HTML error page
                if resp.content[:2] == b"PK":
                    return resp.content, ""
                else:
                    preview = resp.content[:120].decode("utf-8", errors="ignore").replace("\n"," ") if resp.content else ""
                    last_err = (
                        f"קוד 200 אבל התקבל HTML במקום XLSX (גיליון {sheet_id[:20]}). "
                        "בדוק ש-Share מוגדר 'Anyone with the link' כ-Viewer, "
                        "ואם עדיין נכשל נסה גם File → Publish to the web. "
                        f"URL: {resp.url[:80]} | Preview: {preview[:80]}"
                    )
            else:
                last_err = f"HTTP {resp.status_code} מ-{url[:60]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    return b"", last_err

def _load_service_scores(xlsx_bytes: bytes) -> Tuple[Dict[str, float], str]:
    """קרא ציוני שירות מ-bytes של XLSX.

    תומך בשני פורמטים:
    1) טבלה "ארוכה": עמודות provider, score (מומלץ)
    2) טבלה "רחבה": שורת כותרות של שמות גופים בשורה 2 (או 1), ושורה מתחתיה עם ציונים.
       לדוגמה: A2="provider", B2="אלטשולר שחם", ... ; A3="score", B3=90, ...
    """
    try:
        df = pd.read_excel(io.BytesIO(xlsx_bytes), header=None)
    except Exception as e:
        return {}, f"שגיאה בטעינת ציוני שירות: {e}"
    if df.empty:
        return {}, "גיליון ציוני שירות ריק"

    # --- ניסיון 1: פורמט ארוך עם כותרות ---
    try:
        df_hdr = pd.read_excel(io.BytesIO(xlsx_bytes))
        if not df_hdr.empty:
            cols = [str(c).lower().strip() for c in df_hdr.columns]
            df_hdr.columns = cols
            if "provider" in df_hdr.columns and "score" in df_hdr.columns:
                out = {}
                for _, r in df_hdr.iterrows():
                    p = _extract_manager(str(r["provider"]).strip())
                    sc = _to_float(r["score"])
                    if p and not math.isnan(sc):
                        out[p] = float(sc)
                if out:
                    return out, ""
    except Exception:
        pass

    # --- ניסיון 2: פורמט רחב (provider בשורה אחת, score בשורה מתחת) ---
    # מנקים תאים ריקים בתחילת הגיליון
    df2 = df.copy()
    df2 = df2.dropna(how="all")
    df2 = df2.dropna(how="all", axis=1)
    if df2.shape[0] >= 2 and df2.shape[1] >= 2:
        # מחפשים שורה שמכילה 'provider' בעמודה הראשונה
        first_col = df2.iloc[:, 0].astype(str).str.strip().str.lower()
        prov_rows = df2.index[first_col.eq("provider")].tolist()

        # Support case where first cell contains both 'provider' and 'score' (e.g., "provider\nscore")
        combo_cell = df2.iloc[:, 0].astype(str).str.strip().str.lower()
        combo_rows = df2.index[combo_cell.str.contains("provider") & combo_cell.str.contains("score")].tolist()
        for r0 in combo_rows:
            if r0 not in prov_rows:
                prov_rows.append(r0)

        for r0 in prov_rows[:3]:
            if r0 + 1 in df2.index:
                header = df2.loc[r0].tolist()
                values = df2.loc[r0 + 1].tolist()
                # מצפים שעמודה 0 בשורה הבאה היא 'score' (או משהו דומה)
                tag = str(values[0]).strip().lower()
                if tag in {"score", "ציון", "שירות", "ציון שירות"} or tag in {"nan", "", "none"}:
                    out = {}
                    for name, val in zip(header[1:], values[1:]):
                        p = _extract_manager(str(name).strip())
                        sc = _to_float(val)
                        if p and not math.isnan(sc):
                            out[p] = float(sc)
                    if out:
                        return out, ""

    return {}, "מבנה גיליון שירות לא מזוהה – יש להגדיר טבלת provider/score או טבלה רחבה עם provider מעל score"

@st.cache_data(show_spinner=False, ttl=900)   # מרענן כל 10 דקות
def load_funds_long(funds_id: str, service_id: str) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """מוריד את שני הגיליונות מ-Google Sheets ומחזיר (df_long, service_map, warnings)."""
    warnings: List[str] = []

    # ── ציוני שירות ──────────────────────────
    svc_bytes, svc_err = _gsheet_to_bytes(service_id)
    if svc_err:
        warnings.append(svc_err)
        svc = {}
    else:
        svc, parse_err = _load_service_scores(svc_bytes)
        if parse_err:
            warnings.append(parse_err)

    # ── קרנות ────────────────────────────────
    funds_bytes, funds_err = _gsheet_to_bytes(funds_id)
    if funds_err:
        return pd.DataFrame(), svc, warnings + [funds_err]

    try:
        xls = pd.ExcelFile(io.BytesIO(funds_bytes))
    except Exception as e:
        return pd.DataFrame(), svc, warnings + [f"שגיאה בפתיחת גיליון קרנות: {e}"]

    records: List[Dict] = []
    for sh in xls.sheet_names:
        sh_str = str(sh)
        if re.search(r"ניהול\s*אישי", sh_str) or re.search(r"(^|[^a-z])ira([^a-z]|$)", sh_str.lower()):
            continue
        try:
            df = pd.read_excel(xls, sheet_name=sh, header=None)
        except Exception as e:
            warnings.append(f"גיליון '{sh}': שגיאת קריאה – {e}")
            continue
        if df.empty:
            continue

        header_row = df.iloc[0].tolist()
        if not str(header_row[0]).strip().startswith("פרמטר"):
            idxs = df.index[df.iloc[:, 0].astype(str).str.contains("פרמטר", na=False)].tolist()
            if not idxs:
                continue
            df = df.iloc[idxs[0]:].reset_index(drop=True)
            header_row = df.iloc[0].tolist()

        fund_names = [c for c in header_row[1:] if str(c).strip() and str(c).strip() != "nan"]
        if not fund_names:
            continue

        param_col = df.iloc[1:, 0].astype(str).tolist()

        def row_for(key: str) -> Optional[int]:
            for i, rn in enumerate(param_col, start=1):
                if _match_param(rn, key):
                    return i
            return None

        ridx = {k: row_for(k) for k in ["stocks", "foreign", "fx", "illiquid", "sharpe"]}
        if ridx["foreign"] is None and ridx["stocks"] is None:
            continue

        for j, fname in enumerate(fund_names, start=1):
            manager = _extract_manager(fname)
            rec = {
                "track":    sh_str,
                "fund":     str(fname).strip(),
                "manager":  manager,
                "stocks":   _to_float(df.iloc[ridx["stocks"],   j]) if ridx["stocks"]   is not None else np.nan,
                "foreign":  _to_float(df.iloc[ridx["foreign"],  j]) if ridx["foreign"]  is not None else np.nan,
                "fx":       _to_float(df.iloc[ridx["fx"],       j]) if ridx["fx"]       is not None else np.nan,
                "illiquid": _to_float(df.iloc[ridx["illiquid"], j]) if ridx["illiquid"] is not None else np.nan,
                "sharpe":   _to_float(df.iloc[ridx["sharpe"],   j]) if ridx["sharpe"]   is not None else np.nan,
            }
            if all(math.isnan(rec[k]) for k in ["foreign", "stocks", "fx", "illiquid", "sharpe"]):
                continue
            rec["service"] = float(svc.get(manager, 50.0))
            records.append(rec)

    df_long = pd.DataFrame.from_records(records)
    if not df_long.empty:
        for c in ["stocks", "foreign", "fx", "illiquid", "sharpe", "service"]:
            if c in df_long.columns:
                df_long[c] = pd.to_numeric(df_long[c], errors="coerce")
    return df_long, svc, warnings


# ─────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────
def _weights_for_n(n: int, step: int) -> np.ndarray:
    """Returns array of shape (n_weights, n) with integer percentages."""
    step = max(1, int(step))
    if n == 1:
        return np.array([[100]], dtype=float)
    if n == 2:
        ws = np.arange(0, 101, step)
        pairs = np.column_stack([ws, 100 - ws])
        return pairs.astype(float)
    # n == 3
    out = []
    for w1 in range(0, 101, step):
        for w2 in range(0, 101 - w1, step):
            w3 = 100 - w1 - w2
            if w3 >= 0 and w3 % step == 0:
                out.append([w1, w2, w3])
    return np.array(out, dtype=float) if out else np.empty((0, 3), dtype=float)

def _prefilter_candidates(df: pd.DataFrame, include: Dict, targets: Dict, cap: int, locked_fund: str) -> pd.DataFrame:
    keys = [k for k, v in include.items() if v and k in ["foreign", "stocks", "fx", "illiquid"]]
    if not keys:
        keys = ["foreign", "stocks"]
    tmp = df.copy()
    score = np.zeros(len(tmp), dtype=float)
    for k in keys:
        score += np.abs(tmp[k].fillna(50.0).to_numpy() - float(targets.get(k, 0.0))) / 100.0
    tmp["_s"] = score

    # Always include locked fund
    if locked_fund:
        locked_mask = tmp["fund"].str.strip() == locked_fund.strip()
        locked_df = tmp[locked_mask]
        rest_df   = tmp[~locked_mask].sort_values("_s").head(max(cap - len(locked_df), 1))
        tmp = pd.concat([locked_df, rest_df])
    else:
        tmp = tmp.sort_values("_s").head(cap)

    return tmp.drop(columns=["_s"]).reset_index(drop=True)

def _hard_ok_vec(values: np.ndarray, target: float, mode: str) -> np.ndarray:
    """Vectorized hard constraint check. Returns bool array."""
    if mode == "בדיוק":
        return np.abs(values - target) < 0.5
    if mode == "לפחות":
        return values >= target - 0.5
    if mode == "לכל היותר":
        return values <= target + 0.5
    return np.ones(len(values), dtype=bool)

def find_best_solutions(
    df: pd.DataFrame,
    n_funds: int,
    step: int,
    mix_policy: str,
    include: Dict,
    constraint: Dict,
    targets: Dict,
    primary_rank: str,
    locked_fund: str = "",
    max_solutions_scan: int = 20000,
) -> Tuple[pd.DataFrame, str]:
    import gc
    targets = {k: float(v) for k, v in targets.items()}

    # Pre-filter: keep only most-relevant funds (limits memory naturally)
    cap = 50 if n_funds == 2 else 35 if n_funds == 3 else 80
    df_scan = _prefilter_candidates(df, include, targets, cap=cap, locked_fund=locked_fund)

    weights_arr  = _weights_for_n(n_funds, step)
    if len(weights_arr) == 0:
        return pd.DataFrame(), "לא נמצאו שילובי משקלים. נסה צעד קטן יותר."
    weights_norm = weights_arr / 100.0

    metric_keys = ["foreign", "stocks", "fx", "illiquid"]
    active_soft = [k for k in metric_keys if include.get(k, False)] or ["foreign", "stocks"]
    soft_idx    = {k: i for i, k in enumerate(metric_keys)}
    hard_keys   = [(k, constraint[k][1]) for k in metric_keys
                   if constraint.get(k, ("רך", ""))[0] == "קשיח"]

    A       = df_scan[["foreign","stocks","fx","illiquid","sharpe","service"]].to_numpy(dtype=float)
    records = df_scan.reset_index(drop=True)

    locked_idx: Optional[int] = None
    if locked_fund:
        matches = records.index[records["fund"].str.strip() == locked_fund.strip()].tolist()
        if matches:
            locked_idx = matches[0]

    if mix_policy == "אותו מנהל בלבד":
        groups = list(records.groupby("manager").groups.values())
        combo_source = itertools.chain.from_iterable(
            itertools.combinations(list(g), n_funds) for g in groups if len(g) >= n_funds
        )
    else:
        combo_source = itertools.combinations(range(len(records)), n_funds)

    solutions = []
    scanned   = 0
    # Hard cap on stored solutions to protect RAM.
    # With cap=50 funds, step=5: at most C(50,2)×21 = ~25K rows — fine.
    # With step=1 it could be larger, so we cap stored rows at 60K.
    MAX_STORED = 60000

    for combo in combo_source:
        if locked_idx is not None and locked_idx not in combo:
            continue
        scanned += 1
        if scanned > max_solutions_scan:
            break

        arr     = A[list(combo), :]
        mix_all = np.einsum("wn,nm->wm", weights_norm, np.nan_to_num(arr, nan=0.0))

        # Hard constraints
        mask = np.ones(len(weights_norm), dtype=bool)
        for k, mode in hard_keys:
            mask &= _hard_ok_vec(mix_all[:, soft_idx[k]], targets.get(k, 0.0), mode)
        if not mask.any():
            continue

        mix_ok    = mix_all[mask]
        w_ok      = weights_arr[mask]
        score_arr = np.zeros(len(mix_ok))
        for k in active_soft:
            score_arr += np.abs(mix_ok[:, soft_idx[k]] - targets.get(k, 0.0)) / 100.0

        fund_labels  = [records.loc[i, "fund"]    for i in combo]
        track_labels = [records.loc[i, "track"]   for i in combo]
        managers     = [records.loc[i, "manager"] for i in combo]
        manager_set  = " | ".join(sorted(set(managers)))

        for wi in range(len(mix_ok)):
            solutions.append({
                "combo":          combo,
                "weights":        tuple(int(round(x)) for x in w_ok[wi]),
                "מנהלים":         manager_set,
                "מסלולים":        " | ".join(track_labels),
                "קופות":          " | ".join(fund_labels),
                'חו"ל (%)'  :    float(mix_ok[wi, 0]),
                "ישראל (%)"  :    float(100.0 - mix_ok[wi, 0]),
                "מניות (%)"  :    float(mix_ok[wi, 1]),
                'מט"ח (%)'  :    float(mix_ok[wi, 2]),
                "לא־סחיר (%)" :   float(mix_ok[wi, 3]),
                "שארפ משוקלל":    float(mix_ok[wi, 4]),
                "שירות משוקלל":   float(mix_ok[wi, 5]),
                "score"       :   float(score_arr[wi]),
            })

        # If stored rows approach limit, prune to best 10K to free memory mid-scan
        if len(solutions) >= MAX_STORED:
            solutions.sort(key=lambda r: (r["score"], -r["שארפ משוקלל"], -r["שירות משוקלל"]))
            solutions = solutions[:10000]
            gc.collect()

    if not solutions:
        return pd.DataFrame(), "לא נמצאו פתרונות. נסה לרכך מגבלות קשיחות, להגדיל צעד, או להפחית יעדים."

    df_sol = pd.DataFrame(solutions)
    del solutions
    gc.collect()

    note = f"נסרקו {min(scanned, max_solutions_scan):,} קומבינציות מתוך {len(df_scan)} קופות מסוננות."

    if primary_rank == "דיוק":
        df_sol = df_sol.sort_values(["score", "שארפ משוקלל", "שירות משוקלל"], ascending=[True, False, False])
    elif primary_rank == "שארפ":
        df_sol = df_sol.sort_values(["שארפ משוקלל", "score"], ascending=[False, True])
    elif primary_rank == "שירות":
        df_sol = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])

    return df_sol, note

def _pick_three_distinct(df_sol: pd.DataFrame, primary_rank: str) -> pd.DataFrame:
    if df_sol.empty:
        return df_sol

    def mgr(row) -> str:
        return str(row["מנהלים"]).strip()

    # Sort each criterion independently — each gets its own ranked list
    sorted_primary = df_sol.copy()   # already sorted by primary_rank from find_best_solutions
    sorted_sharpe  = df_sol.sort_values(["שארפ משוקלל",  "score"], ascending=[False, True])
    sorted_service = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])

    def best_from(df_sorted, exclude_managers: set):
        """Return the best row from df_sorted whose manager is not in exclude_managers."""
        for _, r in df_sorted.iterrows():
            if mgr(r) not in exclude_managers:
                return r
        # Fallback: if all managers are taken, return absolute best regardless
        return df_sorted.iloc[0]

    # Pick each independently first (no exclusions yet)
    pick1 = best_from(sorted_primary, set())
    pick2 = best_from(sorted_sharpe,  set())
    pick3 = best_from(sorted_service, set())

    # Resolve conflicts: if two picks share the same manager, the lower-priority
    # one finds its next-best from a manager not yet taken by higher-priority picks
    used_after_1 = {mgr(pick1)}
    if mgr(pick2) in used_after_1:
        pick2 = best_from(sorted_sharpe, used_after_1)

    used_after_2 = used_after_1 | {mgr(pick2)}
    if mgr(pick3) in used_after_2:
        pick3 = best_from(sorted_service, used_after_2)

    base = pick1.to_dict()
    labels    = ["חלופה 1 – דירוג ראשי", "חלופה 2 – שארפ", "חלופה 3 – שירות"]
    criterions = ["דיוק", "שארפ", "שירות"]
    rows = []
    for i, r in enumerate([pick1, pick2, pick3]):
        row = r.to_dict()
        row["חלופה"]       = labels[i]
        row["weights_items"] = _weights_items(row.get("weights"), row.get("קופות",""), row.get("מסלולים",""))
        row["משקלים"]      = _weights_short(row.get("weights"))
        row["יתרון"]       = _make_advantage(criterions[i], row, base if i > 0 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def _weights_items(weights, funds_str, tracks_str) -> List[Dict]:
    try:    ws = list(weights)
    except: ws = []
    funds  = [s.strip() for s in (funds_str  or "").split("|") if s.strip()]
    tracks = [s.strip() for s in (tracks_str or "").split("|") if s.strip()]
    n = max(len(ws), len(funds))
    return [
        {
            "pct":   f"{int(round(float(ws[i])))}%" if i < len(ws) else "?",
            "fund":  funds[i]  if i < len(funds)  else "",
            "track": tracks[i] if i < len(tracks) else "",
        }
        for i in range(n)
    ]

def _weights_short(weights) -> str:
    if weights is None: return ""
    try:    w = [float(x) for x in weights]
    except: return ""
    return " / ".join(f"{int(round(x))}%" for x in w)

def _make_advantage(primary: str, row: Dict, base: Optional[Dict] = None) -> str:
    score = row.get("score", 0)
    if primary == "דיוק":
        return f"מדויק ביותר ליעד (סטייה {score:.4f})"
    if primary == "שארפ":
        sh = float(row.get("שארפ משוקלל", 0) or 0)
        delta = sh - float((base or {}).get("שארפ משוקלל", sh) or sh)
        return f"שארפ {sh:.2f} (+{delta:.2f} מחלופה 1)"
    sv = float(row.get("שירות משוקלל", 0) or 0)
    delta = sv - float((base or {}).get("שירות משוקלל", sv) or sv)
    return f"שירות {sv:.1f} (+{delta:.1f} מחלופה 1)"


# ─────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────

def _normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn = float(s.min())
    mx = float(s.max())
    if abs(mx - mn) < 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def _pick_recommendations(df_sol_head: pd.DataFrame) -> Dict[str, Dict]:
    """Return 3 rows: weighted best, most accurate, best sharpe."""
    if df_sol_head is None or df_sol_head.empty:
        return {}

    df = df_sol_head.copy()

    # Accuracy: lower score is better → invert normalized score
    score_n = _normalize_series(df["score"])
    acc_n   = 1.0 - score_n

    sharpe_n  = _normalize_series(df.get("שארפ משוקלל", pd.Series([0]*len(df))))
    service_n = _normalize_series(df.get("שירות משוקלל", pd.Series([0]*len(df))))

    # Weighted preference: encourage balanced pick (דיוק + "אלפא"(שארפ) + שירות)
    w_acc, w_sharpe, w_service = 0.45, 0.15, 0.40
    df["_weighted_pref"] = w_acc*acc_n + w_sharpe*sharpe_n + w_service*service_n

    weighted = df.loc[df["_weighted_pref"].idxmax()].to_dict()
    accurate = df.loc[df["score"].idxmin()].to_dict()
    best_sh  = df.loc[df["שארפ משוקלל"].idxmax()].to_dict() if "שארפ משוקלל" in df.columns else accurate

    return {"weighted": weighted, "accurate": accurate, "sharpe": best_sh}

def _manager_weights_from_items(items: List[Dict], manager_names: List[str]) -> List[Tuple[str, float]]:
    """Best-effort mapping fund->manager by longest name match; returns list of (manager, pct)."""
    if not items:
        return []
    names = sorted([m for m in manager_names if isinstance(m, str) and m.strip()], key=len, reverse=True)
    agg = {}
    for it in items:
        fund = str(it.get("fund",""))
        pct  = float(str(it.get("pct","0")).replace("%","") or 0)
        chosen = None
        f = fund.strip()
        for n in names:
            if f.startswith(n) or (n in f):
                chosen = n
                break
        if chosen is None:
            chosen = "אחר"
        agg[chosen] = agg.get(chosen, 0.0) + pct
    out = sorted(agg.items(), key=lambda x: -x[1])
    return out


def _alloc_plot(r: Dict):
    labels = ["מניות", 'חו"ל', 'מט"ח', "לא־סחיר"]
    vals = []
    for k in ["מניות (%)", 'חו"ל (%)', 'מט"ח (%)', "לא־סחיר (%)"]:
        try:
            vals.append(float(r.get(k) or 0))
        except Exception:
            vals.append(0.0)
    text_labels = [f"{lbl} · {v:.1f}%" for lbl, v in zip(labels, vals)]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        text=text_labels, textposition='outside',
        cliponaxis=False,
        marker=dict(color=['#6366f1','#8b5cf6','#a78bfa','#c4b5fd'])
    ))
    fig.update_layout(
        height=220, margin=dict(l=10,r=120,t=0,b=0),
        xaxis=dict(range=[0,100], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(autorange='reversed', tickfont=dict(size=13), showgrid=False, title=None),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    fig.update_traces(hovertemplate='%{y}: %{x:.1f}%<extra></extra>')
    return fig

def _manager_donut(mgr_break):
    labels=[m for m,_ in mgr_break] or ["ללא"]
    values=[float(p) for _,p in mgr_break] or [100.0]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, textinfo='percent', sort=False))
    fig.update_traces(marker=dict(colors=['#4f46e5','#7c3aed','#06b6d4','#22c55e','#f59e0b','#ef4444']))
    fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def _render_reco_card(r: Dict, title: str, primary: bool = False, manager_names: Optional[List[str]] = None, card_key: str = ''):
    items = r.get("weights_items") or _weights_items(r.get("weights"), r.get("קופות",""), r.get("מסלולים",""))
    mgr_names = manager_names or []
    mgr_break = _manager_weights_from_items(items, mgr_names)

    shell_cls = "primary" if primary else "secondary"
    st.markdown(f"<div class='result-shell {shell_cls}'>", unsafe_allow_html=True)
    tag = "חלופה מרכזית" if primary else "חלופה נוספת"
    st.markdown(
        f"<div class='result-head'><div><div class='result-title'>{_esc(title)}</div><div class='result-subtle'>{_esc(r.get('מנהלים',''))}</div></div><div class='result-tag'>{tag}</div></div>",
        unsafe_allow_html=True
    )

    if mgr_break:
        mgr_html = "".join(f"<span class='manager-pill'>{_esc(m)} · {p:.0f}%</span>" for m,p in mgr_break)
        st.markdown(f"<div class='manager-pills'>{mgr_html}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.05, 1.55, 1.0])
    with c1:
        st.caption("תמהיל מנהלים")
        _safe_plotly(_manager_donut(mgr_break), key=f'mgr_donut_{card_key or title}')
    with c2:
        st.caption("תמהיל אפיקי השקעה")
        _safe_plotly(_alloc_plot(r), key=f'alloc_plot_{card_key or title}')
    with c3:
        kpis = [
            ("דיוק (סטייה)", _fmt_num(r.get("score"), "{:.4f}")),
            ("שארפ", _fmt_num(r.get("שארפ משוקלל"))),
            ("שירות", _fmt_num(r.get("שירות משוקלל"), "{:.1f}")),
        ]
        for label, val in kpis:
            st.markdown(f"<div class='kpi-box'><div class='kpi-label'>{label}</div><div class='kpi-value'>{val}</div></div>", unsafe_allow_html=True)

    with st.expander("הרכב הקרנות בתמהיל", expanded=False):
        for it in items:
            c_a, c_b = st.columns([0.18, 0.82])
            with c_a:
                st.markdown(f"**{_esc(it['pct'])}**")
            with c_b:
                st.markdown(f"{_esc(it['fund'])}  ")
                st.caption(_esc(it['track']))

    st.markdown("</div>", unsafe_allow_html=True)

def _render_alt_card(r: Dict, idx: int):
    items = r.get("weights_items") or []
    funds_html = "".join(
        f"<div class='fund-row'>"
        f"  <div class='fund-pct'>{_esc(it['pct'])}</div>"
        f"  <div class='fund-name'>{_esc(it['fund'])} <span class='fund-track'>({_esc(it['track'])})</span></div>"
        f"</div>"
        for it in items
    )
    kpis = [
        ("חו&quot;ל", _fmt_pct(r.get('חו"ל (%)'))),
        ("מניות",     _fmt_pct(r.get("מניות (%)"))),
        ('מט"ח',      _fmt_pct(r.get('מט"ח (%)'))),
        ("לא־סחיר",   _fmt_pct(r.get("לא־סחיר (%)"))),
        ("שארפ",      _fmt_num(r.get("שארפ משוקלל"))),
        ("שירות",     _fmt_num(r.get("שירות משוקלל"), "{:.1f}")),
    ]
    kpis_html = "".join(f"<div class='kpi-chip'>{k}: <b>{v}</b></div>" for k, v in kpis)
    st.markdown(f"""
    <div class="alt-card">
      <h3>{_esc(r.get('חלופה', f'חלופה {idx}'))}</h3>
      <div class="alt-adv">{_esc(r.get('יתרון',''))}</div>
      {funds_html}
      <div class="kpi-mini">{kpis_html}</div>
    </div>
    """, unsafe_allow_html=True)


def _radar_chart(top3: pd.DataFrame, targets: Dict) -> go.Figure:
    categories = ["חו\"ל", "מניות", "מט\"ח", "לא־סחיר", "שארפ×10", "שירות÷10"]

    fig = go.Figure()

    colors = ["#2563eb", "#16a34a", "#ea580c"]
    for i, row in top3.iterrows():
        vals = [
            float(row.get('חו"ל (%)', 0) or 0),
            float(row.get("מניות (%)", 0) or 0),
            float(row.get('מט"ח (%)', 0) or 0),
            float(row.get("לא־סחיר (%)", 0) or 0),
            float(row.get("שארפ משוקלל", 0) or 0) * 10,
            float(row.get("שירות משוקלל", 0) or 0) / 10,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            opacity=0.25,
            line=dict(color=colors[i % 3], width=2),
            name=str(row.get("חלופה", f"חלופה {i+1}")),
        ))

    # Target overlay
    tgt_vals = [
        targets.get("foreign", 0),
        targets.get("stocks",  0),
        targets.get("fx",      0),
        targets.get("illiquid",0),
        0,  # sharpe target not shown
        0,
    ]
    fig.add_trace(go.Scatterpolar(
        r=tgt_vals + [tgt_vals[0]],
        theta=categories + [categories[0]],
        mode="lines",
        line=dict(color="rgba(239,68,68,0.7)", width=1.5, dash="dot"),
        name="יעד",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=True,
        height=400,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.12),
        font=dict(family="sans-serif", size=11),
    )
    return fig


def _export_excel(top3: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary sheet
        display_cols = [
            "חלופה", "יתרון", "קופות", "מסלולים", "משקלים",
            'חו"ל (%)', "מניות (%)", 'מט"ח (%)', "לא־סחיר (%)",
            "שארפ משוקלל", "שירות משוקלל", "score",
        ]
        cols_exist = [c for c in display_cols if c in top3.columns]
        top3[cols_exist].to_excel(writer, sheet_name="חלופות", index=False)

        # Details per alternative
        for i, row in top3.iterrows():
            items = row.get("weights_items") or []
            if items:
                detail_df = pd.DataFrame(items)
                detail_df.columns = ["אחוז", "קרן", "מסלול"]
                sheet_name = f"חלופה {i+1}"[:31]
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
with st.spinner("🔄 טוען נתונים מ-Google Sheets..."):
    df_long, service_map, load_warnings = load_funds_long(FUNDS_GSHEET_ID, SERVICE_GSHEET_ID)
    if load_warnings:
        with st.expander('אזהרות טעינת נתונים', expanded=False):
            for w in load_warnings:
                st.warning(w)


if load_warnings:
    for w in load_warnings:
        st.warning(f"⚠️ {w}")

if df_long.empty:
    err_details = " | ".join(load_warnings) if load_warnings else "סיבה לא ידועה"
    st.error(
        f"❌ לא הצלחתי לטעון נתונים מ-Google Sheets.\n\n"
        f"**פרטי השגיאה:** {err_details}\n\n"
        f"ודא שהגיליונות פתוחים לשיתוף ('Anyone with the link') ושמבנה הגיליון תקין."
    )
    st.stop()

n_tracks  = df_long["track"].nunique()
n_records = len(df_long)
all_funds = sorted(df_long["fund"].unique().tolist())


# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
def _init_state():
    st.session_state.setdefault("n_funds",           2)
    st.session_state.setdefault("mix_policy",        "מותר לערבב מנהלים")
    st.session_state.setdefault("step",              5)
    st.session_state.setdefault("primary_rank",      "דיוק")
    st.session_state.setdefault("locked_fund",       "")
    st.session_state.setdefault("selected_managers", None)  # None = כולם נבחרו
    st.session_state.setdefault("targets",      {"foreign": 30.0, "stocks": 40.0, "fx": 25.0, "illiquid": 20.0})
    st.session_state.setdefault("include",      {"foreign": True, "stocks": True, "fx": False, "illiquid": False})
    st.session_state.setdefault("constraint",   {
        "foreign":  ("רך",    "בדיוק"),
        "stocks":   ("רך",    "בדיוק"),
        "fx":       ("רך",    "לפחות"),
        "illiquid": ("קשיח",  "לכל היותר"),
    })
    st.session_state.setdefault("last_results", None)
    st.session_state.setdefault("last_note",    "")
    st.session_state.setdefault("run_history",  [])

_init_state()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">📊 Profit Mix Optimizer</div>
  <div class="app-sub">חיפוש תמהיל אופטימלי בין מסלולי קרנות השתלמות</div>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c, col_refresh = st.columns([2, 2, 2, 1])
col_a.metric("מסלולי השקעה",        n_tracks)
col_b.metric("קופות (מנהל×מסלול)", n_records)
with col_refresh:
    st.markdown("<div style='padding-top:24px'>", unsafe_allow_html=True)
    if st.button("🔄 רענן נתונים", help="טוען מחדש את הנתונים מ-Google Sheets", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

all_managers = sorted(df_long["manager"].unique().tolist())

# Initialize selected_managers to all if not set yet
if st.session_state["selected_managers"] is None:
    st.session_state["selected_managers"] = all_managers.copy()

# Keep only valid managers (in case data changed)
st.session_state["selected_managers"] = [
    m for m in st.session_state["selected_managers"] if m in all_managers
]
if not st.session_state["selected_managers"]:
    st.session_state["selected_managers"] = all_managers.copy()

n_active_managers = len(st.session_state["selected_managers"])
n_total_managers  = len(all_managers)

with col_c:
    st.metric(
        "מנהלים פעילים",
        f"{n_active_managers} / {n_total_managers}",
        help="לחץ על החץ כדי לבחור אילו מנהלים לכלול בניתוח"
    )

with st.expander(
    f"🏢 סינון מנהלים — {'כולם נבחרו' if n_active_managers == n_total_managers else f'{n_active_managers} מתוך {n_total_managers} נבחרו'}",
    expanded=False
):
    st.caption("בטל סימון מנהלים שאינך רוצה שיופיעו בניתוח ובהשוואה.")

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("✅ בחר הכל", key="mgr_all", use_container_width=True):
            st.session_state["selected_managers"] = all_managers.copy()
            st.rerun()
    with btn_col2:
        if st.button("☐ נקה הכל", key="mgr_none", use_container_width=True):
            st.session_state["selected_managers"] = []
            st.rerun()

    st.markdown("---")

    # Show managers in a responsive grid with checkboxes
    mgr_cols = st.columns(3)
    new_selection = []
    for i, mgr_name in enumerate(all_managers):
        with mgr_cols[i % 3]:
            checked = mgr_name in st.session_state["selected_managers"]
            # Show fund count per manager as hint
            fund_count = df_long[df_long["manager"] == mgr_name]["fund"].nunique()
            if st.checkbox(
                f"{mgr_name}  ·  {fund_count} קרנות",
                value=checked,
                key=f"mgr_cb_{mgr_name}",
            ):
                new_selection.append(mgr_name)

    if new_selection != st.session_state["selected_managers"]:
        st.session_state["selected_managers"] = new_selection
        st.rerun()

# ── Apply manager filter to df_long for all downstream analysis ──
if st.session_state["selected_managers"] and len(st.session_state["selected_managers"]) < n_total_managers:
    df_active = df_long[df_long["manager"].isin(st.session_state["selected_managers"])].copy()
    st.info(f"🔍 מנותחות **{len(df_active)}** קרנות מתוך {n_records} (מנהלים: {', '.join(st.session_state['selected_managers'])})")
else:
    df_active = df_long  # כולם נבחרו — ללא סינון

all_funds  = sorted(df_active["fund"].unique().tolist())

st.divider()

NAV_OPTIONS = ["⚙️ הגדרות יעד", "📈 תוצאות", "⚖️ השוואת מסלולים", "🔍 שקיפות", "🕓 היסטוריה"]
st.session_state.setdefault("active_tab", "⚙️ הגדרות יעד")
active_tab = st.radio(
    "ניווט",
    options=NAV_OPTIONS,
    horizontal=True,
    key="active_tab",
    label_visibility="collapsed",
)


# ─────────────────────────────────────────────
# Tab 1: Settings
# ─────────────────────────────────────────────
if active_tab == "⚙️ הגדרות יעד":
    st.subheader("הגדרות בסיס")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state["n_funds"] = st.selectbox(
            "מספר קופות לשלב",
            options=[1, 2, 3],
            index=[1, 2, 3].index(st.session_state["n_funds"]),
        )
    with c2:
        st.session_state["mix_policy"] = st.selectbox(
            "מדיניות מנהלים",
            options=["מותר לערבב מנהלים", "אותו מנהל בלבד"],
            index=0 if st.session_state["mix_policy"] == "מותר לערבב מנהלים" else 1,
        )
    with c3:
        st.session_state["step"] = st.selectbox(
            "צעד משקלים (%)",
            options=[1, 2, 5, 10, 20],
            index=[1, 2, 5, 10, 20].index(st.session_state["step"]),
            help="צעד קטן → חיפוש יסודי יותר אך איטי יותר",
        )
    with c4:
        st.session_state["primary_rank"] = st.selectbox(
            "דירוג ראשי",
            options=["דיוק", "שארפ", "שירות"],
            index=["דיוק", "שארפ", "שירות"].index(st.session_state["primary_rank"]),
        )

    # Locked fund
    st.subheader("נעילת קרן (אופציונלי)")
    lock_opts = ["ללא"] + all_funds
    lock_idx = 0
    if st.session_state["locked_fund"] in all_funds:
        lock_idx = all_funds.index(st.session_state["locked_fund"]) + 1
    locked = st.selectbox(
        "בחר קרן שחייבת להופיע בכל חלופה",
        options=lock_opts,
        index=lock_idx,
        help="האופטימייזר יסנן רק תמהילים שכוללים קרן זו",
    )
    st.session_state["locked_fund"] = "" if locked == "ללא" else locked

    st.divider()
    st.subheader("יעדים ומגבלות")

    # Score explanation
    st.markdown("""
    <div class="score-tip">
    💡 <b>מה זה Score?</b> סכום הסטיות המנורמלות מהיעדים שבחרת עבור כל פרמטר מסומן.
    Score = 0 = התאמה מושלמת. Score גבוה = סטייה גדולה מהיעד.
    מגבלה <b>קשיחה</b> = פתרון שלא עומד בה נפסל לחלוטין.
    מגבלה <b>רכה</b> = משפיעה רק על הדירוג (Score).
    </div>
    """, unsafe_allow_html=True)

    header_cols = st.columns([1.4, 1.1, 1.3, 1.0, 1.1])
    for col, lbl in zip(header_cols, ["פרמטר", "כלול בדירוג", "יעד (%)", "קשיחות", "כיוון"]):
        col.markdown(f"**{lbl}**")

    def _metric_row(key: str, label: str, default_mode: str, max_val: float = 100.0):
        cols = st.columns([1.4, 1.1, 1.3, 1.0, 1.1])
        with cols[0]: st.write(label)
        with cols[1]:
            inc = st.checkbox(" ", value=st.session_state["include"].get(key, False), key=f"inc_{key}")
        with cols[2]:
            val = st.slider(" ", 0.0, max_val,
                            float(st.session_state["targets"].get(key, 0.0)),
                            step=0.5, key=f"tgt_{key}", label_visibility="collapsed")
        with cols[3]:
            hard = st.selectbox(" ", ["רך", "קשיח"],
                                index=0 if st.session_state["constraint"].get(key, ("רך",))[0] == "רך" else 1,
                                key=f"hard_{key}", label_visibility="collapsed")
        with cols[4]:
            mode = st.selectbox(" ", ["בדיוק", "לפחות", "לכל היותר"],
                                index=["בדיוק", "לפחות", "לכל היותר"].index(
                                    st.session_state["constraint"].get(key, ("רך", default_mode))[1]),
                                key=f"mode_{key}", label_visibility="collapsed")
        st.session_state["include"][key]    = inc
        st.session_state["targets"][key]    = float(val)
        st.session_state["constraint"][key] = (hard, mode)

    _metric_row("foreign",  "חו״ל",      "בדיוק",       120.0)
    _metric_row("stocks",   "מניות",     "בדיוק")
    _metric_row("fx",       'מט"ח',      "לפחות",       120.0)
    _metric_row("illiquid", "לא־סחיר",   "לכל היותר")

    st.divider()
    run = st.button("🔍 חשב 3 חלופות", type="primary", use_container_width=True)

    if run:
        with st.spinner("מחשב... ⚡ (חיפוש מואץ עם NumPy)"):
            try:
                sols, note = find_best_solutions(
                    df=df_active,
                    n_funds=st.session_state["n_funds"],
                    step=st.session_state["step"],
                    mix_policy=st.session_state["mix_policy"],
                    include=st.session_state["include"],
                    constraint=st.session_state["constraint"],
                    targets=st.session_state["targets"],
                    primary_rank=st.session_state["primary_rank"],
                    locked_fund=st.session_state["locked_fund"],
                    max_solutions_scan=20000,
                )
                st.session_state["last_note"] = note
                if sols.empty:
                    st.session_state["last_results"] = None
                    st.session_state["active_tab"] = "📈 תוצאות"
                    st.error(f"לא נמצאו פתרונות. {note}")
                    st.rerun()
                else:
                    top3 = _pick_three_distinct(sols, st.session_state["primary_rank"])
                    result = {"solutions_all": sols.head(100), "top3": top3, "targets": dict(st.session_state["targets"]), "ts": datetime.now().strftime("%H:%M:%S")}
                    del sols
                    st.session_state["last_results"] = result
                    hist = st.session_state.get("run_history", [])
                    hist.insert(0, result)
                    st.session_state["run_history"] = hist[:2]
                    st.session_state["active_tab"] = "📈 תוצאות"
                    st.success(f"✅ מוכן! {note}")
                    st.rerun()
            except Exception as _e:
                err_txt = traceback.format_exc()
                st.session_state["_last_error"] = err_txt
                st.error(f"שגיאה בחישוב: {_e}\n\nפרטים נוספים בטאב 'שקיפות'.")


# ─────────────────────────────────────────────
# Tab 2: Results
# ─────────────────────────────────────────────
if active_tab == "📈 תוצאות":
    st.subheader("תוצאות")
    res = st.session_state.get("last_results")
    if res is None:
        st.info("עבור לטאב **הגדרות יעד** ולחץ **חשב 3 חלופות**.")
    else:
        targets_used = res.get("targets", {})
        st.caption(st.session_state.get("last_note", ""))

        df_head = res.get("solutions_all")
        recs = _pick_recommendations(df_head)

        if not recs:
            st.warning("אין מספיק פתרונות להצגת תוצאות. נסה להריץ חיפוש מחדש.")
        else:
            manager_names = sorted(df_active["manager"].dropna().unique().tolist()) if "manager" in df_active.columns else []

            # Build 3-row dataframe for charts/tables
            rows = []
            weighted = recs["weighted"]
            accurate = recs["accurate"]
            sharpe   = recs["sharpe"]

            for key, row, title in [
                ("weighted", weighted, "חלופה משוקללת"),
                ("accurate", accurate, "החלופה המדויקת ביותר"),
                ("sharpe",   sharpe,   "החלופה עם השארפ הטוב ביותר"),
            ]:
                row = dict(row)
                row["חלופה"] = title
                row["weights_items"] = _weights_items(row.get("weights"), row.get("קופות",""), row.get("מסלולים",""))
                row["משקלים"] = _weights_short(row.get("weights"))
                rows.append(row)

            top3 = pd.DataFrame(rows)

            # Export button (for these 3)
            excel_bytes = _export_excel(top3)
            st.download_button(
                "⬇️ ייצוא לאקסל",
                data=excel_bytes,
                file_name="profit_mix_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.markdown("---")

            # Primary recommendation (prominent)
            _render_reco_card(weighted, "חלופה משוקללת", primary=True, manager_names=manager_names, card_key='weighted')

            # Secondary recommendations (side-by-side, less prominent)
            c1, c2 = st.columns(2)
            with c1:
                _render_reco_card(accurate, "החלופה המדויקת ביותר", primary=False, manager_names=manager_names, card_key='accurate')
            with c2:
                _render_reco_card(sharpe, "החלופה עם השארפ הטוב ביותר", primary=False, manager_names=manager_names, card_key='sharpe')

            st.markdown("---")

            # Radar chart
            st.subheader("📡 השוואה ויזואלית – גרף Radar")
            st.caption("ערכי השארפ מוכפלים ×10 וציוני שירות מחולקים ÷10 כדי להציגם באותה סקלה. הקו המנוקד האדום = היעדים שלך.")
            fig = _radar_chart(top3, targets_used)
            _safe_plotly(fig)

            st.markdown("---")

            # Comparison table
            st.subheader("📊 טבלת השוואה")
            compare_cols = [
                "חלופה", "מנהלים", "משקלים",
                'חו"ל (%)', "מניות (%)", 'מט"ח (%)', "לא־סחיר (%)",
                "שארפ משוקלל", "שירות משוקלל", "score",
            ]
            exist_cols = [c for c in compare_cols if c in top3.columns]
            display_df = top3[exist_cols].copy()
            display_df = display_df.rename(columns={"score": "Score (סטייה)"})
            for col in ["חו\"ל (%)", "מניות (%)", "מט\"ח (%)", "לא־סחיר (%)"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
            if "שארפ משוקלל" in display_df.columns:
                display_df["שארפ משוקלל"] = display_df["שארפ משוקלל"].apply(lambda x: f"{x:.2f}")
            if "שירות משוקלל" in display_df.columns:
                display_df["שירות משוקלל"] = display_df["שירות משוקלל"].apply(lambda x: f"{x:.1f}")
            if "Score (סטייה)" in display_df.columns:
                display_df["Score (סטייה)"] = display_df["Score (סטייה)"].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# Tab 3: Fund Comparison
# ─────────────────────────────────────────────
if active_tab == "⚖️ השוואת מסלולים":
    st.subheader("⚖️ השוואת מסלולי השקעה")
    st.caption("בחר עד 6 מסלולים להשוואה מלאה – נתוני חשיפות, שארפ וציון שירות זה לצד זה.")

    all_tracks = sorted(df_active["track"].unique().tolist())

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        compare_tracks = st.multiselect(
            "🔍 בחר לפי מסלול",
            options=all_tracks,
            placeholder="הקלד שם מסלול...",
            help="כל הקרנות במסלול יוצגו בטבלה, ממוינות לפי שארפ",
        )
    with col_s2:
        compare_funds_direct = st.multiselect(
            "🔍 בחר לפי שם קרן ספציפית",
            options=all_funds,
            placeholder="הקלד שם קרן...",
            max_selections=10,
        )

    # Gather rows
    selected_rows = []
    for track in (compare_tracks or []):
        subset = df_active[df_active["track"] == track]
        if not subset.empty:
            # הוסף את כל הקרנות במסלול, ממוינות לפי שארפ
            for _, row in subset.sort_values("sharpe", ascending=False).iterrows():
                selected_rows.append(row)
    for fund_name in (compare_funds_direct or []):
        rows = df_active[df_active["fund"] == fund_name]
        if not rows.empty:
            selected_rows.append(rows.iloc[0])

    # De-duplicate by fund name
    seen_funds: set = set()
    unique_rows = []
    for r in selected_rows:
        key = str(r["fund"])
        if key not in seen_funds:
            seen_funds.add(key)
            unique_rows.append(r)

    if not unique_rows:
        st.info("בחר לפחות מסלול או קרן אחת כדי לראות את ההשוואה.")
    else:
        n_sel = len(unique_rows)
        st.divider()

        # ── Build comparison dataframe ─────────
        comp_data = []
        for r in unique_rows:
            comp_data.append({
                "קרן":          str(r.get("fund", "")),
                "מסלול":        str(r.get("track", "")),
                "מנהל":         str(r.get("manager", "")),
                'חו"ל (%)':     r.get("foreign",  np.nan),
                "ישראל (%)":    round(100.0 - float(r.get("foreign", 0) or 0), 2),
                "מניות (%)":    r.get("stocks",   np.nan),
                'מט"ח (%)':     r.get("fx",       np.nan),
                "לא־סחיר (%)":  r.get("illiquid", np.nan),
                "שארפ":         r.get("sharpe",   np.nan),
                "ציון שירות":   r.get("service",  np.nan),
            })
        comp_df = pd.DataFrame(comp_data)

        numeric_cols_cmp = ['חו"ל (%)', "ישראל (%)", "מניות (%)", 'מט"ח (%)', "לא־סחיר (%)", "שארפ", "ציון שירות"]

        # ── Smart filter ──────────────────────
        with st.expander("🔎 סינון חכם – הצג רק שורות שעומדות בתנאי", expanded=False):
            st.caption("הגדר עד 3 תנאי סינון. רק שורות שעומדות בכולם יוצגו בטבלה.")

            # Up to 3 filter rows
            filters = []
            for fi in range(3):
                fc1, fc2, fc3 = st.columns([2, 1.5, 1.5])
                with fc1:
                    param = st.selectbox(
                        "פרמטר" if fi == 0 else " ",
                        options=["—"] + numeric_cols_cmp,
                        key=f"flt_param_{fi}",
                        label_visibility="visible" if fi == 0 else "collapsed",
                    )
                with fc2:
                    direction = st.selectbox(
                        "כיוון" if fi == 0 else " ",
                        options=["לפחות (≥)", "לכל היותר (≤)", "בדיוק (=)"],
                        key=f"flt_dir_{fi}",
                        label_visibility="visible" if fi == 0 else "collapsed",
                    )
                with fc3:
                    threshold = st.number_input(
                        "ערך" if fi == 0 else " ",
                        value=0.0,
                        step=1.0,
                        key=f"flt_val_{fi}",
                        label_visibility="visible" if fi == 0 else "collapsed",
                    )
                if param != "—":
                    filters.append((param, direction, float(threshold)))

        # Apply filters to comp_df
        filtered_df = comp_df.copy()
        active_filter_desc = []
        for (param, direction, threshold) in filters:
            if param not in filtered_df.columns:
                continue
            col_numeric = pd.to_numeric(filtered_df[param], errors="coerce")
            if "לפחות" in direction:
                mask = col_numeric >= threshold
                active_filter_desc.append(f"{param} ≥ {threshold:.1f}")
            elif "לכל היותר" in direction:
                mask = col_numeric <= threshold
                active_filter_desc.append(f"{param} ≤ {threshold:.1f}")
            else:
                mask = col_numeric.between(threshold - 0.5, threshold + 0.5)
                active_filter_desc.append(f"{param} = {threshold:.1f}")
            filtered_df = filtered_df[mask]

        if active_filter_desc:
            n_before = len(comp_df)
            n_after  = len(filtered_df)
            if n_after == 0:
                st.warning(f"אין שורות שעומדות בתנאי: {' | '.join(active_filter_desc)}")
                filtered_df = comp_df.copy()   # fallback: show all
            else:
                st.success(f"✅ מציג {n_after} מתוך {n_before} שורות — תנאי: {' | '.join(active_filter_desc)}")

        # ── Styled HTML table ──────────────────
        def _cell_bg(val, col_name, col_series) -> str:
            try:
                v = float(val)
                nums = col_series.dropna().astype(float).tolist()
                if len(nums) < 2: return ""
                mn, mx = min(nums), max(nums)
                if mx == mn: return ""
                ratio = 1.0 - (v - mn)/(mx - mn) if col_name == "לא־סחיר (%)" else (v - mn)/(mx - mn)
                g = int(220 + ratio * 35)
                r_ch = int(255 - ratio * 80)
                return f"background: rgba({r_ch},{g},200,0.28);"
            except Exception:
                return ""

        header_cells = "".join(f"<th>{_esc(c)}</th>" for c in filtered_df.columns)
        rows_html = ""
        for _, row in filtered_df.iterrows():
            cells = ""
            for col in filtered_df.columns:
                val = row[col]
                style = _cell_bg(val, col, filtered_df[col]) if col in numeric_cols_cmp else ""
                if col in numeric_cols_cmp:
                    try:
                        dec = 1 if col in ["שארפ", "ציון שירות"] else 2
                        unit = "%" if "%" in col else ""
                        display = f"{float(val):.{dec}f}{unit}"
                    except Exception:
                        display = "—"
                else:
                    display = _esc(str(val))
                cells += f"<td style='{style}'>{display}</td>"
            rows_html += f"<tr>{cells}</tr>"

        st.markdown(f"""
        <style>
        .cmp-table {{
          width:100%; border-collapse:separate; border-spacing:0;
          border-radius:14px; overflow:hidden; border:1px solid #e2e8f0;
          font-size:13px; direction:rtl; margin-top:4px;
        }}
        .cmp-table th {{
          background:#0f172a; color:#f8fafc; padding:10px 12px;
          font-weight:700; white-space:nowrap; text-align:right;
        }}
        .cmp-table td {{
          padding:9px 12px; border-bottom:1px solid #f1f5f9;
          text-align:right; vertical-align:middle;
        }}
        .cmp-table tr:last-child td {{ border-bottom:none; }}
        .cmp-table tr:hover td {{ filter:brightness(0.97); }}
        @media (prefers-color-scheme:dark) {{
          .cmp-table {{ border-color:#334155; }}
          .cmp-table td {{ border-color:#1e293b; color:#cbd5e1; }}
        }}
        </style>
        <div style="overflow-x:auto">
          <table class="cmp-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        <p style="font-size:11px;color:#94a3b8;margin-top:6px;">
          🟢 ירוק = ערך גבוה יותר (טוב יותר) | 🔴 אדום = ערך נמוך יותר | עבור לא-סחיר: הפוך
        </p>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Radar chart ────────────────────────
        if len(filtered_df) >= 2:
            st.subheader("📡 גרף Radar – שורות מסוננות")
            st.caption("שארפ ×10 | שירות ÷10 | לא-סחיר הפוך (100 − ערך)")
            radar_cats = ['חו"ל', "מניות", 'מט"ח', "לא-סחיר (הפוך)", "שארפ×10", "שירות÷10"]
            palette = ["#2563eb","#16a34a","#ea580c","#7c3aed","#0891b2","#b45309"]
            fig_cmp = go.Figure()
            for i, (_, r) in enumerate(filtered_df.iterrows()):
                vals = [
                    float(r.get('חו"ל (%)',    0) or 0),
                    float(r.get("מניות (%)",   0) or 0),
                    float(r.get('מט"ח (%)',    0) or 0),
                    max(0.0, 100.0 - float(r.get("לא־סחיר (%)", 0) or 0)),
                    float(r.get("שארפ",        0) or 0) * 10,
                    float(r.get("ציון שירות",  0) or 0) / 10,
                ]
                label = str(r.get("קרן", f"קרן {i+1}"))[:28]
                fig_cmp.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=radar_cats + [radar_cats[0]],
                    fill="toself", opacity=0.22,
                    line=dict(color=palette[i % len(palette)], width=2.5),
                    name=label,
                ))
            fig_cmp.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
                    angularaxis=dict(direction="clockwise"),
                ),
                showlegend=True, height=430,
                margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                font=dict(family="sans-serif", size=11),
            )
            _safe_plotly(fig_cmp)

        # ── Bar chart per metric ───────────────
        st.subheader("📊 פרמטר בודד – השוואה")
        bar_metric = st.selectbox(
            "בחר פרמטר להצגה",
            options=['חו"ל (%)', "מניות (%)", 'מט"ח (%)', "לא־סחיר (%)", "שארפ", "ציון שירות"],
            key="cmp_bar_metric",
        )
        bar_df = filtered_df[["קרן", bar_metric]].dropna().sort_values(bar_metric, ascending=False)
        if not bar_df.empty:
            unit = "%" if "%" in bar_metric else ""
            colors = ["#2563eb"] * len(bar_df)
            if len(bar_df) > 1:
                colors[0] = "#16a34a"   # best = green
                colors[-1] = "#ef4444"  # worst = red (for most metrics)
                if bar_metric == "לא־סחיר (%)":
                    colors[0], colors[-1] = "#ef4444", "#16a34a"

            fig_bar = go.Figure(go.Bar(
                x=bar_df["קרן"],
                y=bar_df[bar_metric],
                marker_color=colors,
                text=bar_df[bar_metric].apply(lambda v: f"{v:.1f}{unit}"),
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=320,
                margin=dict(t=30, b=90, l=10, r=10),
                xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            _safe_plotly(fig_bar)

        # ── Export ─────────────────────────────
        cmp_out = io.BytesIO()
        with pd.ExcelWriter(cmp_out, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, sheet_name="השוואת מסלולים", index=False)
        st.download_button(
            "⬇️ ייצוא השוואה לאקסל",
            data=cmp_out.getvalue(),
            file_name="fund_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ─────────────────────────────────────────────
# Tab 4: Transparency
# ─────────────────────────────────────────────
if active_tab == "🔍 שקיפות":
    st.subheader("פירוט חישוב")

    # ── Debug: show last unhandled exception if any ──
    if st.session_state.get("_last_error"):
        with st.expander("⚠️ שגיאה אחרונה (לדיבוג)", expanded=True):
            st.code(st.session_state["_last_error"], language="python")

    with st.expander("פרמטרי הריצה האחרונה"):
        st.json({
            "מספר קופות":    st.session_state["n_funds"],
            "מדיניות":       st.session_state["mix_policy"],
            "צעד":           st.session_state["step"],
            "דירוג ראשי":    st.session_state["primary_rank"],
            "קרן נעולה":     st.session_state["locked_fund"] or "ללא",
            "כולל בדירוג":   st.session_state["include"],
            "יעדים":         st.session_state["targets"],
            "קשיחות/כיוון":  {k: list(v) for k, v in st.session_state["constraint"].items()},
            "הערת ריצה":     st.session_state.get("last_note", ""),
        }, expanded=False)

    res = st.session_state.get("last_results")
    if res is None:
        st.info("אין נתונים – הרץ חיפוש תחילה.")
    else:
        st.markdown("**מועמדים (200 ראשונים לאחר מיון):**")
        cand = res["solutions_all"].head(200).copy()
        show_cols = ["מנהלים", "קופות", "מסלולים",
                     'חו"ל (%)', "מניות (%)", 'מט"ח (%)', "לא־סחיר (%)",
                     "שארפ משוקלל", "שירות משוקלל", "score", "weights"]
        exist = [c for c in show_cols if c in cand.columns]
        cand = cand[exist].copy()
        if "weights" in cand.columns:
            cand["משקלים"] = cand["weights"].apply(
                lambda w: " / ".join(f"{int(x)}%" for x in w) if isinstance(w, (tuple, list)) else str(w)
            )
            cand = cand.drop(columns=["weights"])
        cand = cand.rename(columns={"score": "Score"})
        st.dataframe(cand, use_container_width=True, hide_index=True,
                     column_config={"קופות": st.column_config.TextColumn(width="large"),
                                    "מסלולים": st.column_config.TextColumn(width="large")})


# ─────────────────────────────────────────────
# Tab 5: History
# ─────────────────────────────────────────────
if active_tab == "🕓 היסטוריה":
    st.subheader("🕓 3 ריצות אחרונות")
    history = st.session_state.get("run_history", [])
    if not history:
        st.info("עדיין לא בוצעה ריצה.")
    else:
        for hi, h_res in enumerate(history):
            ts = h_res.get("ts", f"ריצה {hi+1}")
            tgts = h_res.get("targets", {})
            tgt_str = " | ".join(f"{k}={v:.0f}%" for k, v in tgts.items())
            with st.expander(f"🕐 {ts}  –  יעדים: {tgt_str}"):
                h_top3 = h_res.get("top3")
                if h_top3 is not None and not h_top3.empty:
                    for _, row in h_top3.iterrows():
                        funds = row.get("קופות", "")
                        weights = row.get("משקלים", "")
                        score = row.get("score", "")
                        alt = row.get("חלופה", "")
                        st.markdown(f"**{alt}** — {funds} ({weights}) — Score: {score:.4f}" if isinstance(score, float) else f"**{alt}** — {funds}")
                    # Re-export button for history
                    excel_h = _export_excel(h_top3)
                    st.download_button(
                        f"⬇️ ייצוא ריצה זו",
                        data=excel_h,
                        file_name=f"profit_mix_{ts.replace(':','-')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"hist_dl_{hi}",
                    )


st.caption("© Profit Mix Optimizer | ישראל = 100% − חו״ל | חיפוש מואץ עם NumPy vectorized")