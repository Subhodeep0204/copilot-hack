import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

# ==== EDIT THIS PATH ONLY ====
MIS_PATH = r"C:\Users\A0807669\OneDrive - Aon\Desktop\CopyPaste\BaseTopup\Claim_Mis0.1_Altimetriksuminsured.xlsx"
# =============================

# Output file will be written next to MIS file
OUT_FILE = os.path.join(
    os.path.dirname(MIS_PATH),
    f"Claim_Mis0.1_Altimetriksuminsured_split_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

# ---------- helpers ----------
def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "").replace("(", "").replace(")", "")

def _find_col(df: pd.DataFrame, candidates, required=True) -> str:
    look = {_norm(c): c for c in df.columns}
    for c in candidates:
        k = _norm(c)
        if k in look:
            return look[k]
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}\nAvailable: {list(df.columns)}")
    return None

def _guess_date_col(df: pd.DataFrame) -> str | None:
    names = ["admissiondate","admitdate","claimdate","claimeddate","servicedate","date"]
    # exact preferred
    for n in names:
        for c in df.columns:
            if _norm(c) == n:
                return c
    # heuristic
    for c in df.columns:
        try:
            p = pd.to_datetime(df[c], errors="coerce")
            if p.notna().mean() > 0.6:
                return c
        except Exception:
            pass
    return None

def _to_topup(val: str) -> str:
    """Convert 'Base_22-23' / 'Base 22-23' / 'Base' -> 'Topup_*' safely; keep 'Topup*' as-is."""
    s = "" if pd.isna(val) else str(val)
    s_low = s.lower()
    if "topup" in s_low:
        return s
    m = re.search(r"(base)[\s_:-]*([\w\-\/]+)?", s, flags=re.IGNORECASE)
    if m:
        suffix = m.group(2) or ""
        if "_" in s:   return f"Topup_{suffix}" if suffix else "Topup"
        if " " in s:   return f"Topup {suffix}" if suffix else "Topup"
        return f"Topup_{suffix}" if suffix else "Topup"
    return "Topup"

# ---------- load ----------
claims = pd.read_excel(MIS_PATH)

# ---------- detect columns ----------
EMP_C  = _find_col(claims, ["employeeid","employee_id","empid","employeeno","employee no"])
POL_C  = _find_col(claims, ["policynumber","policy number","policy_no","policyno","policyid"])
INC_C  = _find_col(claims, ["incurredamount","incurred amount"])
CLM_C  = _find_col(claims, ["claimamount","claim amount"], required=False)   # optional
SET_C  = _find_col(claims, ["settledamount","settled"], required=False)      # optional
ENT_C  = _find_col(claims, ["entityname","entity","baseortopup","covertype"], required=False) or "entityname"

if ENT_C not in claims.columns:
    claims[ENT_C] = "Base"

DATE_C = _guess_date_col(claims)

# sum insured fields in MIS
SI_C  = _find_col(claims, ["suminsured","sum insured","si"], required=False)
BAL_C = _find_col(claims, ["balancesuminusred","balance sum insured","balance si","topup si","topupsi"], required=False)

if not SI_C or not BAL_C:
    raise KeyError("Could not detect 'suminsured' / 'balancesuminusred' in MIS file.")

# ensure numeric
claims[INC_C] = pd.to_numeric(claims[INC_C], errors="coerce").fillna(0.0)
claims[SI_C]  = pd.to_numeric(claims[SI_C],  errors="coerce").fillna(0.0)
claims[BAL_C] = pd.to_numeric(claims[BAL_C], errors="coerce").fillna(0.0)
if CLM_C: claims[CLM_C] = pd.to_numeric(claims[CLM_C], errors="coerce").fillna(0.0)
if SET_C: claims[SET_C] = pd.to_numeric(claims[SET_C], errors="coerce").fillna(0.0)

# keep original for totals check
claims_orig = claims.copy()

# key
claims["_key"] = claims[POL_C].astype(str).str.strip() + "||" + claims[EMP_C].astype(str).str.strip()

# ---------- derive CAP_MAP from MIS ----------
CAP_MAP: dict[str, float] = {}

for (pol, emp), grp in claims.groupby([POL_C, EMP_C]):
    base_si   = float(grp[SI_C].max())       # assume constant per employee
    topup_si  = float(grp[BAL_C].max())      # >0 means topup opted
    total_inc = float(grp[INC_C].sum())

    # Equivalent to: has topup AND incurred > base SI
    if topup_si > 0 and total_inc > base_si + 0.5:
        key = f"{pol}||{emp}"
        CAP_MAP[key] = base_si

# helper: set Topup suminsured from balancesuminusred
def _apply_topup_si(row: pd.Series) -> pd.Series:
    """
    For Topup rows, if balancesuminusred is present and > 0,
    copy it into suminsured.
    """
    try:
        val = float(row[BAL_C])
    except Exception:
        val = np.nan
    if not pd.isna(val) and val > 0:
        row[SI_C] = val
    return row

# ---------- allocation ----------
def allocate_group(grp: pd.DataFrame, base_cap: float) -> pd.DataFrame:
    g = grp.copy()

    # chronological order
    if DATE_C and DATE_C in g.columns:
        g["_sort"] = pd.to_datetime(g[DATE_C], errors="coerce")
    else:
        g["_sort"] = np.arange(len(g))

    g.sort_values(by="_sort", kind="mergesort", inplace=True)

    out = []
    remaining = float(base_cap)

    for _, row in g.iterrows():
        inc  = float(row[INC_C])
        clm  = float(row[CLM_C]) if CLM_C else 0.0
        sett = float(row[SET_C]) if SET_C else 0.0

        # Base exhausted or no incurred -> full Topup
        if remaining <= 0 or inc <= 0:
            r = row.copy()
            r[ENT_C] = _to_topup(r[ENT_C])
            r = _apply_topup_si(r)
            out.append(r)
            continue

        # Fits entirely in remaining base
        if inc <= remaining + 1e-9:
            out.append(row.copy())
            remaining -= inc
        else:
            # split into base + topup
            base_inc = remaining
            top_inc  = inc - remaining

            ratio_c = (clm / inc) if inc else 0.0
            ratio_s = (sett / inc) if inc else 0.0

            # Base part
            r_base = row.copy()
            r_base[INC_C] = base_inc
            if CLM_C: r_base[CLM_C] = round(base_inc * ratio_c, 0)
            if SET_C: r_base[SET_C] = round(base_inc * ratio_s, 0)
            out.append(r_base)

            # Topup part
            r_top = row.copy()
            r_top[INC_C] = top_inc
            if CLM_C: r_top[CLM_C] = round(top_inc * ratio_c, 0)
            if SET_C: r_top[SET_C] = round(top_inc * ratio_s, 0)
            r_top[ENT_C] = _to_topup(r_top[ENT_C])
            r_top = _apply_topup_si(r_top)
            out.append(r_top)

            remaining = 0.0

    return pd.DataFrame(out).drop(columns=["_sort"], errors="ignore")

# process only target (policy, employee) pairs
to_proc = claims[claims["_key"].isin(CAP_MAP.keys())].copy()
to_keep = claims[~claims["_key"].isin(CAP_MAP.keys())].copy()

processed = []
for key, grp in to_proc.groupby("_key", sort=False):
    processed.append(allocate_group(grp, CAP_MAP[key]))

processed_all = pd.concat(processed, ignore_index=True) if processed else pd.DataFrame(columns=claims.columns)
final = pd.concat(
    [
        to_keep.drop(columns=["_key"], errors="ignore"),
        processed_all.drop(columns=["_key"], errors="ignore"),
    ],
    ignore_index=True
)

# ---------- summary ----------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pol, emp), grp in df.groupby([POL_C, EMP_C]):
        ent_lower = grp[ENT_C].astype(str).str.lower()
        base_mask = ent_lower.str.startswith("base")
        top_mask  = ent_lower.str.startswith("topup")

        def s(col, mask):
            return pd.to_numeric(grp.loc[mask, col], errors="coerce").sum() if (col and col in grp.columns) else np.nan

        key = f"{pol}||{emp}"
        base_cap = CAP_MAP.get(key, np.nan)
        base_inc = s(INC_C, base_mask)

        if pd.isna(base_cap) or pd.isna(base_inc):
            ok = False
        else:
            ok = abs(base_inc - base_cap) < 0.5

        rows.append({
            "PolicyNumber": pol,
            "EmployeeID": emp,
            "Base_SI_Cap": base_cap,
            "Base_ClaimAmount": s(CLM_C, base_mask),
            "Topup_ClaimAmount": s(CLM_C, top_mask),
            "Base_Incurred": base_inc,
            "Topup_Incurred": s(INC_C, top_mask),
            "Base_Settled": s(SET_C, base_mask),
            "Topup_Settled": s(SET_C, top_mask),
            "Check_BaseEqualsCap": ok,
        })
    return pd.DataFrame(rows)

summary = summarize(final)

# ---------- Check_Totals sheet with traffic-light flags ----------
def make_totals_check(orig: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    def sum_col(df: pd.DataFrame, col_name: str | None) -> float:
        if not col_name or col_name not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[col_name], errors="coerce").sum())

    def totals_block(name: str, df: pd.DataFrame) -> dict:
        return {
            "Version": name,
            "Row_Count": len(df),
            "Total_Incurred": sum_col(df, INC_C),
            "Total_ClaimAmount": sum_col(df, CLM_C),
            "Total_Settled": sum_col(df, SET_C),
            "Incurred_Flag": "",
            "Claim_Flag": "",
            "Settled_Flag": "",
            "Overall_Flag": "",
        }

    orig_block = totals_block("Original_MIS", orig)
    new_block  = totals_block("Post_Split",   new)

    # differences = new - original
    diff_incurred = new_block["Total_Incurred"] - orig_block["Total_Incurred"]
    if np.isnan(orig_block["Total_ClaimAmount"]) or np.isnan(new_block["Total_ClaimAmount"]):
        diff_claim = np.nan
    else:
        diff_claim = new_block["Total_ClaimAmount"] - orig_block["Total_ClaimAmount"]

    if np.isnan(orig_block["Total_Settled"]) or np.isnan(new_block["Total_Settled"]):
        diff_settled = np.nan
    else:
        diff_settled = new_block["Total_Settled"] - orig_block["Total_Settled"]

    tol = 1.0  # rupees

    def flag(val: float) -> str:
        if np.isnan(val):
            return ""
        return "OK" if abs(val) <= tol else "NOT OK"

    incurred_flag = flag(diff_incurred)
    claim_flag    = flag(diff_claim)
    settled_flag  = flag(diff_settled)

    # Overall flag: if all non-empty flags are OK -> OK, else NOT OK
    non_empty_flags = [f for f in [incurred_flag, claim_flag, settled_flag] if f]
    if non_empty_flags and all(f == "OK" for f in non_empty_flags):
        overall_flag = "OK"
    elif non_empty_flags:
        overall_flag = "NOT OK"
    else:
        overall_flag = ""

    diff_block = {
        "Version": "Difference (Post_Split - Original)",
        "Row_Count": new_block["Row_Count"] - orig_block["Row_Count"],
        "Total_Incurred": diff_incurred,
        "Total_ClaimAmount": diff_claim,
        "Total_Settled": diff_settled,
        "Incurred_Flag": incurred_flag,
        "Claim_Flag": claim_flag,
        "Settled_Flag": settled_flag,
        "Overall_Flag": overall_flag,
    }

    return pd.DataFrame([orig_block, new_block, diff_block])

check_totals_df = make_totals_check(claims_orig, final)

# ---------- write ----------
with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter") as w:
    final.to_excel(w, index=False, sheet_name="Claims_Split")
    summary.to_excel(w, index=False, sheet_name="Allocation_Summary")
    check_totals_df.to_excel(w, index=False, sheet_name="Check_Totals")

print(f"Done. Output -> {OUT_FILE}")
