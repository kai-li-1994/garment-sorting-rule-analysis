import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

INPUT_FILE = r"6_JSONL_component_normalized.jsonl"
OUTPUT_DIR = Path("rule_eval_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ROW_JSONL_OUT = OUTPUT_DIR / "7_rule_flags_row_level.jsonl"
ROW_CSV_OUT = OUTPUT_DIR / "7_rule_flags_row_level.csv"
SUMMARY_CSV_OUT = OUTPUT_DIR / "7_rule_summary.csv"
ANY_BARRIER_CSV_OUT = OUTPUT_DIR / "7_any_barrier_summary.csv"
DIAG_MATERIAL_CSV_OUT = OUTPUT_DIR / "7_rule_diagnostic_material.csv"
DIAG_CATEGORY_CSV_OUT = OUTPUT_DIR / "7_rule_diagnostic_category.csv"
SUMMARY_TXT_OUT = OUTPUT_DIR / "7_rule_evaluation_summary.txt"


# =========================================================
# S1 supported recognition scope sets
# =========================================================

S1_MONO = {
    "conservative": {"acrylic", "cotton", "nylon", "polyester", "viscose", "wool"},
    "central": {"acrylic", "cotton", "nylon", "polyester", "viscose", "wool"},
    "expanded": {
        "acetate", "acrylic", "cotton", "elastane", "linen", "lyocell",
        "modal", "nylon", "polyester", "polypropylene", "silk", "viscose", "wool"
    },
}

S1_BINARY = {
    "conservative": {
        frozenset(["acrylic", "cotton"]),
        frozenset(["acrylic", "wool"]),
        frozenset(["cotton", "polyester"]),
        frozenset(["elastane", "polyester"]),
        frozenset(["polyester", "viscose"]),
    },
    "central": {
        frozenset(["acrylic", "cotton"]),
        frozenset(["acrylic", "wool"]),
        frozenset(["cotton", "elastane"]),
        frozenset(["cotton", "polyester"]),
        frozenset(["elastane", "polyester"]),
        frozenset(["polyester", "viscose"]),
    },
    "expanded": {
        frozenset(["acrylic", "cotton"]),
        frozenset(["acrylic", "wool"]),
        frozenset(["cotton", "elastane"]),
        frozenset(["cotton", "nylon"]),
        frozenset(["cotton", "polyester"]),
        frozenset(["cotton", "silk"]),
        frozenset(["cotton", "viscose"]),
        frozenset(["elastane", "nylon"]),
        frozenset(["elastane", "polyester"]),
        frozenset(["nylon", "polyester"]),
        frozenset(["polyester", "viscose"]),
        frozenset(["polyester", "wool"]),
    },
}

S3_THRESHOLDS = {
    "conservative": 10.0,
    "central": 5.0,
    "expanded": 3.0,
}

HIDDEN_COMPONENT_CLASSES = {"lining_component", "filling_component"}

HIDDEN_COMPONENT_NAMES = {
    "lining", "body_lining", "hood_lining", "sleeve_lining", "skirt_lining",
    "cup_lining", "inner_layer", "interlining", "inner_pants", "petticoat",
    "filling", "padding", "body_filling", "upper_body_filling", "under_body_filling",
    "down_proof_fabric",
}


def canon_material(x):
    if x is None:
        return None

    m = str(x).strip().lower()
    if not m:
        return None

    # keep this slightly defensive, even though upstream normalization should already handle most of it
    if m in {"polyamide", "pa", "pa6", "pa66"}:
        return "nylon"
    if m in {"spandex", "lycra"}:
        return "elastane"
    if m in {"merino", "merino wool", "cashmere", "alpaca", "mohair"}:
        return "wool"
    if m == "rayon":
        return "viscose"
    if m == "flax":
        return "linen"
    if m in {"tencel", "tencel lyocell"}:
        return "lyocell"
    if m == "tencel modal":
        return "modal"
    if m == "naia":
        return "acetate"
    if m == "supima":
        return "cotton"
    if m in {"pet", "pes", "repreve"}:
        return "polyester"
    if m == "pp":
        return "polypropylene"

    return m


rows = []
summary_rows = []
any_barrier_rows = []
diag_material_rows = []
diag_category_rows = []
brand_counter = Counter()
parent_counter = Counter()

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(ROW_JSONL_OUT, "w", encoding="utf-8") as fout:
    for line_no, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        rec = json.loads(line)
        brand_counter[str(rec.get("brand"))] += 1
        parent_counter[str(rec.get("parent_category"))] += 1

        out = {
            "row_id": line_no,
            "brand": rec.get("brand"),
            "region": rec.get("region"),
            "parent_product_id": rec.get("parent_product_id"),
            "variant_colour": rec.get("variant_colour"),
            "parent_category": rec.get("parent_category"),
            "detail_category": rec.get("detail_category"),
        }

        comps = rec.get("components_structured") or []
        if not isinstance(comps, list):
            comps = []

        # =====================================================
        # Build one surface/main composition once for S1-S4
        # First choice: first surface_component in original order
        # Fallback: first component in original order
        # Do NOT merge multiple surface components or all components together
        # =====================================================
        selected_comp = None
        surface_ref_component_name = ""
        surface_ref_source = ""

        for comp in comps:
            cclass = str(comp.get("component_class") or "").strip().lower()
            if cclass == "surface_component":
                selected_comp = comp
                surface_ref_component_name = str(comp.get("component_name_norm") or "").strip().lower()
                surface_ref_source = "surface_component"
                break

        if selected_comp is None and comps:
            selected_comp = comps[0]
            surface_ref_component_name = str(comps[0].get("component_name_norm") or "").strip().lower()
            surface_ref_source = "first_component_fallback"

        picked_materials = selected_comp.get("materials") if isinstance(selected_comp, dict) else []
        if not isinstance(picked_materials, list):
            picked_materials = []

        comp_dict = defaultdict(float)
        for m in picked_materials:
            mat = canon_material(m.get("material"))
            pct = m.get("pct")
            if mat is None:
                continue
            try:
                pct_val = float(pct)
            except Exception:
                continue
            comp_dict[mat] += pct_val

        comp_dict = dict(comp_dict)
        comp_sorted = sorted(comp_dict.items(), key=lambda x: (-x[1], x[0]))
        mats_set = {m for m, pct in comp_dict.items() if pct > 0}

        out["n_distinct_fibres"] = len(comp_dict)
        out["composition_norm"] = json.dumps(comp_dict, ensure_ascii=False, sort_keys=True)
        out["composition_norm_sum"] = sum(comp_dict.values()) if comp_dict else 0.0
        out["surface_ref_component_name"] = surface_ref_component_name
        out["surface_ref_source"] = surface_ref_source

        # =====================================================
        # S1 inline + diagnostics
        # Diagnostic = unsupported mono or unsupported combination signature
        # =====================================================
        for setting in ["conservative", "central", "expanded"]:
            if not mats_set:
                out[f"s1_violate_{setting}"] = True
                out[f"s1_signature_{setting}"] = "no_material"
                s1_diag_signature = "no_material"

            elif len(mats_set) == 1:
                mono = next(iter(mats_set))
                violate = mono not in S1_MONO[setting]
                out[f"s1_violate_{setting}"] = violate
                out[f"s1_signature_{setting}"] = mono
                s1_diag_signature = mono

            elif len(mats_set) == 2:
                pair = frozenset(mats_set)
                sig = "+".join(sorted(mats_set))
                violate = pair not in S1_BINARY[setting]
                out[f"s1_violate_{setting}"] = violate
                out[f"s1_signature_{setting}"] = sig
                s1_diag_signature = sig

            else:
                sig = "+".join(sorted(mats_set))
                out[f"s1_violate_{setting}"] = True
                out[f"s1_signature_{setting}"] = sig
                s1_diag_signature = sig

            if out[f"s1_violate_{setting}"]:
                diag_material_rows.append({
                    "rule": "S1",
                    "setting": setting,
                    "row_id": line_no,
                    "material_diag": s1_diag_signature,
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })
                diag_category_rows.append({
                    "rule": "S1",
                    "setting": setting,
                    "row_id": line_no,
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })

        # =====================================================
        # S2 inline + diagnostics
        # Diagnostic = full unordered fibre combination
        # =====================================================
        out["s2_violate"] = len(mats_set) >= 3
        out["s2_signature"] = "+".join(sorted(mats_set)) if mats_set else "no_material"

        if out["s2_violate"]:
            diag_material_rows.append({
                "rule": "S2",
                "setting": "fixed",
                "row_id": line_no,
                "material_diag": out["s2_signature"],
                "parent_category": rec.get("parent_category"),
                "detail_category": rec.get("detail_category"),
            })
            diag_category_rows.append({
                "rule": "S2",
                "setting": "fixed",
                "row_id": line_no,
                "parent_category": rec.get("parent_category"),
                "detail_category": rec.get("detail_category"),
            })

        # =====================================================
        # S3 inline + diagnostics
        # Diagnostic = each violating minor material counted independently
        # =====================================================
        for setting in ["conservative", "central", "expanded"]:
            threshold = S3_THRESHOLDS[setting]
            violating_minor_mats = []

            if len(comp_sorted) <= 1:
                out[f"s3_violate_{setting}"] = False
                out[f"s3_signature_{setting}"] = ""
            else:
                for mat, pct in comp_sorted[1:]:
                    if pct < threshold:
                        violating_minor_mats.append(mat)

                out[f"s3_violate_{setting}"] = len(violating_minor_mats) > 0
                out[f"s3_signature_{setting}"] = "+".join(sorted(set(violating_minor_mats)))

            if out[f"s3_violate_{setting}"]:
                for mat in sorted(set(violating_minor_mats)):
                    diag_material_rows.append({
                        "rule": "S3",
                        "setting": setting,
                        "row_id": line_no,
                        "material_diag": mat,
                        "parent_category": rec.get("parent_category"),
                        "detail_category": rec.get("detail_category"),
                    })
                diag_category_rows.append({
                    "rule": "S3",
                    "setting": setting,
                    "row_id": line_no,
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })

        # =====================================================
        # S4 inline + diagnostics
        # Diagnostic = colour label itself
        # =====================================================
        colour = str(rec.get("variant_colour") or "").strip().lower()
        out["s4_violate_conservative"] = (colour == "black")
        out["s4_violate_central"] = (colour == "black")
        out["s4_violate_expanded"] = ("black" in colour)
        out["s4_signature_conservative"] = colour if out["s4_violate_conservative"] else ""
        out["s4_signature_central"] = colour if out["s4_violate_central"] else ""
        out["s4_signature_expanded"] = colour if out["s4_violate_expanded"] else ""

        for setting in ["conservative", "central", "expanded"]:
            if out[f"s4_violate_{setting}"]:
                diag_material_rows.append({
                    "rule": "S4",
                    "setting": setting,
                    "row_id": line_no,
                    "material_diag": colour,
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })
                diag_category_rows.append({
                    "rule": "S4",
                    "setting": setting,
                    "row_id": line_no,
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })

        # =====================================================
        # S5 inline (revised)
        # choose one surface reference: coating first, otherwise first surface_component,
        # otherwise first component overall
        # then compare that against hidden lining/filling structures
        # =====================================================
        surface_ref_name = ""
        surface_ref_dict = {}
        surface_ref_sorted = []

        # first: coating as outermost readable surface
        for comp in comps:
            cclass = str(comp.get("component_class") or "").strip().lower()
            cname = str(comp.get("component_name_norm") or "").strip().lower()
            if cclass == "surface_component" and cname == "coating":
                tmp = defaultdict(float)
                for m in comp.get("materials") or []:
                    mat = canon_material(m.get("material"))
                    pct = m.get("pct")
                    if mat is None:
                        continue
                    try:
                        pct_val = float(pct)
                    except Exception:
                        continue
                    tmp[mat] += pct_val
                tmp = dict(tmp)
                if tmp:
                    surface_ref_name = "coating"
                    surface_ref_dict = tmp
                    surface_ref_sorted = sorted(tmp.items(), key=lambda x: (-x[1], x[0]))
                    break

        # second: first surface_component in original order
        if not surface_ref_dict:
            for comp in comps:
                cclass = str(comp.get("component_class") or "").strip().lower()
                cname = str(comp.get("component_name_norm") or "").strip().lower()
                if cclass == "surface_component":
                    tmp = defaultdict(float)
                    for m in comp.get("materials") or []:
                        mat = canon_material(m.get("material"))
                        pct = m.get("pct")
                        if mat is None:
                            continue
                        try:
                            pct_val = float(pct)
                        except Exception:
                            continue
                        tmp[mat] += pct_val
                    tmp = dict(tmp)
                    if tmp:
                        surface_ref_name = cname
                        surface_ref_dict = tmp
                        surface_ref_sorted = sorted(tmp.items(), key=lambda x: (-x[1], x[0]))
                        break

        # third: first component overall as fallback
        if not surface_ref_dict and comps:
            comp = comps[0]
            cname = str(comp.get("component_name_norm") or "").strip().lower()
            tmp = defaultdict(float)
            for m in comp.get("materials") or []:
                mat = canon_material(m.get("material"))
                pct = m.get("pct")
                if mat is None:
                    continue
                try:
                    pct_val = float(pct)
                except Exception:
                    continue
                tmp[mat] += pct_val
            tmp = dict(tmp)
            if tmp:
                surface_ref_name = cname
                surface_ref_dict = tmp
                surface_ref_sorted = sorted(tmp.items(), key=lambda x: (-x[1], x[0]))

        surface_dom = surface_ref_sorted[0][0] if surface_ref_sorted else None

        hidden = []
        for comp in comps:
            cclass = str(comp.get("component_class") or "").strip().lower()
            cname = str(comp.get("component_name_norm") or "").strip().lower()
            if cclass in HIDDEN_COMPONENT_CLASSES or cname in HIDDEN_COMPONENT_NAMES:
                hidden.append(comp)

        # conservative: dominant hidden material differs from dominant surface material
        violate = False
        s5_sig = ""
        for comp in hidden:
            hidden_name = str(comp.get("component_name_norm") or "").strip().lower()
            hidden_dict = defaultdict(float)
            for m in comp.get("materials") or []:
                mat = canon_material(m.get("material"))
                pct = m.get("pct")
                if mat is None:
                    continue
                try:
                    pct_val = float(pct)
                except Exception:
                    continue
                hidden_dict[mat] += pct_val
            hidden_dict = dict(hidden_dict)
            if hidden_dict:
                hidden_sorted = sorted(hidden_dict.items(), key=lambda x: (-x[1], x[0]))
                hidden_dom = hidden_sorted[0][0]
                if surface_dom is not None and hidden_dom != surface_dom:
                    violate = True
                    s5_sig = f"{surface_ref_name}[{surface_dom}]__{hidden_name}[{hidden_dom}]"
                    break
        out["s5_violate_conservative"] = violate
        out["s5_signature_conservative"] = s5_sig

        # central: hidden layer contains >5% material absent from surface reference
        violate = False
        s5_sig = ""
        for comp in hidden:
            hidden_name = str(comp.get("component_name_norm") or "").strip().lower()
            hidden_dict = defaultdict(float)
            for m in comp.get("materials") or []:
                mat = canon_material(m.get("material"))
                pct = m.get("pct")
                if mat is None:
                    continue
                try:
                    pct_val = float(pct)
                except Exception:
                    continue
                hidden_dict[mat] += pct_val
            hidden_dict = dict(hidden_dict)
            for mat, pct in hidden_dict.items():
                if pct > 5 and mat not in surface_ref_dict:
                    violate = True
                    s5_sig = f"{surface_ref_name}[{'+'.join(sorted(surface_ref_dict.keys()))}]__{hidden_name}[{mat}]"
                    break
            if violate:
                break
        out["s5_violate_central"] = violate
        out["s5_signature_central"] = s5_sig

        # expanded: any hidden layer counts
        out["s5_violate_expanded"] = bool(hidden)
        out["s5_signature_expanded"] = f"{surface_ref_name}__{str(hidden[0].get('component_name_norm') or '').strip().lower()}" if hidden else ""

        for setting in ["conservative", "central", "expanded"]:
            if out[f"s5_violate_{setting}"]:
                diag_material_rows.append({
                    "rule": "S5",
                    "setting": setting,
                    "row_id": line_no,
                    "material_diag": out[f"s5_signature_{setting}"],
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })
                diag_category_rows.append({
                    "rule": "S5",
                    "setting": setting,
                    "row_id": line_no,
                    "parent_category": rec.get("parent_category"),
                    "detail_category": rec.get("detail_category"),
                })

        # =====================================================
        # Any-barrier bundle
        # =====================================================
        out["any_barrier_conservative"] = any([
            out["s1_violate_conservative"],
            out["s2_violate"],
            out["s3_violate_conservative"],
            out["s4_violate_conservative"],
            out["s5_violate_conservative"],
        ])

        out["any_barrier_central"] = any([
            out["s1_violate_central"],
            out["s2_violate"],
            out["s3_violate_central"],
            out["s4_violate_central"],
            out["s5_violate_central"],
        ])

        out["any_barrier_expanded"] = any([
            out["s1_violate_expanded"],
            out["s2_violate"],
            out["s3_violate_expanded"],
            out["s4_violate_expanded"],
            out["s5_violate_expanded"],
        ])

        rows.append(out)
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")


# =========================================================
# Row-level outputs
# =========================================================

df = pd.DataFrame(rows)
df.to_csv(ROW_CSV_OUT, index=False, encoding="utf-8-sig")

n_total = len(df)


# =========================================================
# Summary outputs
# =========================================================

summary_specs = [
    ("S1", "Restricted library", "s1_violate_conservative"),
    ("S1", "Baseline library", "s1_violate_central"),
    ("S1", "Extended library", "s1_violate_expanded"),
    ("S2", "Fixed", "s2_violate"),
    ("S3", "10% threshold", "s3_violate_conservative"),
    ("S3", "5% threshold", "s3_violate_central"),
    ("S3", "3% threshold", "s3_violate_expanded"),
    ("S4", "Exact black", "s4_violate_central"),
    ("S4", "Contains 'black'", "s4_violate_expanded"),
    ("S5", "Dominant-material mismatch", "s5_violate_conservative"),
    ("S5", ">5% concealed-material mismatch", "s5_violate_central"),
    ("S5", "Any hidden layer", "s5_violate_expanded"),
]

for rule_id, setting, col in summary_specs:
    n_violate = int(df[col].sum())
    n_pass = int(n_total - n_violate)
    summary_rows.append({
        "rule": rule_id,
        "setting": setting,
        "n_total": n_total,
        "n_violate": n_violate,
        "share_violate": n_violate / n_total if n_total else None,
        "n_pass": n_pass,
        "share_pass": n_pass / n_total if n_total else None,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV_OUT, index=False, encoding="utf-8-sig")

# Any-barrier is reported as one baseline share plus direct lower/upper boundary cases.
# No separate combination file is exported.

# Baseline:
# S1 = Baseline library; S2 = Fixed; S3 = 5% threshold;
# S4 = Exact black; S5 = >5% concealed-material mismatch.
any_baseline = df[[
    "s1_violate_central",
    "s2_violate",
    "s3_violate_central",
    "s4_violate_central",
    "s5_violate_central",
]].any(axis=1)

# Lower-bound candidate 1:
# S1 = Extended library; S2 = Fixed; S3 = 3% threshold;
# S4 = Exact black; S5 = Dominant-material mismatch.
any_lower_s5_conservative = df[[
    "s1_violate_expanded",
    "s2_violate",
    "s3_violate_expanded",
    "s4_violate_central",
    "s5_violate_conservative",
]].any(axis=1)

# Lower-bound candidate 2:
# S1 = Extended library; S2 = Fixed; S3 = 3% threshold;
# S4 = Exact black; S5 = >5% concealed-material mismatch.
# S5 conservative and central are not perfectly nested, so choose the lower aggregate result.
any_lower_s5_central = df[[
    "s1_violate_expanded",
    "s2_violate",
    "s3_violate_expanded",
    "s4_violate_central",
    "s5_violate_central",
]].any(axis=1)

if int(any_lower_s5_conservative.sum()) <= int(any_lower_s5_central.sum()):
    any_lower = any_lower_s5_conservative
    lower_s5_label = "Dominant-material mismatch"
else:
    any_lower = any_lower_s5_central
    lower_s5_label = ">5% concealed-material mismatch"

# Upper-bound:
# S1 = Restricted library; S2 = Fixed; S3 = 10% threshold;
# S4 = Contains 'black'; S5 = Any hidden layer.
any_upper = df[[
    "s1_violate_conservative",
    "s2_violate",
    "s3_violate_conservative",
    "s4_violate_expanded",
    "s5_violate_expanded",
]].any(axis=1)

n_barrier_baseline = int(any_baseline.sum())
share_barrier_baseline = n_barrier_baseline / n_total if n_total else None

n_barrier_lower = int(any_lower.sum())
share_min = n_barrier_lower / n_total if n_total else None

n_barrier_upper = int(any_upper.sum())
share_max = n_barrier_upper / n_total if n_total else None

any_barrier_rows.append({
    "summary_type": "Lower-bound any-barrier",
    "s1_scenario": "Extended library",
    "s2_scenario": "Fixed",
    "s3_scenario": "3% threshold",
    "s4_scenario": "Exact black",
    "s5_scenario": lower_s5_label,
    "n_total": n_total,
    "n_with_any_barrier": n_barrier_lower,
    "share_with_any_barrier": share_min,
    "n_with_no_barrier": n_total - n_barrier_lower,
    "share_with_no_barrier": (n_total - n_barrier_lower) / n_total if n_total else None,
})

any_barrier_rows.append({
    "summary_type": "Baseline any-barrier",
    "s1_scenario": "Baseline library",
    "s2_scenario": "Fixed",
    "s3_scenario": "5% threshold",
    "s4_scenario": "Exact black",
    "s5_scenario": ">5% concealed-material mismatch",
    "n_total": n_total,
    "n_with_any_barrier": n_barrier_baseline,
    "share_with_any_barrier": share_barrier_baseline,
    "n_with_no_barrier": n_total - n_barrier_baseline,
    "share_with_no_barrier": (n_total - n_barrier_baseline) / n_total if n_total else None,
})

any_barrier_rows.append({
    "summary_type": "Upper-bound any-barrier",
    "s1_scenario": "Restricted library",
    "s2_scenario": "Fixed",
    "s3_scenario": "10% threshold",
    "s4_scenario": "Contains 'black'",
    "s5_scenario": "Any hidden layer",
    "n_total": n_total,
    "n_with_any_barrier": n_barrier_upper,
    "share_with_any_barrier": share_max,
    "n_with_no_barrier": n_total - n_barrier_upper,
    "share_with_no_barrier": (n_total - n_barrier_upper) / n_total if n_total else None,
})

any_barrier_df = pd.DataFrame(any_barrier_rows)
any_barrier_df.to_csv(ANY_BARRIER_CSV_OUT, index=False, encoding="utf-8-sig")


# =========================================================
# Diagnostic outputs
# material-side and category-side for every rule
# =========================================================

diag_material_df = pd.DataFrame(diag_material_rows)
if not diag_material_df.empty:
    diag_material_df["scenario"] = diag_material_df.apply(
        lambda r: {
            ("S1", "conservative"): "Restricted library",
            ("S1", "central"): "Baseline library",
            ("S1", "expanded"): "Extended library",
            ("S2", "fixed"): "Fixed",
            ("S3", "conservative"): "10% threshold",
            ("S3", "central"): "5% threshold",
            ("S3", "expanded"): "3% threshold",
            ("S4", "central"): "Exact black",
            ("S4", "expanded"): "Contains 'black'",
            ("S5", "conservative"): "Dominant-material mismatch",
            ("S5", "central"): ">5% concealed-material mismatch",
            ("S5", "expanded"): "Any hidden layer",
        }.get((r["rule"], r["setting"])),
        axis=1,
    )
    diag_material_df = diag_material_df[diag_material_df["scenario"].notna()]
    diag_material_summary = (
        diag_material_df
        .groupby(["rule", "scenario", "material_diag"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["rule", "scenario", "n", "material_diag"], ascending=[True, True, False, True])
    )
else:
    diag_material_summary = pd.DataFrame(columns=["rule", "scenario", "material_diag", "n"])

diag_material_summary.to_csv(DIAG_MATERIAL_CSV_OUT, index=False, encoding="utf-8-sig")

diag_category_df = pd.DataFrame(diag_category_rows)
if not diag_category_df.empty:
    diag_category_df["scenario"] = diag_category_df.apply(
        lambda r: {
            ("S1", "conservative"): "Restricted library",
            ("S1", "central"): "Baseline library",
            ("S1", "expanded"): "Extended library",
            ("S2", "fixed"): "Fixed",
            ("S3", "conservative"): "10% threshold",
            ("S3", "central"): "5% threshold",
            ("S3", "expanded"): "3% threshold",
            ("S4", "central"): "Exact black",
            ("S4", "expanded"): "Contains 'black'",
            ("S5", "conservative"): "Dominant-material mismatch",
            ("S5", "central"): ">5% concealed-material mismatch",
            ("S5", "expanded"): "Any hidden layer",
        }.get((r["rule"], r["setting"])),
        axis=1,
    )
    diag_category_df = diag_category_df[diag_category_df["scenario"].notna()]
    diag_category_summary = (
        diag_category_df
        .groupby(["rule", "scenario", "parent_category", "detail_category"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["rule", "scenario", "n", "parent_category", "detail_category"], ascending=[True, True, False, True, True])
    )
else:
    diag_category_summary = pd.DataFrame(columns=["rule", "scenario", "parent_category", "detail_category", "n"])

diag_category_summary.to_csv(DIAG_CATEGORY_CSV_OUT, index=False, encoding="utf-8-sig")


# =========================================================
# Summary text
# =========================================================

with open(SUMMARY_TXT_OUT, "w", encoding="utf-8") as f:
    f.write("Rule evaluation summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Total evaluated rows: {n_total}\n\n")

    f.write("Brand counts\n")
    f.write("-" * 60 + "\n")
    for k, v in sorted(brand_counter.items()):
        f.write(f"{k}: {v}\n")
    f.write("\n")

    f.write("Parent category counts\n")
    f.write("-" * 60 + "\n")
    for k, v in sorted(parent_counter.items()):
        f.write(f"{k}: {v}\n")
    f.write("\n")

    f.write("Rule violation summary\n")
    f.write("-" * 60 + "\n")
    for _, row in summary_df.iterrows():
        f.write(
            f"{row['rule']} | {row['setting']} | "
            f"n_violate={int(row['n_violate'])} | "
            f"share_violate={row['share_violate']:.6f}\n"
        )
    f.write("\n")

    f.write("Any-barrier summary\n")
    f.write("-" * 60 + "\n")
    f.write(
        f"Lower-bound any-barrier | n_with_any_barrier={n_barrier_lower} | "
        f"share_with_any_barrier={share_min:.6f}\n"
    )
    f.write(
        "Lower-bound conditions | S1=Extended library | S2=Fixed | "
        f"S3=3% threshold | S4=Exact black | S5={lower_s5_label}\n"
    )
    f.write(
        f"Baseline any-barrier | n_with_any_barrier={n_barrier_baseline} | "
        f"share_with_any_barrier={share_barrier_baseline:.6f}\n"
    )
    f.write(
        "Baseline conditions | S1=Baseline library | S2=Fixed | "
        "S3=5% threshold | S4=Exact black | S5=>5% concealed-material mismatch\n"
    )
    f.write(
        f"Upper-bound any-barrier | n_with_any_barrier={n_barrier_upper} | "
        f"share_with_any_barrier={share_max:.6f}\n"
    )
    f.write(
        "Upper-bound conditions | S1=Restricted library | S2=Fixed | "
        "S3=10% threshold | S4=Contains 'black' | S5=Any hidden layer\n"
    )

print("Done.")
print(f"Row-level JSONL: {ROW_JSONL_OUT}")
print(f"Row-level CSV:   {ROW_CSV_OUT}")
print(f"Rule summary:    {SUMMARY_CSV_OUT}")
print(f"Any barrier:     {ANY_BARRIER_CSV_OUT}")
print(f"Material diag:   {DIAG_MATERIAL_CSV_OUT}")
print(f"Category diag:   {DIAG_CATEGORY_CSV_OUT}")
print(f"Summary txt:     {SUMMARY_TXT_OUT}")
