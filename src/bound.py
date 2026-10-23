#!/usr/bin/env python3
"""
Bound-state optimizer driven by a plan file (see plan.py).

For each ESEP factor in the plan's scan, the starting deck's ESEP is
multiplied by the factor and the plan's phases run in order, each
building on the best deck so far. A phase is accepted only if it lowers
the energy; phases whose SS blocks select nothing (e.g. s-shell decks)
are skipped.

The best result across all ESEP factors is kept as the final deck.
If nothing improves the energy the original deck is preserved.

If ALPHA_DIR is set in the utility file, the run starts from a deck
seeded with optimized He-4 alpha-core parameters. Otherwise it uses
the ctrl-file default deck.
"""
import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from contextlib import redirect_stdout
from io import StringIO

sys.path.append(os.path.dirname(__file__))

from deck import deck_t, GenerateOptFile
from wavefunction import wavefunction_t, InitNShellBoundWF
from parameters import parameters_t


# ---------------------------------------------------------------------------
# Shared utility helpers
# ---------------------------------------------------------------------------

def is_finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False

def strip_quotes(s):
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def quote_path(path):
    return f"'{path}'"

def ensure_dir(p):
    if p:
        Path(p).mkdir(parents=True, exist_ok=True)

def create_pair_name(pot2b, pot3b):
    return f"{pot2b.split('.')[0]}.{pot3b.split('.')[0]}"

def esep_tag(factor):
    return f"esep{factor:.2f}"

def write_esep_scaled_deck(target, src, factor, out):
    """Copy deck src to out with ESEP multiplied by factor (factor 1.0 returns src)."""
    if factor == 1.0:
        return src
    load_dk_unquoted(target, src)
    target.DK.ESEP = [f"{float(v)*factor:.5f}" for v in target.DK.ESEP]
    target.DK.Write(target.PARAMS, quote_path(out))
    return out

def load_dk_unquoted(target, path):
    p = strip_quotes(path)
    target.DK.FILE_NAME = p
    target.DK.Read(target.PARAMS)
    target.CTRL.INPUT_BRA.DECK_FILE = quote_path(p)


# ---------------------------------------------------------------------------
# Directory and target setup
# ---------------------------------------------------------------------------

def apply_calc_overrides(target, util):
    """Override calc_t fields with per-nucleus values from util."""
    target.CALC.NUM_OPT_EVALUATIONS  = str(util.NUM_OPT_EVALUATIONS)
    target.CALC.CFE                  = util.CFE
    target.CALC.CFV                  = util.CFV
    target.CALC.CFR                  = util.CFR
    target.CALC.NEUTRON_RADIUS_LIMIT = util.NEUTRON_RADIUS_LIMIT
    target.CALC.PROTON_RADIUS_LIMIT  = util.PROTON_RADIUS_LIMIT

def build_target(util, pot_dir, pot_index):
    """Build a fresh wavefunction_t for the given potential pair."""
    target = wavefunction_t(util.CTRL_FILE, util.NQMCC_DIR, util.BIN_DIR, util.RUN_CMD)
    target.CTRL.FILE_NAME          = f"{pot_dir}ctrl/target.ctrl"
    target.CTRL.CALC_TYPE          = util.CALC_TYPE
    target.CTRL.CALC_FILE          = quote_path(f"{pot_dir}ctrl/target.calc")
    target.SetCalc(util.CALC_TYPE, f"{util.NQMCC_DIR}ctrl/calc/{util.CALC_FILE}")
    apply_calc_overrides(target, util)
    target.CTRL.NUM_BLOCKS         = util.NUM_BLOCKS
    target.CTRL.BLOCK_SIZE         = util.BLOCK_SIZE
    target.CTRL.NUM_WALKERS_PER_NODE = util.WALKERS_PER_NODE
    ensure_dir(f"{pot_dir}scratch")
    target.CTRL.SCRATCH_DIR        = quote_path(f"{pot_dir}scratch/")

    target.CTRL.CONST_FILE = quote_path(f"{util.NQMCC_DIR}constants/{util.CONSTANTS_FILES[pot_index]}")
    target.CTRL.L2BP_FILE  = quote_path(f"{util.NQMCC_DIR}pots/{util.TWO_BODY_FILES[pot_index]}")
    target.CTRL.L3BP_FILE  = quote_path(f"{util.NQMCC_DIR}pots/{util.THREE_BODY_FILES[pot_index]}")
    return target


# ---------------------------------------------------------------------------
# Alpha seeding
# ---------------------------------------------------------------------------

def find_alpha_core(alpha_cores_dir, pot_pair_name):
    deck_name = f"{pot_pair_name}_optimized.dk"
    path = os.path.join(alpha_cores_dir, pot_pair_name, "dk", deck_name)
    return path if os.path.exists(path) else None

def generate_alpha_seeded_deck(target, pot_pair_name, output_path, alpha_cores_dir, he4_params_path, copy_esep=True):
    """
    Build a starting deck by copying optimized He-4 alpha-core parameters
    into the target deck.
    """
    alpha_path = find_alpha_core(alpha_cores_dir, pot_pair_name)
    he4_params = parameters_t(he4_params_path)
    alpha_deck = deck_t(he4_params, alpha_path)

    class MinimalAlpha:
        def __init__(self, dk):
            self.DK = dk

    alpha = MinimalAlpha(alpha_deck)
    ensure_dir(os.path.dirname(output_path))
    InitNShellBoundWF(target, alpha, output_path, copy_esep=copy_esep)
    return output_path


# ---------------------------------------------------------------------------
# Core optimization call
# ---------------------------------------------------------------------------

def try_optimize(target, opt_file, dk_out, log_prefix, instructions, e_ref, require_improvement=True):
    """
    Run one optimization pass and return (e, v, ok, msg).

    Writes stdout from the optimizer to log_prefix.txt.
    Returns ok=False if the energy is non-finite, the output deck is missing,
    or (when require_improvement=True) the energy did not decrease.
    """
    ensure_dir(os.path.dirname(opt_file))
    ensure_dir(os.path.dirname(dk_out))
    ensure_dir(os.path.dirname(log_prefix))

    opt = GenerateOptFile(target.PARAMS, target.DK, quote_path(opt_file), instructions)
    opt.UpdateFloats(target.PARAMS, 6)

    log_txt = f"{log_prefix}.txt"
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            e, v = target.Optimize(opt, quote_path(dk_out), True, log_prefix)
    except Exception as ex:
        with open(log_txt, "w") as f:
            f.write(buf.getvalue())
            f.write(f"\nEXCEPTION: {ex}\n")
        return None, None, False, f"Optimize exception: {ex}"

    with open(log_txt, "w") as f:
        f.write(buf.getvalue())

    if not is_finite(e):
        return None, None, False, "Non-finite energy"
    if not os.path.exists(dk_out):
        return None, None, False, f"No DK output: {dk_out}"

    e = float(e)
    v = float(v) if is_finite(v) else float("nan")

    if require_improvement:
        dE = e - float(e_ref)
        if dE >= 0.0:
            return None, None, False, f"No improvement: ΔE={dE:.6f}"

    return e, v, True, ""


# ---------------------------------------------------------------------------
# Phase execution helper
# ---------------------------------------------------------------------------

def run_phase(target, phase_name, instructions, factor, tag,
              pot_dir, pair_name, attempts, e_ref):
    opt_file = f"{pot_dir}opt/{pair_name}_opt_{tag}_{phase_name}.opt"
    dk_out   = f"{pot_dir}dk/{pair_name}_opt_{tag}_{phase_name}.dk"
    log_pref = f"{pot_dir}logs/{pair_name}.opt_{tag}_{phase_name}"

    e, v, ok, msg = try_optimize(
        target, opt_file, dk_out, log_pref, instructions,
        e_ref, require_improvement=True,
    )

    attempts.append({
        "esep_factor": factor, "tag": tag, "phase": phase_name, "ok": ok,
        "energy": e if ok else None, "variance": v if ok else None,
        "dk": dk_out, "opt": opt_file, "msg": msg if not ok else "",
    })

    if ok:
        print(f"  ✅ {phase_name}: E={e:.6f}  ΔE={(e - e_ref):.6f}")
    else:
        print(f"  ⚠️  {phase_name} failed: {msg}")

    return ok, e, v, dk_out, opt_file


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def BoundStateOptimizePhases(util, pair_name, pot_dir, pot_index):
    print(f"{pair_name}: building target + optional alpha-seeding + initial Evaluate")
    print("=" * 72)

    # --- Build initial target and optionally seed from a He-4 alpha core ---
    target0 = build_target(util, pot_dir, pot_index)

    print(f"COST FACTORS (E VAR RADII): {util.CFE} {util.CFV} {util.CFR}")
    if float(util.CFR) != 0.0:
        print(f"NEUTRON_RADIUS_LIMIT: {util.NEUTRON_RADIUS_LIMIT}")
        print(f"PROTON_RADIUS_LIMIT:  {util.PROTON_RADIUS_LIMIT}")

    starting_deck_label = "ctrl-default"
    if getattr(util, "ALPHA_DIR", None):
        he4_params_path  = f"{util.NQMCC_DIR}nuclei/params/he4.params"
        seeded_deck_path = f"{pot_dir}dk/{pair_name}_alpha_seed.dk"
        try:
            generate_alpha_seeded_deck(
                target0, pair_name, seeded_deck_path, util.ALPHA_DIR, he4_params_path, copy_esep=True
            )
            target0.DK = deck_t(target0.PARAMS, seeded_deck_path)
            target0.CTRL.INPUT_BRA.DECK_FILE = quote_path(seeded_deck_path)
            starting_deck_label = "alpha-seeded"
            print(f"Alpha seeding successful: {seeded_deck_path}")
        except Exception as ex:
            print(f"Alpha seeding failed ({ex}), continuing with ctrl-default deck")

    # --- Initial evaluation to establish the baseline energy ---
    log_init = f"{pot_dir}logs/{pair_name}.initial"
    e0, v0 = target0.Evaluate(True, log_init)
    if not is_finite(e0):
        raise RuntimeError("Initial evaluation failed")
    e0 = float(e0)
    v0 = float(v0) if is_finite(v0) else float("nan")
    print(f"INITIAL ({starting_deck_label}): E={e0:.6f}  V={v0:.6f}")

    dk_backup = f"{pot_dir}dk/{pair_name}_original.dk"
    target0.DK.Write(target0.PARAMS, quote_path(dk_backup))
    print(f"Saved ORIGINAL DK: {dk_backup}")

    plan = util.PLAN
    n_ss = len(target0.DK.SS)
    if n_ss > 0:
        print(f"Spatial symmetries detected: {n_ss} SS blocks")
    else:
        print("No SS blocks: phases that need SS blocks are skipped")
    print(f"PLAN: {plan.FILE_NAME}")
    print(f"ESEP factors: {plan.ESEP_FACTORS}")

    attempts = []
    best = None

    # -----------------------------------------------------------------------
    # ESEP factor scan — run the plan's phases at each factor
    # -----------------------------------------------------------------------
    for factor in plan.ESEP_FACTORS:
        tag = esep_tag(factor)
        print(f"\n{pair_name}: ESEP factor={factor} (tag={tag})")

        target = build_target(util, pot_dir, pot_index)
        dk_start = write_esep_scaled_deck(target, dk_backup, factor, f"{pot_dir}dk/{pair_name}_start_{tag}.dk")
        load_dk_unquoted(target, dk_start)

        # Track the best energy achieved so far at this factor.
        # Each phase uses this as its improvement threshold.
        best_e_factor  = e0
        best_v_factor  = v0
        best_dk_factor = dk_start
        best_opt_factor = None

        def update_best(ok, e, v, dk, opt):
            """Update per-factor best state and reload target deck."""
            nonlocal best, best_e_factor, best_v_factor, best_dk_factor, best_opt_factor
            if ok:
                best_e_factor   = float(e)
                best_v_factor   = float(v) if is_finite(v) else float("nan")
                best_dk_factor  = dk
                best_opt_factor = opt
                if best is None or best_e_factor < best["energy"]:
                    best = {"factor": factor, "tag": tag, "energy": best_e_factor,
                            "variance": best_v_factor, "dk": best_dk_factor,
                            "opt": best_opt_factor}
            # Always reload — on success to advance, on failure to recover
            load_dk_unquoted(target, best_dk_factor)

        for phase in plan.PHASES:
            if phase.BLOCKS is not None and not phase.SSIndices(n_ss):
                continue
            inst = phase.Instructions(target.DK)
            ok, e, v, dk, opt = run_phase(target, phase.NAME, inst, factor, tag, pot_dir, pair_name, attempts, best_e_factor)
            update_best(ok, e, v, dk, opt)

    # --- Finalize: copy the best result to the canonical output paths ---
    final_dk  = f"{pot_dir}dk/{pair_name}_optimized.dk"
    final_opt = f"{pot_dir}opt/{pair_name}_opt.opt"

    if best is None:
        print("No successful improvements across any ESEP factor; keeping ORIGINAL deck")
        shutil.copy2(dk_backup, final_dk)
        best_e, best_v, best_factor = e0, v0, None
    else:
        shutil.copy2(best["dk"], final_dk)
        if best.get("opt") and os.path.exists(best["opt"]):
            shutil.copy2(best["opt"], final_opt)
        best_e, best_v, best_factor = best["energy"], best["variance"], best["factor"]
        print(f"\nBest ESEP factor={best_factor}  E={best_e:.6f}  ΔE={(best_e - e0):.6f}")

    result = {
        "POTENTIAL_PAIR":     pair_name,
        "STARTING_DECK_MODE": starting_deck_label,
        "ALPHA_CORES_DIR":    getattr(util, "ALPHA_DIR", None),
        "E_INITIAL":          e0,
        "V_INITIAL":          v0,
        "E_OPTIMIZED":        best_e,
        "V_OPTIMIZED":        best_v,
        "DELTA_E":            best_e - e0,
        "PLAN":               plan.FILE_NAME,
        "BEST_ESEP_FACTOR":   best_factor,
        "FINAL_DK":           final_dk,
        "FINAL_OPT":          final_opt,
        "DK_BACKUP":          dk_backup,
        "ATTEMPTS":           attempts,
    }
    with open(f"{pot_dir}{pair_name}_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

