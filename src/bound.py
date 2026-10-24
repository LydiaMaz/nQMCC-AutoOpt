#!/usr/bin/env python3
"""
Bound-state optimizer driven by a plan file (see plan.py).

The plan's phases run in order, each starting from the best deck so far,
and a phase is kept only if it lowers the energy. The first phase runs
once per ESEP step in ESEP_STEPS (ESEP step = s * |ESEP|, replacing the
plan's ESEP step) and keeps the lowest result; later phases use the
plan's steps. Phases whose SS blocks select nothing (e.g. s-shell decks)
are skipped. If nothing improves the energy the original deck is preserved.

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

# The first phase of every plan runs once per ESEP step (fraction of the
# deck's ESEP; 0 = ESEP fixed) and keeps the best.
ESEP_STEPS = [0.0, 0.25, 0.5]

def estep_tag(step):
    return f"estep{step:.2f}"

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

def run_phase(target, phase_name, instructions, esep_step, tag,
              pot_dir, pair_name, attempts, e_ref):
    opt_file = f"{pot_dir}opt/{pair_name}_opt_{tag}.opt"
    dk_out   = f"{pot_dir}dk/{pair_name}_opt_{tag}.dk"
    log_pref = f"{pot_dir}logs/{pair_name}.opt_{tag}"

    e, v, ok, msg = try_optimize(
        target, opt_file, dk_out, log_pref, instructions,
        e_ref, require_improvement=True,
    )

    attempts.append({
        "esep_step": esep_step, "tag": tag, "phase": phase_name, "ok": ok,
        "energy": e if ok else None, "variance": v if ok else None,
        "dk": dk_out, "opt": opt_file, "msg": msg if not ok else "",
    })

    if ok:
        print(f"  ✅ {tag}: E={e:.6f}  ΔE={(e - e_ref):.6f}")
    else:
        print(f"  ⚠️  {tag} failed: {msg}")

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

    attempts = []
    target = build_target(util, pot_dir, pot_index)
    cur = {"energy": e0, "variance": v0, "dk": dk_backup, "opt": None}
    best_step = None
    first = True

    for phase in plan.PHASES:
        if phase.BLOCKS is not None and not phase.SSIndices(n_ss):
            continue
        steps = ESEP_STEPS if first else [None]
        first = False
        print(f"\n{pair_name}: {phase.NAME}" + (f"  ESEP steps {steps}" if steps != [None] else ""))
        load_dk_unquoted(target, cur["dk"])
        base = phase.Instructions(target.DK)
        # one run per ESEP step (first phase), or a single run with the plan's steps
        best_run = None
        for step in steps:
            tag = phase.NAME if step is None else f"{phase.NAME}_{estep_tag(step)}"
            inst = base if step is None else \
                [i for i in base if i["key"] != "ESEP"] + [{"key": "ESEP", "rel": step, "abs": 0.0}]
            load_dk_unquoted(target, cur["dk"])
            ok, e, v, dk, opt = run_phase(target, phase.NAME, inst, step, tag,
                                          pot_dir, pair_name, attempts, cur["energy"])
            if ok and (best_run is None or e < best_run["energy"]):
                best_run = {"energy": float(e), "variance": float(v) if is_finite(v) else float("nan"),
                            "dk": dk, "opt": opt, "step": step}
        if best_run is not None:
            cur = best_run
            if best_run["step"] is not None:
                best_step = best_run["step"]
                print(f"  continuing from ESEP step {best_step} (E={cur['energy']:.6f})")
        # continue from the best deck so far
        load_dk_unquoted(target, cur["dk"])

    # --- Finalize: copy the best result to the canonical output paths ---
    final_dk  = f"{pot_dir}dk/{pair_name}_optimized.dk"
    final_opt = f"{pot_dir}opt/{pair_name}_opt.opt"

    if cur["opt"] is None:
        print("No phase improved the energy; keeping ORIGINAL deck")
        shutil.copy2(dk_backup, final_dk)
    else:
        shutil.copy2(cur["dk"], final_dk)
        if os.path.exists(cur["opt"]):
            shutil.copy2(cur["opt"], final_opt)
        print(f"\nFinal E={cur['energy']:.6f}  ΔE={(cur['energy'] - e0):.6f}  ESEP step={best_step}")
    best_e, best_v = cur["energy"], cur["variance"]

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
        "BEST_ESEP_STEP":     best_step,
        "FINAL_DK":           final_dk,
        "FINAL_OPT":          final_opt,
        "DK_BACKUP":          dk_backup,
        "ATTEMPTS":           attempts,
    }
    with open(f"{pot_dir}{pair_name}_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

