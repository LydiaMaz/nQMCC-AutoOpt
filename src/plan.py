"""
plan.py
Optimization plan: which correlations each phase frees and their step sizes
"""
#-----------------------------------------------------------------------
import os
import tomllib
from fnmatch import fnmatchcase
#-----------------------------------------------------------------------
from deck import MAIN_KEYS, SS_KEYS
#-----------------------------------------------------------------------
# A plan file (TOML) has
#   [scan]        ESEP_factor = [...]   multiplies the starting deck's ESEP (optional)
#   [groups.NAME] KEY = {rel = r, abs = a}   step = max(r*|value|, a)
#   [[phase]]     name, params = [group names], blocks = first|rest|all
# blocks picks the spatial-symmetry blocks a phase's SS keys apply to.
#-----------------------------------------------------------------------
BLOCK_CHOICES = ["first", "rest", "all"]
#-----------------------------------------------------------------------
def MatchKeys(pattern, keys):
    return [k for k in keys if fnmatchcase(k, pattern)]
#-----------------------------------------------------------------------
class phase_t:
#-----------------------------------------------------------------------
    def __init__(self, data, plan):
        unknown = set(data) - {"name", "params", "blocks"}
        if unknown:
            raise ValueError(f"{plan.FILE_NAME}: phase {data.get('name')}: unknown fields {sorted(unknown)}")
        self.NAME   = data["name"]
        self.GROUPS = list(data["params"])
        self.BLOCKS = data.get("blocks")
        if self.BLOCKS is not None and self.BLOCKS not in BLOCK_CHOICES:
            raise ValueError(f"{plan.FILE_NAME}: phase {self.NAME}: blocks must be one of {BLOCK_CHOICES}")
        steps = plan.Steps(self.GROUPS, where=f"phase {self.NAME}")
        if any(MatchKeys(k, SS_KEYS) for k in steps) and self.BLOCKS is None:
            raise ValueError(f"{plan.FILE_NAME}: phase {self.NAME}: SS keys need a blocks setting")
        self.PLAN = plan
#-----------------------------------------------------------------------
    def SSIndices(self, n_ss):
        match self.BLOCKS:
            case "first": return [0] if n_ss > 0 else []
            case "rest":  return list(range(1, n_ss))
            case "all":   return list(range(n_ss))
            case _:       return []
#-----------------------------------------------------------------------
    def Instructions(self, dk):
        return self.PLAN.Instructions(self.GROUPS, dk, self.SSIndices(len(dk.SS)))
#-----------------------------------------------------------------------
class plan_t:
#-----------------------------------------------------------------------
    def __init__(self, file_name):
        self.FILE_NAME = file_name
        with open(file_name, "rb") as f:
            data = tomllib.load(f)
        self.DATA = data
        unknown = set(data) - {"scan", "groups", "phase"}
        if unknown:
            raise ValueError(f"{file_name}: unknown sections {sorted(unknown)}")
#-----------------------------------------------------------------------
        scan = data.get("scan", {})
        if set(scan) - {"ESEP_factor"}:
            raise ValueError(f"{file_name}: [scan] only supports ESEP_factor")
        self.ESEP_FACTORS = [float(f) for f in scan.get("ESEP_factor", [1.0])]
#-----------------------------------------------------------------------
        self.GROUPS = {}
        for name, keys in data.get("groups", {}).items():
            self.GROUPS[name] = {}
            for key, step in keys.items():
                if not MatchKeys(key, MAIN_KEYS + SS_KEYS):
                    raise ValueError(f"{file_name}: group {name}: {key} is not an optimizable deck key")
                if not isinstance(step, dict) or set(step) - {"rel", "abs"}:
                    raise ValueError(f"{file_name}: group {name}: {key} needs {{rel = .., abs = ..}}")
                self.GROUPS[name][key] = (float(step.get("rel", 0.0)), float(step.get("abs", 0.0)))
#-----------------------------------------------------------------------
        self.PHASES = [phase_t(p, self) for p in data.get("phase", [])]
#-----------------------------------------------------------------------
    def Steps(self, group_names, where=""):
        """Merge groups into {key pattern: (rel, abs)}; a key may come from one group only."""
        steps = {}
        for g in group_names:
            if g not in self.GROUPS:
                raise ValueError(f"{self.FILE_NAME}: {where}: unknown group '{g}'")
            for key, step in self.GROUPS[g].items():
                if key in steps:
                    raise ValueError(f"{self.FILE_NAME}: {where}: {key} set by two groups")
                steps[key] = step
        return steps
#-----------------------------------------------------------------------
    def Instructions(self, group_names, dk, ss_indices):
        """GenerateOptFile instructions for these groups on deck dk.
        Wildcard keys (e.g. "Q*") expand to the keys dk has; SS keys go to ss_indices."""
        inst = []
        for pattern, (rel, abs_) in self.Steps(group_names).items():
            for k in MatchKeys(pattern, MAIN_KEYS):
                if k == pattern or hasattr(dk, k):
                    inst.append({"key": k, "rel": rel, "abs": abs_})
            for ssi in ss_indices:
                for k in MatchKeys(pattern, SS_KEYS):
                    if k == pattern or hasattr(dk.SS[ssi], k):
                        inst.append({"ss_idx": ssi, "key": k, "rel": rel, "abs": abs_})
        return inst
#-----------------------------------------------------------------------
def PlanPath(path):
    """Relative plan paths are resolved from the AutoOpt repo root."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
#-----------------------------------------------------------------------
