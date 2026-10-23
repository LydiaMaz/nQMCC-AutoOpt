# AutoOpt :: A Parameter Optimization Framework for nQMCC

This project automates the optimizations of variational parameters for nuclear Quantum Monte Carlo with Continuum (nQMCC).

## Maintainer
Lydia Mazeeva (WASHU)

## AutoOpt Contributors
Lydia Mazeeva (WASHU)\
Abraham R. Flores (WASHU)\
Juan Jose Silva (WASHU)\
Maria Piarulli (WASHU)

## Todo
1. Generate zero deck main program
2. Improve bounds checking in scan
3. Update logic for +1 node
4. Update Documentation
5. Automate tests

## Usage

To run a full optimization:

```bash
python /absolute/path/to/AutoOpt.py --utility /absolute/path/to/file.util
```
Requires Python ≥ 3.12 (nested-quote f-strings) and numpy. The ctrl, calc and deck readers follow the nQMCC input format with a `calc_type calc_file` ctrl line (nQMCC `add_d2f2` branch and later); see `examples/` for working util files.

> ✅ Make sure your `utility` file includes the following fields:
- `NAME`: label for outputs
- `SYSTEM_TYPE`: bound, sc_scattering
- `RUN_CMD`: prefix to run binary (mpirun -np X)
- `NQMCC_DIR`: /path/to/nQMCC
- `BIN_DIR`: /path/to/nQMCC/bin
- `WORKING_DIR`:/path/to/your_logs
- `CTRL_FILE`:/path/to/sys.ctrl
- `NUM_POTENTIALS`:number of potentials you want to loop over
- `CONSTANT_FILES`:constant file for each 2/3 body potential listed
- `TWO_BODY_POT_FILES`: list of 2 body files assumed in nQMCC/pots/
- `THREE_BODY_POT_FILES`: list of 3 body files assumed in nQMCC/pots/
- `NUM_BLOCKS BLOCK_SIZE WALKERS_PER_NODE`: Monte Carlo Sample information and topology (0 walkers per node unless Abe says otherwise)
- `NUM_OPT_EVALUATIONS`: number of evaluations in a single COBYLA optimization (every phase)
- `CALC_TYPE CALC_FILE`: nQMCC calc type (1 = optimize) and template file in `nQMCC/ctrl/calc/` (e.g. `1 opt.calc`); AutoOpt writes a per-run copy with its own overrides
- `PLAN`: plan file saying which correlations to optimize and their step sizes (see **Plan file**); relative paths are resolved from the AutoOpt directory, e.g. `plans/s_shell.toml`
> ⚠️ `IF SYSTEM_TYPE == bound`:
- `CFE CFV CFR NR_LIMIT PR_LIMIT`: optimizer cost = CFE·E + CFV·variance + CFR·(radius penalty above the neutron/proton radius limits); `CFR 0.0` turns the radius limits off
- `ALPHA_DIR` *(optional)*: path to directory containing pre-optimized He-4 alpha-core decks; enables alpha-core seeding for improved convergence. If not found the run continues with the ctrl-file default deck.

Optimization runs in two stages, using the phases from the plan:
1. For each ESEP factor in the plan's scan, the starting deck's ESEP is multiplied by the factor and the first phase runs.
2. The remaining phases run once, from the factor whose first phase gave the lowest energy below the starting energy (from the original deck if none did).

A phase is kept only if it lowers the energy; phases whose SS blocks select nothing (s-shell decks) are skipped. The lowest-energy deck is the final deck; if nothing improves the energy, the original deck is preserved. Output decks are tagged by ESEP factor, e.g. `{pair}_opt_esep1.10_p0_main.dk`.

Outputs include optimized nQMCC decks (`.dk`), per-pair results in `{pair}_results.json`, a combined summary in `combined_results.json`, and a full run log in `optimization_log.out`; all saved to the working directory.

> ⚠️ `IF SYSTEM_TYPE == sc_scattering`:
- `NUM_CHANNELS`: total number of channels to loop over
- `SCATTERING_CTRL_FILES`: list of /path/to/scattering.ctrl
- `SPATIAL_SYMMETRIES_INDEXS`: index of spatial symmetry
- `OPTIMIZE_TARGET`: Optimize the A-1 body (0: no  1: yes)
- `ENERGY_LOWER_BOUND ENERGY_UPPER_BOUND DELTA_ENERGY`: scan will terminate if out of bounds, delta_e is approximated by foward difference scheme
- `INITIAL_BSCAT INITIAL_DELTA_BSCAT MAX_BSCAT_SLOPE`: first bscat +- delta, used to calculate db/de
- `MAX_NUMBER_OF_DECKS_GENERATED`: scan will terminate if number of evals is greater or equal to

At each BSCAT point the channel's WSE is scanned first (fixed: 5 evaluations x 5 tries, step 0.5), then the plan's `corr` group is optimized without and with WSE.

Outputs include optimized nQMCC decks (`.dk`), logs of nQMCC calculations, and a summary of results with `BSCAT`, `EREL`, `VREL`, `WSE`, and others; saved in json format to the working directory.

## Plan file

A plan is a TOML file with the correlations to optimize. Each step is
`step = max(rel * |value|, abs)`: `rel` is a fraction of the parameter's current value, `abs` a fixed amount (needed for parameters that start at 0).

```toml
[scan]                          # bound only, optional (default: one run at factor 1.0)
ESEP_factor = [0.9, 1.0, 1.1]   # multiplies the starting deck's ESEP

[groups.main]                   # named groups of deck keys and their steps
ESEP = { rel = 0.5 }
ETA  = { abs = 0.0065 }
"Q*" = { rel = 0.25 }           # wildcards match every such key the deck has

[groups.ss]
"SP?" = { abs = 0.2 }

[[phase]]                       # bound: phases run in this order
name   = "p0_main"
params = ["main"]

[[phase]]
name   = "p1_ss0"
params = ["ss"]
blocks = "first"                # SS blocks: first | rest | all (needed when a phase has SS keys)
```

Shipped plans:
- `plans/s_shell.toml` — s-shell nuclei (H3, He3, He4): one main + three-body phase
- `plans/p_shell.toml` — p-shell nuclei: main, then the first SS block, then main + three-body + all SS blocks
- `plans/sc_scattering.toml` — the `corr` group used at each BSCAT point

D2/F2 three-body correlations (`D2COR`, `F2COR`) start at 0 and only act when the `.3bp` file switches D2/F2 on; add them in their own group (with an `abs` step) to plans used with D2/F2 potentials.
