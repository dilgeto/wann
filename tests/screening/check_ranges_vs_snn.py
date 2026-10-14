#!/usr/bin/env python3
"""Checks that INITIAL_SPACE here equals wann-cpp's for every hyperparameter both repos share, and that
nothing SNN-only slipped in.   usage: check_ranges_vs_snn.py [path/to/wann-cpp]"""
import importlib.util, os, sys, warnings
warnings.filterwarnings("ignore")
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.normpath(os.path.join(here, "..", ".."))
snn_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "..", "wann-cpp")
SNN_ONLY = {"prob_toggleExcitatory", "snn_window_ms", "snn_tau_exc", "snn_tau_inh", "snn_ttfs_threshold"}

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

if not os.path.exists(os.path.join(snn_dir, "screening_reduce.py")):
    print(f"SKIP: {snn_dir}/screening_reduce.py not found"); sys.exit(0)
sys.path.insert(0, root)
ann = load("ann_reduce", os.path.join(root, "screening_reduce.py")).INITIAL_SPACE
snn = load("snn_reduce", os.path.join(snn_dir, "screening_reduce.py")).INITIAL_SPACE
bad = [k for k in ann if k not in snn or tuple(ann[k]) != tuple(snn[k])]
missing = [k for k in snn if k not in ann and k not in SNN_ONLY]
leaked = [k for k in ann if k in SNN_ONLY]
for k in ann: print(f"  {k:<20} {tuple(ann[k])}")
if bad or missing or leaked:
    print("MISMATCH  different:", bad, " shared but missing here:", missing, " SNN-only present:", leaked); sys.exit(1)
print(f"OK: {len(ann)} shared hyperparameters have the same ranges as wann-cpp; SNN-only ones ({len(SNN_ONLY)}) are absent")
