import os
import sys
import time
import numpy as np
import pandas as pd
import jax.numpy as jp
import scipy.sparse

from paths import Paths
from simulation_utils import (
    stack_connectomes,
    setup_delays,
    setup_ja,
    run_bold_sweep,
)
import gast_model as gm

# ------------------------
# Paths and inputs
# ------------------------

pid_full = sys.argv[1]          # e.g. sub-1272
pid = pid_full.split("-")[1]    # e.g. 1272
input_dir = sys.argv[2]

RESULTS_DIR = Paths.RESULTS     # should point to results/bold
PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../.."))
CENTILE_DIR = os.path.join(PROJECT_DIR, "data", "processed")

output_dir = os.path.join(RESULTS_DIR, pid_full)
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# Parameter sweeps
# ------------------------

we = np.round(np.linspace(0, 1, 10), 2)
means = np.round(np.linspace(0, 50, 10), 2)
stds = np.round(np.linspace(0, 5, 10), 2)

# ------------------------
# Load data
# ------------------------

W = pd.read_csv(
    os.path.join(input_dir, "dk_weights_with_sero_and_dopa.csv"),
    index_col=0
)
L = pd.read_csv(
    os.path.join(input_dir, "dk_lengths_with_sero_and_dopa.csv"),
    index_col=0
)
zscores = pd.read_csv(
    os.path.join(CENTILE_DIR, "zscore_full_chinese_all_with_fu_cortical_thick.csv"),
    index_col="SubjectID"
)

# ------------------------
# Model setup
# ------------------------

setup = {
    "Seids": [],
    "idelays": [],
    "params": gm.sigm_d1d2sero_default_theta,
    "v_c": 3.9,
    "horizon": 650,
    "num_item": we.shape[0],
    "dt": 0.1,
    "num_skip": 10,
    "num_time": 300000,
    "init_state": jp.array(
        [.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(10, 1),
    "noise": 0.03,
}

Ceids = stack_connectomes(W)
setup["Seids"] = scipy.sparse.csr_matrix(Ceids)
setup["idelays"] = setup_delays(L, Ceids, setup["v_c"], setup["dt"])

# ------------------------
# Run sweep
# ------------------------

print(f"Running simulations for {pid_full}")

for mean in means:
    for std in stds:
        print(f"  mean={mean}, std={std}")

        Ja = setup_ja(zscores, W, pid, mean, std)

        theta = gm.sigm_d1d2sero_default_theta._replace(
            I=46.5,
            Ja=Ja,
            Jsa=Ja,
            Jsg=0,
            Jg=0,
            Rd1=0,
            Rd2=0,
            Rs=0,
            Sd1=-10.0,
            Sd2=-10.0,
            Ss=-40.0,
            Zd1=0.5,
            Zd2=1.0,
            Zs=0.25,
            we=we,
            wi=0,
            wd=0,
            ws=0.0,
            sigma_V=setup["noise"],
            sigma_u=0.1 * setup["noise"],
        )

        setup["params"] = theta

        tic = time.time()
        bold = run_bold_sweep((theta, setup))
        toc = time.time()

        bold = np.asarray(bold)

        output_file = os.path.join(
            output_dir,
            f"JJa_{mean}_{std}_bold.npy"
        )

        np.save(output_file, bold)

print(f"Finished {pid_full}")
