import numpy as np
import pandas as pd
import jax.numpy as jp
import os
import scipy.sparse
import time
import sys

from simulation_utils import stack_connectomes, setup_delays, setup_receptors, run_bold_sweep
import gast_model as gm


pid = sys.argv[1]
input_dir = sys.argv[2]
output_file = sys.argv[3]

W = pd.read_csv(os.path.join(input_dir, "dk_weights_with_sero_and_dopa.csv"), index_col=0)
L = pd.read_csv(os.path.join(input_dir, "dk_lengths_with_sero_and_dopa.csv"), index_col=0)

setup = {
    'Seids': [],
    'idelays': [],
    'params': gm.sigm_d1d2sero_default_theta,
    'v_c': 3.9,
    'horizon': 650,
    'num_item': 1,
    'dt': 0.1,
    'num_skip': 10,
    'num_time': 300000,
    'init_state': jp.array([.01, -55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(10, 1),
    'noise': 0.03
}

Ceids = stack_connectomes(W)
setup['Seids'] = scipy.sparse.csr_matrix(Ceids)
setup['idelays'] = setup_delays(L, Ceids, setup['v_c'], setup['dt'])

Rd1, Rd2, Rsero = setup_receptors()

theta = gm.sigm_d1d2sero_default_theta._replace(
    I=46.5, Ja=13, Jsa=13., Jsg=0, Jg=0,
    Rd1=Rd1, Rd2=Rd2, Rs=Rsero,
    Sd1=-10.0, Sd2=-10.0, Ss=-40.0,
    Zd1=0.5, Zd2=1., Zs=.25,
    we=0.3, wi=0, wd=1, ws=0.,
    sigma_V=setup['noise'],
    sigma_u=0.1 * setup['noise'],
)

setup['params'] = theta

print(f"Running simulation for {pid}")

tic = time.time()
bold = run_bold_sweep((theta, setup))
toc = time.time()

bold = np.array(bold)

np.save(output_file, bold)
