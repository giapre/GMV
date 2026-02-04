import numpy as np
import pandas as pd
import jax.numpy as jp
import os 
import scipy.sparse
from simulation_utils import stack_connectomes, setup_delays, setup_receptors, run_bold_sweep
import gast_model as gm
import matplotlib.pyplot as plt
import time

setup = {'Seids': [],
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

PATIENTS_DIR = '/data3/VBT_SCZ/GMV/data/derivatives'
pid = "sub-2015052501"
PID_DIR = os.path.join(PATIENTS_DIR, pid)
W = pd.read_csv(os.path.join(PID_DIR, "dk_weights_with_sero_and_dopa.csv"), index_col=0)
L = pd.read_csv(os.path.join(PID_DIR, "dk_lengths_with_sero_and_dopa.csv"), index_col=0)
print(W.shape, L.shape)
Ceids = stack_connectomes(W)
setup['Seids'] = scipy.sparse.csr_matrix(Ceids)
setup['idelays'] = setup_delays(L, Ceids, setup['v_c'], setup['dt'])
Rd1, Rd2, Rsero = setup_receptors()

#g_p1, g_p2 = np.mgrid[0.3, 1]

#jp_p1 = jp.array(g_p1.ravel())
#jp_p2 = jp.array(g_p2.ravel())

we=0.3
wd=1
setup['num_item'] = 1#we.shape[0]

theta = gm.sigm_d1d2sero_default_theta._replace(
        I=46.5, Ja=13, Jsa=13., Jsg=0, Jg=0, 
         Rd1=Rd1, Rd2=Rd2, Rs=Rsero, Sd1=-10.0, Sd2=-10.0, Ss=-40.0, Zd1=0.5, Zd2=1., Zs=.25,
        we=we, wi=0, wd=wd, ws=0., sigma_V=setup['noise'], sigma_u=0.1*setup['noise'],
    )

setup['params'] = theta
print(f'Start simulationg {pid}')
tic = time.time()
bold = run_bold_sweep((theta, setup))
toc = time.time()

print(f'Bold took: {toc-tic} seconds')
print(bold.shape)
bold = np.array(bold[:,:,0])
plt.plot(range(90) + 10*bold/bold.max(), linewidth=0.5);
plt.savefig('test.png')

