import numpy as np
import pandas as pd
import os
from analysis_utils import compute_cortical_emp_sim_alff_correlation, compute_cortical_emp_sim_falff_correlation

from paths import Paths

demo = pd.read_csv(f'{Paths.DEMOGRAPHICS}/full_demographics.csv', index_col='SubjectID')
all_sim_dfs = [] 
all_emp_dfs = []
for pid in os.listdir(Paths.RESULTS):

    if not pid.startswith('sub'): continue
    else:
        pid_dir = os.path.join(Paths.RESULTS, pid)

        trad_emp_dir = f'{pid_dir}/traditional_emp_results.npz'
        trad_emp_data = np.load(trad_emp_dir)
        trad_emp_fc = trad_emp_data['FC']
        trad_emp_fcd = trad_emp_data['FCD']
        trad_emp_alff = trad_emp_data['ALFF']

        emp_data_dir = f'{pid_dir}/emp_results.npz'
        emp_data = np.load(emp_data_dir)
        emp_fc = emp_data['FC']
        emp_fcd = emp_data['FCD']
        emp_alff = emp_data['ALFF']

        sim_dir = f'{pid_dir}/sim_results.npz'
        sim_data = np.load(sim_dir)
        sim_fc = sim_data['FC']
        sim_fcd = sim_data['FCD']
        sim_alff = sim_data['ALFF']
        params = sim_data['params']

        sim_gbc = np.mean(sim_fc,axis=0)
        sim_var_fcd = np.var(sim_fcd,axis=0)
        emp_gbc = np.mean(trad_emp_fc)
        emp_var_fcd = np.var(trad_emp_fcd)
        alff_corr = compute_cortical_emp_sim_alff_correlation(trad_emp_dir, sim_dir)
        falff_corr = compute_cortical_emp_sim_falff_correlation(trad_emp_dir, sim_dir)

        sim_df = pd.DataFrame({'GBC': sim_gbc, 'VAR_FCD': sim_var_fcd, 'ALFF_CORR':alff_corr, 'fALFF_CORR':falff_corr, 'mean': params[:,0], 'std': params[:,1], 'we': params[:,2]})

        pid_code = pid.split('-')[1]
        pid_demo = demo.loc[int(pid_code)]
        sim_df['pid'] = pid
        for key in pid_demo.keys():
            sim_df[key] = pid_demo[key]

        all_sim_dfs.append(sim_df)
        all_emp_dfs.append([pid, emp_gbc, emp_var_fcd])

final_df = pd.concat(all_sim_dfs)
final_df.to_csv(f'{Paths.RESULTS}/demo_and_features.csv')
emp_df = pd.DataFrame(data=all_emp_dfs, columns=['pid', 'GBC', 'VAR_FCD'])
emp_df.to_csv(f'{Paths.RESULTS}/full_emp_features.csv')

                
