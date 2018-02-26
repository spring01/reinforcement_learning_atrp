'''
List of registered environment names in this file:

ATRP-psnt-td-gv24-v0
ATRP-psnt-td-gv28-v0
ATRP-psnt-td-gv32-v0
ATRP-psnt-td-gv36-v0
ATRP-psnt-td-gv40-v0
ATRP-psnt-td-gv44-v0
ATRP-psnt-td-gv48-v0
ATRP-psnt-td-gv52-v0

ATRP-pst-td-gv24-v0
ATRP-pst-td-gv28-v0
ATRP-pst-td-gv32-v0
ATRP-pst-td-gv36-v0
ATRP-pst-td-gv40-v0
ATRP-pst-td-gv44-v0
ATRP-pst-td-gv48-v0
ATRP-pst-td-gv52-v0

ATRP-psnt-td-srcg-gv24-v0
ATRP-psnt-td-srcg-gv28-v0
ATRP-psnt-td-srcg-gv32-v0
ATRP-psnt-td-srcg-gv36-v0
ATRP-psnt-td-srcg-gv40-v0
ATRP-psnt-td-srcg-gv44-v0
ATRP-psnt-td-srcg-gv48-v0
ATRP-psnt-td-srcg-gv52-v0

ATRP-pst-td-srcg-gv24-v0
ATRP-pst-td-srcg-gv28-v0
ATRP-pst-td-srcg-gv32-v0
ATRP-pst-td-srcg-gv36-v0
ATRP-pst-td-srcg-gv40-v0
ATRP-pst-td-srcg-gv44-v0
ATRP-pst-td-srcg-gv48-v0
ATRP-pst-td-srcg-gv52-v0
'''

'''
Gaussian variance control environments
'''
import numpy as np
from register_targets import register_targets


''' Two-level reward environments. +0.1 if close, +1.0 if very close. '''
kwargs_common = dict(
    max_rad_len=100,
    step_time=1e2,
    completion_time=1e5,
    min_steps=100,
    termination=None, # must be set later
    k_prop=1.6e3,
    k_act=0.45,
    k_deact=1.1e7,
    k_ter=1e8,
    mono_init=0.0,
    cu1_init=0.0,
    cu2_init=0.0,
    dorm1_init=0.0,
    mono_unit=0.1,
    cu1_unit=0.004,
    cu2_unit=0.004,
    dorm1_unit=0.008,
    mono_cap=10.0,
    cu1_cap=0.2,
    cu2_cap=0.2,
    dorm1_cap=0.4,
    mono_density=8.73,
    sol_init=0.01,
    sol_cap=0.0,
    reward_chain_type='dorm',
    reward_loose=0.1,
    reward_tight=1.0,
    thres_loose=1e-2,
    thres_tight=3e-3,)


mean = 24
space = np.linspace(1, 100, 100)
space_shifted = space - mean

gv24 = np.exp(- space_shifted * space_shifted / (2 * 24))
gv24 /= np.sum(gv24)
gv28 = np.exp(- space_shifted * space_shifted / (2 * 28))
gv28 /= np.sum(gv28)
gv32 = np.exp(- space_shifted * space_shifted / (2 * 32))
gv32 /= np.sum(gv32)
gv36 = np.exp(- space_shifted * space_shifted / (2 * 36))
gv36 /= np.sum(gv36)
gv40 = np.exp(- space_shifted * space_shifted / (2 * 40))
gv40 /= np.sum(gv40)
gv44 = np.exp(- space_shifted * space_shifted / (2 * 44))
gv44 /= np.sum(gv44)
gv48 = np.exp(- space_shifted * space_shifted / (2 * 48))
gv48 /= np.sum(gv48)
gv52 = np.exp(- space_shifted * space_shifted / (2 * 52))
gv52 /= np.sum(gv52)
target_list = [('gv24', gv24), ('gv28', gv28), ('gv32', gv32), ('gv36', gv36),
               ('gv40', gv40), ('gv44', gv44), ('gv48', gv48), ('gv52', gv52),]

'''
No-termination, no-noise envs
'''
kwargs_nt_common = kwargs_common.copy()
kwargs_nt_common['termination'] = False
register_targets(kwargs_nt_common, 'psnt-td', target_list)

'''
With-termination, no-noise envs
'''
kwargs_t_common = kwargs_nt_common.copy()
kwargs_t_common['termination'] = True
register_targets(kwargs_t_common, 'pst-td', target_list)


'''
Gaussian variance control environments with "simulated realistic concerns"
(i.e., noises), gaussian noise version
All currently implemented noises are turned on, including:
    step_time_noise = 0.03  : gaussian sigma = 0.01 noise in step_time;
    k_noise = 0.3           : gaussian sigma = 0.1 noise in all rate constants;
    obs_noise = 3e-3        : gaussian sigma = 1e-3 noise on observations;
    addition_noise = 0.03   : gaussian sigma = 0.01 noise in addition unit amount of species.
'''
'''
No-termination, srcg-noise envs
'''
kwargs_nt_srcg_common = kwargs_nt_common.copy()
kwargs_nt_srcg_common['step_time_noise'] = 'gaussian', 0.03
kwargs_nt_srcg_common['k_noise'] = 'gaussian', 0.3
kwargs_nt_srcg_common['obs_noise'] = 'gaussian', 3e-3
kwargs_nt_srcg_common['addition_noise'] = 'gaussian', 0.03
kwargs_nt_srcg_common['render_obs'] = 'dorm', (8, 108)
register_targets(kwargs_nt_srcg_common, 'psnt-td-srcg', target_list)

'''
With-termination, srcg-noise envs
'''
kwargs_t_srcg_common = kwargs_nt_srcg_common.copy()
kwargs_t_srcg_common['termination'] = True
register_targets(kwargs_t_srcg_common, 'pst-td-srcg', target_list)

