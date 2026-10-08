"""
pool_model_functions -- thin compatibility shim over fusion_core
==================================================================

Every function and constant that used to be defined in this file has been
moved into :mod:`pool_paper_casestudy.fusion_core`, sorted by what it does:

- Dataframe construction/extraction (``create_df_poolpaper``,
  ``extract_observables_from_df``)        -> ``fusion_core.dtf``
- Data generation/loading (``data_generation_poolpaper``, ``model_wotemp``,
  ``generate_data_dfs``, ``experimental_values``, ``get_param_dfs``)
                                            -> ``fusion_core.data``
- The ODE models and everything coupled to them (``ode_model_coculture*``,
  ``set_initial_vals``, ``observable``, ``pH_func``, ``interpolate_series``,
  ``cost``, ``sq_diff_oneexp``)            -> ``fusion_core.mdl``
- Global optimization driver (``calculate_model_params``)
                                            -> ``fusion_core.pest``
- Plotting (style constants, ``plot_all_curves``, ``plot_cases_separately``,
  ``set_labels``, ``pH_LA_dependence``)    -> ``fusion_core.plotting``

This file just re-exports everything under its original name, so every
other pool_paper_casestudy file that does
`from pool_paper_casestudy.pool_model_functions import *` keeps working
unchanged.
"""

import os
import sys

sys.path.append(os.getcwd())
from pool_paper_casestudy import fusion_core as fm

# ----------------------------------------------------------------------
# fusion_core.dtf
# ----------------------------------------------------------------------
create_df_poolpaper = fm.dtf.create_df_poolpaper
extract_observables_from_df = fm.dtf.extract_observables_from_df

# ----------------------------------------------------------------------
# fusion_core.data
# ----------------------------------------------------------------------
data_generation_poolpaper = fm.data.data_generation_poolpaper
model_wotemp = fm.data.model_wotemp
generate_data_dfs = fm.data.generate_data_dfs
experimental_values = fm.data.experimental_values
get_param_dfs = fm.data.get_param_dfs

# ----------------------------------------------------------------------
# fusion_core.mdl
# ----------------------------------------------------------------------
set_initial_vals = fm.mdl.set_initial_vals
observable = fm.mdl.observable
pH_func = fm.mdl.pH_func
interpolate_series = fm.mdl.interpolate_series
ode_model_coculture = fm.mdl.ode_model_coculture
ode_model_coculture2 = fm.mdl.ode_model_coculture2
ode_model_coculture3 = fm.mdl.ode_model_coculture3
ode_model_coculture_wopH = fm.mdl.ode_model_coculture_wopH
ode_model_coculture_wopH_expsat = fm.mdl.ode_model_coculture_wopH_expsat
ode_model_coculture_wopH_MM = fm.mdl.ode_model_coculture_wopH_MM
ode_model_coculture_withpH_MM = fm.mdl.ode_model_coculture_withpH_MM
sq_diff_oneexp = fm.mdl.sq_diff_oneexp
cost = fm.mdl.cost

# ----------------------------------------------------------------------
# fusion_core.pest
# ----------------------------------------------------------------------
calculate_model_params = fm.pest.calculate_model_params

# ----------------------------------------------------------------------
# fusion_core.plotting
# ----------------------------------------------------------------------
colors_all = fm.plotting.colors_all
figsize_default = fm.plotting.figsize_default
figsize_default_small = fm.plotting.figsize_default_small
figsize_default2subpl = fm.plotting.figsize_default2subpl
set_labels = fm.plotting.set_labels
pH_LA_dependence = fm.plotting.pH_LA_dependence
plot_all_curves = fm.plotting.plot_all_curves
plot_cases_separately = fm.plotting.plot_cases_separately