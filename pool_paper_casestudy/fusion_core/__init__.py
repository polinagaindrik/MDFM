"""
fusion_core: self-contained replacement for `fusion_model` inside pool_paper_casestudy
========================================================================================

`pool_paper_casestudy` used to do ``import fusion_model as fm`` and only ever called
eight functions from it (``fm.data.json_dump``, ``fm.data.save_all_dfs``,
``fm.dtf.merge_dfs``, ``fm.mdl.model_ODE_solution``, ``fm.output.json_dump``,
``fm.output.read_from_json``, ``fm.pest.cost_arithmetic_mean``,
``fm.pest.optimization_func`` and ``fm.plotting.plot_cost_function``).

Importing the real ``fusion_model`` package, however, pulls in its OpenBIS data
download layer (``fusion_model/data/read_ZL2030data.py``), which requires the
``pybis`` package -- a dependency this case study never actually needs, since all
of its experimental data is read directly from local Excel files.

This subpackage vendors just those eight functions, verbatim in behaviour, under
the same ``fm.<namespace>.<function>`` access pattern as before, so that the rest
of ``pool_paper_casestudy`` keeps working unchanged after simply swapping the
import line

    import fusion_model as fm

for

    from pool_paper_casestudy import fusion_core as fm

No file in `pool_paper_casestudy` needs to import `fusion_model` (or `pybis`)
anymore.
"""

from . import data
from . import output
from . import dtf
from . import mdl
from . import pest
from . import plotting
from . import likelihood
from . import model_selection

__all__ = ["data", "output", "dtf", "mdl", "pest", "plotting", "likelihood", "model_selection"]
