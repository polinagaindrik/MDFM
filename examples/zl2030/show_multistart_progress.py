"""Progress of main_zl2030_multistart.py, from the history files of its starts (safe to run during the fit).
Run from anywhere: python examples/zl2030/show_multistart_progress.py"""
import glob
import os
import re
import sys
import time

LOG_K = '--log-k' in sys.argv  # second batch, see main_zl2030_multistart.py
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out', 'zl2030', 'multistart_logk' if LOG_K else 'multistart', '')
N_STARTS = 48 if LOG_K else 96
EARLIER_COST = 3443.41  # earlier parameter set under the objective of this fit

rows = []
for f in glob.glob(DIR + 'optimization_history_multistart_start*.csv'):
    r = int(re.search(r'_start(\d+)\.csv$', f).group(1))
    costs = []
    with open(f) as fh:
        next(fh, None)
        for line in fh:
            try:
                costs.append(float(line.rsplit(',', 1)[1]))
            except (IndexError, ValueError):
                pass  # line still being written
    if costs:
        rows.append((min(costs), r, len(costs), costs[0], (time.time() - os.path.getmtime(f)) / 60.))
rows.sort()
idle = sum(row[4] > 15 for row in rows)
print(f'{len(rows)} of {N_STARTS} starts have begun; {len(rows) - idle} wrote within 15 min, {idle} quiet for longer (finished or slow)')
print(f'cost of the earlier parameter set: {EARLIER_COST}')
print(f'{"start":>5} {"best cost":>12} {"to earlier":>11} {"points":>7} {"first cost":>12} {"last write":>10}')
for c, r, n, c0, age in rows[:25]:
    print(f'{r:5d} {c:12.2f} {c - EARLIER_COST:11.2f} {n:7d} {c0:12.2f} {age:7.1f}min' + ('  <- earlier set' if r == 0 and not LOG_K else ''))
