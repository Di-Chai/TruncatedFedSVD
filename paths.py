import os

log_dir = 'log'

results_dir = 'results'

paths = [log_dir, results_dir]

for path in paths:
    if os.path.isdir(path) is False:
        os.makedirs(path, exist_ok=True)
