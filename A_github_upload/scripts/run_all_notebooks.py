"""Restart & Run All on all four notebooks in order, from a clean kernel each time. Verifies
there is no hidden inter-notebook state dependency beyond the explicit results/tables/*.json
hand-off. Run from the project root: python scripts/run_all_notebooks.py"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    '01_data_and_static_baseline.ipynb',
    '02_relationship_stability.ipynb',
    '03_walkforward_strategy.ipynb',
    '04_results_and_model_comparison.ipynb',
]


def run_one(path, notebooks_dir):
    print(f'--- Executing {path} ---', flush=True)
    nb = nbformat.read(os.path.join(notebooks_dir, path), as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name='python3',
                             resources={'metadata': {'path': notebooks_dir}})
    t0 = time.time()
    try:
        client.execute()
    except Exception as e:
        print(f'FAILED: {path}: {e}', flush=True)
        nbformat.write(nb, os.path.join(notebooks_dir, path))
        return False
    elapsed = time.time() - t0
    nbformat.write(nb, os.path.join(notebooks_dir, path))
    print(f'OK: {path} in {elapsed:.1f}s', flush=True)
    return True


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebooks_dir = os.path.join(project_root, 'notebooks')
    os.chdir(project_root)
    all_ok = True
    for nb_name in NOTEBOOKS:
        ok = run_one(nb_name, notebooks_dir)
        all_ok = all_ok and ok
        if not ok:
            print(f'Stopping: {nb_name} failed, downstream notebooks depend on its output.')
            break
    print('ALL_NOTEBOOKS_OK' if all_ok else 'NOTEBOOK_RUN_FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
