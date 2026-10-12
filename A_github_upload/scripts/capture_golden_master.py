"""Capture the Golden Master: execute the original, unmodified notebook end-to-end (true
Restart & Run All, fresh kernel) and dump the exact model outputs to results/golden_master.json
and results/golden_master_trades/*.json, per REFACTOR_INSTRUCTIONS_CLAUDE_CODE.md Section 2.

This does NOT modify legacy/Stat_Arb_cont_res_1.ipynb. It loads that notebook, appends a
throwaway extraction cell to an in-memory copy, executes the copy, and discards the executed
copy after harvesting the JSON outputs -- the on-disk legacy notebook stays byte-identical to
the original.

The frozen legacy notebook's own cells read `px.pkl`/`ret.pkl` via a bare relative path (its
cells are never edited, by design -- see CLAUDE.md "S0 is the control model and is never
edited"). Since the canonical data files now live in `data/raw/` (see
docs/project_management/CLAUDE_FOLDER_CLEANUP_AND_NEXT_RESEARCH.md), this script transiently
copies them into the project root immediately before execution and removes the copies
immediately after (success or failure) -- the root never holds a permanent duplicate.

Run from the project root:  python scripts/capture_golden_master.py
"""
import os
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient

EXTRACTION_SOURCE = r"""
# --- Golden Master extraction (appended by scripts/capture_golden_master.py, not part of the
# original notebook) -------------------------------------------------------------------------
import json
import os as _os
import numpy as _np
import pandas as _pd


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, _pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        v = float(obj)
        return None if v != v else v  # NaN -> null
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, _pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


_os.makedirs('results', exist_ok=True)
_os.makedirs('results/golden_master_trades', exist_ok=True)

_golden_master = {
    'S0_STATIC_BASELINE': {
        'summary_row': _jsonable(S0_ROW),
        'selected_pairs': _jsonable(FROZEN_MODEL['pairs']),
        'final_equity': float(S0_OOS_PORTFOLIO_STD['frame']['equity'].iloc[-1]),
    },
    'S1_STATIC_PAIR_HALFLIFE_STOP': {
        'summary_row': _jsonable(S1_ROW),
        'selected_pairs': _jsonable(S1_ELIGIBLE_PAIRS),
        'final_equity': float(S1_OOS_PORTFOLIO['frame']['equity'].iloc[-1]),
    },
    'WF1_SEMIANNUAL_REESTIMATION': {
        'summary_row': _jsonable(WF1_ROW),
        'selected_pairs': _jsonable({str(k): v for k, v in WF1_SELECTIONS.items()}),
        'final_equity': float((1 + WF1_FULL_FRAME['net_ret'].fillna(0)).cumprod().iloc[-1]),
    },
    'WF2_ROLLING_ENTRY_HEDGE_RATIO': {
        'summary_row': _jsonable(wf2_row),
        'final_equity': float((1 + WF2_FULL_FRAME['net_ret'].fillna(0)).cumprod().iloc[-1]),
    },
}

with open('results/golden_master.json', 'w') as _f:
    json.dump(_golden_master, _f, indent=2, default=str)


def _dump_trades(trades_dict, path):
    with open(path, 'w') as _f:
        json.dump({str(k): _jsonable(v) for k, v in trades_dict.items()}, _f, indent=2, default=str)


_dump_trades(oos_trades, 'results/golden_master_trades/S0_trades.json')
_dump_trades(S1_OOS_TRADES, 'results/golden_master_trades/S1_trades.json')
_dump_trades(WF1_ALL_TRADES, 'results/golden_master_trades/WF1_trades.json')
_dump_trades(WF2_ALL_TRADES, 'results/golden_master_trades/WF2_trades.json')

print('GOLDEN_MASTER_CAPTURE_OK')
print(json.dumps({k: v['summary_row'] for k, v in _golden_master.items()}, indent=2, default=str))
"""


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    src_path = os.path.join('legacy', 'Stat_Arb_cont_res_1.ipynb')
    print(f'Loading {src_path} ...')
    nb = nbformat.read(src_path, as_version=4)

    extraction_cell = nbformat.v4.new_code_cell(source=EXTRACTION_SOURCE)
    nb.cells.append(extraction_cell)

    # Transient copies for the frozen notebook's bare-relative-path px.pkl/ret.pkl reads --
    # removed in the `finally` block below regardless of outcome.
    transient_paths = [
        (os.path.join(project_root, 'data', 'raw', 'px.pkl'), os.path.join(project_root, 'px.pkl')),
        (os.path.join(project_root, 'data', 'raw', 'ret.pkl'), os.path.join(project_root, 'ret.pkl')),
    ]
    for src, dst in transient_paths:
        shutil.copyfile(src, dst)

    try:
        client = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': project_root}})

        t0 = time.time()
        print('Executing notebook end-to-end (fresh kernel, true Restart & Run All) ...')
        client.execute()
        elapsed = time.time() - t0
        print(f'Execution finished in {elapsed:.1f}s')

        last_cell = nb.cells[-1]
        stdout = ''.join(o.get('text', '') for o in last_cell.get('outputs', []) if o.get('output_type') == 'stream')
        if 'GOLDEN_MASTER_CAPTURE_OK' not in stdout:
            print('ERROR: extraction cell did not report success. Last cell outputs:')
            print(last_cell.get('outputs'))
            sys.exit(1)

        print('Golden Master captured successfully -> results/golden_master.json')
        print(stdout)
    finally:
        for _, dst in transient_paths:
            if os.path.exists(dst):
                os.remove(dst)


if __name__ == '__main__':
    main()
