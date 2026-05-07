"""Run multi-diffusion big-reservoir generation for all 7 reservoir types
at a custom seed, dropping the outputs into
   paper_figures/gen_seeds/seed_<NN>/<type>/results/reservoir_hard_ov24.npz
which is exactly what `figure4.py` defaults to.

This re-creates the data path that was lost when gen_seeds/ was cleaned up.
Each `big_reservoir/<type>/generate.py` honours `RESERVOIR_SEED` and
`RESERVOIR_OUT_DIR` env vars (the SEED line was patched once to read from
RESERVOIR_SEED).

Parallelism: one subprocess per GPU. With 3 GPUs and 7 types, the runner
distributes types across GPUs round-robin so all three GPUs stay busy.

Run:
    python examples/reservoirs/paper_figures/gen_seeds.py --seed 11
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


NEURIPS_DIR = Path(__file__).resolve().parent
BIG_RES_DIR = NEURIPS_DIR.parent / 'big_reservoir'

# Source dirs under big_reservoir/. figure4.py expects these subdirs (lobes
# for the square panel, *_long for the 6 channel strips).
TYPE_DIRS = [
    'lobes',
    'meander_oxbow_long',
    'pv_shoestring_long',
    'cb_jigsaw_long',
    'cb_labyrinth_long',
    'sh_proximal_long',
    'sh_distal_long',
]


def out_dir_for(seed_root: Path, type_dir: str) -> Path:
    """Output dir mirroring figure4.py's expectation:
        seed_root/<type_dir>/results/reservoir_hard_ov24.npz
    where type_dir is e.g. 'lobes' or 'cb_jigsaw_long' (matches the
    big_reservoir/<type>/ directory names exactly)."""
    return seed_root / type_dir / 'results'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out-root', default=None,
                    help='Defaults to neurips_figures/gen_seeds/seed_<NN>')
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--types', nargs='+', default=TYPE_DIRS,
                    help='Subset of TYPE_DIRS to run (default: all 7).')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip a type whose reservoir_hard_ov24.npz already '
                         'exists.')
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = str(NEURIPS_DIR / 'gen_seeds' / f'seed_{args.seed:02d}')
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'seed={args.seed}  out_root={out_root}')
    print(f'gpus={args.gpus}  types={args.types}')

    # Round-robin assign types -> gpus.
    gpu_pool = args.gpus
    pending = []
    for i, type_dir in enumerate(args.types):
        out = out_dir_for(out_root, type_dir)
        if args.skip_existing and (out / 'reservoir_hard_ov24.npz').exists():
            print(f'  [skip] {type_dir}: {out}/reservoir_hard_ov24.npz exists')
            continue
        pending.append((type_dir, gpu_pool[i % len(gpu_pool)]))

    # Group by GPU so each GPU runs its types serially in one subprocess.
    by_gpu = {g: [] for g in gpu_pool}
    for type_dir, gpu in pending:
        by_gpu[gpu].append(type_dir)

    procs = []
    log_dir = NEURIPS_DIR / 'gen_seeds' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    for gpu, type_dirs in by_gpu.items():
        if not type_dirs:
            continue
        # bash -c that loops over type_dirs and runs each generate.py
        commands = []
        for type_dir in type_dirs:
            cwd = BIG_RES_DIR / type_dir
            out = out_dir_for(out_root, type_dir)
            out.mkdir(parents=True, exist_ok=True)
            cmd = (f'cd {cwd} && '
                   f'CUDA_VISIBLE_DEVICES={gpu} '
                   f'RESERVOIR_SEED={args.seed} '
                   f'RESERVOIR_OUT_DIR={out} '
                   f'python generate.py')
            commands.append(cmd)
        bash_cmd = ' && '.join(commands)
        log_path = log_dir / f'seed_{args.seed:02d}_gpu{gpu}.log'
        print(f'\n[GPU {gpu}] {len(type_dirs)} types: {type_dirs}')
        print(f'  log -> {log_path}')
        f = open(log_path, 'w')
        proc = subprocess.Popen(['bash', '-c', bash_cmd],
                                stdout=f, stderr=subprocess.STDOUT)
        procs.append((gpu, proc, f, type_dirs))

    print(f'\nLaunched {len(procs)} workers. Waiting ...')
    t0 = time.time()
    for gpu, proc, f, type_dirs in procs:
        rc = proc.wait()
        f.close()
        elapsed = time.time() - t0
        status = 'OK' if rc == 0 else f'FAIL ({rc})'
        print(f'  [GPU {gpu}] {status}  types={type_dirs}  '
              f'elapsed={elapsed:.0f}s')

    # Summary
    missing = []
    for type_dir in args.types:
        out = out_dir_for(out_root, type_dir) / 'reservoir_hard_ov24.npz'
        if not out.exists():
            missing.append(str(out))
    if missing:
        print(f'\n[!] {len(missing)} outputs missing:')
        for m in missing:
            print(f'   - {m}')
        sys.exit(1)
    print('\nAll outputs present. Render with:')
    print(f'  python {NEURIPS_DIR / "figure4.py"} '
          f'--results-root {out_root}  # rasterized')
    print(f'  python {NEURIPS_DIR / "figure4.py"} --vector '
          f'--results-root {out_root}  # vector')


if __name__ == '__main__':
    main()
