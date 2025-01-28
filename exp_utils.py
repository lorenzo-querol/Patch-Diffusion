import os
import json
import re
import dnnlib


def create_output_directory(trainer_kwargs):
    os.makedirs(trainer_kwargs.run_dir, exist_ok=True)
    with open(os.path.join(trainer_kwargs.run_dir, "training_options.json"), "wt") as f:
        json.dump(trainer_kwargs, f, indent=2)

    dnnlib.util.Logger(
        file_name=os.path.join(trainer_kwargs.run_dir, "log.txt"),
        file_mode="a",
        should_flush=True,
    )


def generate_run_id(opts):
    prev_run_dirs = []
    if os.path.isdir(opts.outdir):
        prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    run_dir = os.path.join(opts.outdir, f"{cur_run_id:05d}-run")
    assert not os.path.exists(run_dir)

    return run_dir


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges
