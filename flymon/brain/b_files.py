"""The files spec J.12's raw records depend on (the resume and summary key) and the files a run hashes (dirty check).

scripts/run_b_test.py is a measurement file: it decides what is measured (the odour pair's seed, the readout cells, the
brains, a fixed X), so a change to it is a new measurement key."""
MEASURE_FILES = ("uv.lock", "flymon/brain/circuits.py", "flymon/brain/config.py", "flymon/brain/connectome.py",
                 "flymon/brain/engine_cpu.py", "flymon/brain/fly_pool.py", "flymon/brain/plasticity.py",
                 "flymon/brain/presentation.py", "flymon/brain/conditioning.py", "flymon/brain/stimuli.py",
                 "flymon/brain/thresholds.py", "flymon/brain/b_spec.py", "flymon/brain/b_runner.py",
                 "scripts/run_b_test.py")
HASHED_FILES = MEASURE_FILES + ("flymon/brain/b_rules.py", "flymon/brain/b_store.py", "flymon/brain/b_files.py",
                                "scripts/write_b_summary.py")
