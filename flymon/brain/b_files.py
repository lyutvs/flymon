"""The files spec J.12's raw records depend on (the resume and summary key) and the files a run hashes (dirty check)."""
MEASURE_FILES = ("uv.lock", "flymon/brain/circuits.py", "flymon/brain/config.py", "flymon/brain/connectome.py",
                 "flymon/brain/engine_cpu.py", "flymon/brain/fly_pool.py", "flymon/brain/plasticity.py",
                 "flymon/brain/presentation.py", "flymon/brain/conditioning.py", "flymon/brain/stimuli.py",
                 "flymon/brain/thresholds.py", "flymon/brain/b_spec.py", "flymon/brain/b_runner.py")
HASHED_FILES = MEASURE_FILES + ("flymon/brain/b_rules.py", "flymon/brain/b_store.py", "flymon/brain/b_files.py",
                                "scripts/run_b_test.py", "scripts/write_b_summary.py")
