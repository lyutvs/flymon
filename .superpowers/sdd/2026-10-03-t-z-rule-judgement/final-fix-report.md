
## Final-review Minor 1 fix
Change: t_runner.check_z_matches_spec(spec, z) refuses (exit 2) unless the runtime h4 z equals spec.z_h4_dict() exactly; build_ctx calls it on cfg.z right after the readout check. Test: test_runtime_z_must_equal_the_declared_z_h4 (match passes; mean or SD off refuses with code 2).
Command: .venv/bin/python -m pytest tests/brain/test_t_runner.py tests/brain/test_t_judge.py -q
Output: ...............................................................          [100%]
