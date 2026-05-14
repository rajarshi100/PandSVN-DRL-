# Legacy code

Place the original research prototype files here:

- `VehicularMobilityTools.py`
- `environment_utils.py`
- `memory_utils.py`
- `model_utils.py`

Recommended command from the repository root:

```bash
mkdir -p legacy
git mv VehicularMobilityTools.py environment_utils.py memory_utils.py model_utils.py legacy/
```

The files are kept for traceability. Future refactoring should move stable components into `src/pandsvn_drl/`.
