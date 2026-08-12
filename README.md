works in terminal by running scripts/generate.py

also, add your own pdf to a data folder in root directory for this to work (hopefully)

## Evaluating the pipeline

`scripts/eval/` scores retrieval and answer quality against an auto-generated
golden set. Run from the repo root:

```bash
venv/bin/python -m scripts.eval.run                   # retrieval metrics + corpus health
venv/bin/python -m scripts.eval.run --with-generation # adds end-to-end answer scoring
```

See [scripts/eval/README.md](scripts/eval/README.md) for what the metrics mean
and how to grow the golden set.
