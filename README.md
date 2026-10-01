# COGITAO research

Independent copy of the COGITAO work from `/home/ubuntu/causal-dynamics`.
The original project is retained. Gaussian experiments and unrelated datasets,
checkpoints, and models are excluded.

Included: COGITAO data, ViT + function-MLP and joint-token ViT code/checkpoints,
direct-patch experiments, geometry/PCA scripts, atomic/composition/progressive
evaluators, sweep YAMLs, local reports, COGITAO baseline archives, shared trainer,
and dataset downloader. Archived Slot Attention experiments are included for
reference; the current ViT models do not import or load their weights.

Run commands from this directory so dataset and checkpoint paths resolve here:

```bash
cd /home/ubuntu/cogitao-research
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements record installed dependency versions from the source environment.
The virtual environment and credentials are not copied.

See `benchmarks/cogitao_vit_cross_attention/README.md` and its linked guides for
training, evaluation, latent geometry, PCA, and W&B commands. No training,
evaluation, PCA, or W&B job is launched as part of this copy.

`COPY_MANIFEST.json` records the source selections and verified file hashes.
Historical report paths retain the original provenance.
