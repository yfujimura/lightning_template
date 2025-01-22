# lightning_template
hydra + wandb + pytorch lightning でMNISTを用いてオートエンコーダをGPU並列(Distributed Data Parallel)で学習

# Requirements
[WandB](https://wandb.ai/site/ja/)のアカウントを作成
```
conda create -n pytorch_lightning python=3.10
conda activte pytorch_lightning
pip install -r requirements.txt
```

# Training
```
python -m src.main
```
