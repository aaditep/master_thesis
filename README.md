##Repository for code involved in my master thesis at ETH 2024

This work explores using self-supervised learning (SSL) deep neural networks to acquire
informative representations of simulated WL (weak lensing) mass-maps through pre-training.
Two computer vision SSL frameworks, generative (Masked AutoEncoder) and contrastive
(SimCLR), were implemented using PyTorch in Python. To assess the representations gained
from SSL pretraining, the representations are used for a downstream task of predicting Ω_m
and σ_8 parameters from simulated mass-maps. The SSL results are compared to a baseline of
a purely supervised model with the equivalent prediction task. While both models showed some
struggles due to limited data amount, SimCLR showed potential, but the Masked AutoEncoder
(MAE) failed to produce useful representations.

## Shortcuts to Specific Model Elements

### SIMCLR:
- **Supervised model**: [`supervised_model.py`](Full_simclr/supervised_model.py)
- **Pretraining model**: [`pretraining_model.py`](Full_simclr/pretraining_model.py)
- **Downstream model**: [`downstream_model.py`](Full_simclr/downstream_model.py)
- **Results notebook**: [`simclr_results.ipynb`](Full_simclr/simclr_results.ipynb)

### MAE:
- **Pretraining model**: [`cosmo_mae2.py`](MAE/cosmo_mae2.py)
- **Pretraining image Results**: [`cosmo_mae_book.ipynb`](MAE/cosmo_mae_book.ipynb)
- **Supervised model/Downstream model**: [`cosmo_mae2_probe.py`](MAE/cosmo_mae2_probe.py) 
- **Results notebook**: [`MAE_downstream_results.ipynb`](MAE/MAE_downstream_results.ipynb)
  
