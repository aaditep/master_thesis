Repository for code involved in my master thesis at ETH 2024

This work explores using self-supervised learning (SSL) deep neural networks to acquire
informative representations of simulated WL (weak lensing) mass-maps through pre-training.
Two computer vision SSL frameworks, generative (Masked AutoEncoder) and contrastive
(SimCLR), were implemented using PyTorch in Python. To assess the representations gained
from SSL pretraining, the representations are used for a downstream task of predicting Ω_m
and σ_8 parameters from simulated mass-maps. The SSL results are compared to a baseline of
a purely supervised model with the equivalent prediction task. While both models showed some
struggles due to limited data amount, SimCLR showed potential, but the Masked AutoEncoder
(MAE) failed to produce useful representations.

Shortcuts to specific model elements
SIMCLR:
Downstream model [downstream_model.py](Full_simclr/downstream_model.py)
Pretraining_model
Supervised_model
Testing_book


MAE:

Cosmo_mae2.py (Pretraining)comc
Cosmo_mae2_probe.py (downstreaming)
Cosmo_mae_book (pretraining)
MAE_downstream_results (downstreaming)![image](https://github.com/user-attachments/assets/1e71a915-eea1-4fad-87d9-4a2cb3429224)


