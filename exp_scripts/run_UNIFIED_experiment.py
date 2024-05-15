# %%
# import relevant modules
import sys

sys.path.append("..")
import torch
import numpy as np
import pandas as pd
from bot.policies.modules.segment_dataset import (
    SegmentAndEnvDatasetWithPeaks,
    PriceSolarTransformer,
    drop_no_solar,
    split_segment_for_validation,
)
from bot.policies.modules import (
    StatefulRNNModel,
    SimplifiedStatefulRNNModel,
    SolarRandomNoiseAugmenter,
    BatteryEnv,
)

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

torch.set_float32_matmul_precision("medium")

MODEL_NAME = "semistateful_sig_model"
EXTRA_TAGS = "-4d"
# %%
# load training and validation data
validation_data = pd.read_csv(
    "../bot/data/validation_data.csv", parse_dates=["timestamp"]
)
training_data = pd.read_csv("../bot/data/training_data.csv", parse_dates=["timestamp"])
training_data = drop_no_solar(training_data)
# %%
# transform data and convert to float32
preprocessor = PriceSolarTransformer(
    include_pos_code=True,
    include_peak_indicator=True,
    include_price_augmentations=True,
    interpolate_missing=True,
    price_transform_patch=True,
)
dat, pv_power, price, peak_ind = preprocessor.fit_transform(training_data)
dat_val, pv_power_val, price_val, peak_ind_val = preprocessor.transform(validation_data)
seq_len = len(validation_data)
dat, price, pv_power = (
    dat.astype(np.float32),
    price.astype(np.float32),
    pv_power.astype(np.float32),
)
dat_val, price_val, pv_power_val = (
    dat_val.astype(np.float32),
    price_val.astype(np.float32),
    pv_power_val.astype(np.float32),
)
# %%
# set aside a part of training data for extra validation
validation_indices = training_data.reset_index(drop=True).query(
    "timestamp >= '2023-04-15 00:00:00 UTC' and timestamp < '2023-05-07 00:00:00 UTC'"
)
val_s, val_e = validation_indices.index[0], validation_indices.index[-1] + 1
val_len = validation_indices.shape[0] - 1
# %%
TRAIN_LEN = 60 // 5 * 24 * 4
# split training data into training and validation sets and combine new validation set with original validation set
ds_train, ds_valid_1 = split_segment_for_validation(
    dat,
    pv_power,
    price,
    peak_ind,
    segment=(val_s, val_e),
    interval_len=TRAIN_LEN,
    validation_len=val_len,
    train_skip_step=1,
)
ds_valid_2 = SegmentAndEnvDatasetWithPeaks(
    dat_val,
    price_val,
    pv_power_val,
    peak_ind_val,
    interval_len=TRAIN_LEN,
    skip_step=TRAIN_LEN // 8,
)
# %%
# size sanity check
print(len(ds_train))
print(len(ds_valid_1))
print(len(ds_valid_2))
print(ds_train[0][0].shape)
print(ds_train[0][1].shape)
print(ds_train[0][2].shape)
print(ds_train[0][3].shape)
# %%
augmenter = SolarRandomNoiseAugmenter(0.01, (2, 3, 4, 5)) # augment solar power columns
if MODEL_NAME == "stateful_rnn_model":
    battery = BatteryEnv(13, 5, 7.5)
    model = StatefulRNNModel(
        battery=battery,
        input_size=dat.shape[-1],
        hidden_size=128,
        fc_size=64,
        num_encoder_layers=4,
        dropout=0.25,
        augmenter=augmenter,
        simultaneous_trade_penalty=0.0,
        beta_min=1,
        beta_increment=0.2,
        beta_max=10.0,
    )
elif MODEL_NAME == "simplified_stateful_rnn_model":
    battery = BatteryEnv(13, 5, 7.5)
    model = SimplifiedStatefulRNNModel(
        battery=battery,
        input_size=dat.shape[-1],
        hidden_size=256,
        fc_size=64,
        num_encoder_layers=3,
        dropout=0.25,
        augmenter=augmenter,
        beta_increment=0.25,
        beta_max=5.0,
        total_variation_constraint=0.01,
    )
else:
    raise ValueError("Invalid model name")
# %%
train_loader = DataLoader(
    ds_train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)
valid_loader_1 = DataLoader(ds_valid_1, batch_size=1, shuffle=False, num_workers=4)
valid_loader_2 = DataLoader(
    ds_valid_2, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)
# %%
val_loss_checkpoint_callback = ModelCheckpoint(
    dirpath=f"best_checkpoints/{MODEL_NAME}",
    monitor="val_loss",
    filename=f"{MODEL_NAME}{EXTRA_TAGS}"
    + "-{epoch:02d}-with-{val_loss:.3f}-{val_loss_sec:.3f}",
    mode="min",
)
train_loss_checkpoint_callback = ModelCheckpoint(
    dirpath=f"best_checkpoints/{MODEL_NAME}",
    monitor="train_loss",
    filename=f"{MODEL_NAME}{EXTRA_TAGS}"
    + "-{epoch:02d}-with-{train_loss:.3f}-{val_loss_sec:.3f}",
    mode="min",
)
early_stopping = EarlyStopping("train_loss", patience=15, mode="min")
# %%
trainer = pl.Trainer(
    max_epochs=200,
    accelerator="gpu",
    check_val_every_n_epoch=1,
    callbacks=[
        val_loss_checkpoint_callback,
        train_loss_checkpoint_callback,
        early_stopping,
    ],
    limit_train_batches=50,
    # gradient_clip_val=5.0,
)

# %%
trainer.fit(model, train_loader, [valid_loader_1, valid_loader_2])