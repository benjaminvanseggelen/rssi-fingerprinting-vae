# %%
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)


# %%
USE_BT = True


DATASET_ROOT = "data/"
TRAIN_LOCATIONS_FILE = "SignatureLocs_altered.csv"  # Replaced commas with dots
TRAIN_STRENGTHS_FILE = "P_Signatures.csv" if USE_BT else "P_SA_Signatures.csv"
NUMBER_OF_BEACONS = 57 if USE_BT else 11

# %% [markdown]
# ### Prepare the data

# %%
df_train_strengths = pd.read_csv(
    DATASET_ROOT+TRAIN_STRENGTHS_FILE, sep=';', names=[x for x in range(NUMBER_OF_BEACONS)])
df_train_locs = pd.read_csv(
    DATASET_ROOT+TRAIN_LOCATIONS_FILE, sep=';', names=['x', 'y'], dtype=float)

train_features = df_train_strengths
train_target = df_train_locs

normalization_values = np.array(train_features)

# %% [markdown]
# ### Define our VAE model

# %%


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# %%
latent_dim = 2

#normalizer = layers.Normalization()
# normalizer.adapt(normalization_values)
encoder_inputs = tf.keras.Input(shape=NUMBER_OF_BEACONS)
x = layers.Dense(NUMBER_OF_BEACONS, activation="relu",
                 name="encoder_0")(encoder_inputs)
x = layers.Dense(math.ceil(0.7*NUMBER_OF_BEACONS),
                 activation="relu", name="encoder_1")(x)
x = layers.Dense(math.ceil(0.5*NUMBER_OF_BEACONS),
                 activation="relu", name="encoder_2")(x)
x = layers.Dense(math.ceil(0.3*NUMBER_OF_BEACONS),
                 activation="relu", name="encoder_3")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(
    encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# %%
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(math.ceil(0.3*NUMBER_OF_BEACONS),
                 activation="relu", name="decoder_0")(latent_inputs)
x = layers.Dense(math.ceil(0.5*NUMBER_OF_BEACONS),
                 activation="relu", name="decoder_1")(x)
x = layers.Dense(math.ceil(0.7*NUMBER_OF_BEACONS),
                 activation="relu", name="decoder_2")(x)
decoder_outputs = layers.Dense(NUMBER_OF_BEACONS, name="decoder_output")(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# %%


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # reconstruction_loss = tf.reduce_mean(
            #    tf.reduce_sum(
            #        tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #    )
            # )
            reconstruction_loss = tf.keras.losses.mse(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# %% [markdown]
# ### Training the model

# %%
EPOCHS = 150000
BATCH_SIZE = 8

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())

#es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500)
#history = vae.fit(train_features, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])
history = vae.fit(train_features, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

# %% [markdown]
# ### Generate new data from training data

# %%
df_train_locs = pd.read_csv(
    DATASET_ROOT+TRAIN_LOCATIONS_FILE, sep=';', names=['x', 'y'], dtype=np.float32)
print(df_train_locs)

# %%
# Controls how many samples will be added, make sure the square root of this is an int
MULTIPLICATION_FACTOR = 25
SAMPLE_VARIANCE = 0.025

positions = []
samples = []

for index, row in df_train_strengths.iterrows():
    x, y = df_train_locs.iloc[index]
    x = round(x, 1)
    y = round(y, 1)

    reshaped_row = np.reshape(row.values, (-1, NUMBER_OF_BEACONS))
    z_mean, _, _ = vae.encoder.predict(reshaped_row)
    sample = vae.decoder.predict(z_mean)
    samples.append(sample[0])
    positions.append((x, y))

    new_z_means = []
    for dx in np.linspace(-SAMPLE_VARIANCE, SAMPLE_VARIANCE, int(math.sqrt(MULTIPLICATION_FACTOR))):
        for dy in np.linspace(-SAMPLE_VARIANCE, SAMPLE_VARIANCE, int(math.sqrt(MULTIPLICATION_FACTOR))):
            new_z_means.append([z_mean[0][0] + dx, z_mean[0][1] + dy])

    new_samples = vae.decoder.predict(new_z_means)
    for new_sample in new_samples:
        samples.append(new_sample)
        positions.append((x, y))


# %%
# Data post processing
assert len(samples) == len(positions)
for i in range(len(samples)):
    assert len(samples[i]) == NUMBER_OF_BEACONS
    for j in range(NUMBER_OF_BEACONS):
        samples[i][j] = int(round(samples[i][j]))


# %%
# Replaced commas with dots
GENERATED_TRAIN_LOCATIONS_FILE = "SignatureLocs_altered_generated.csv"
GENERATED_TRAIN_STRENGTHS_FILE = "P_Signatures_generated.csv" if USE_BT else "P_SA_Signatures_generated.csv"

# %%
with open(DATASET_ROOT + GENERATED_TRAIN_LOCATIONS_FILE, 'w') as f:
    writer = csv.writer(f, delimiter=';')
    for pos in positions:
        writer.writerow(pos)

with open(DATASET_ROOT + GENERATED_TRAIN_STRENGTHS_FILE, 'w') as f:
    writer = csv.writer(f, delimiter=';')
    for sample in samples:
        writer.writerow(sample)
