# %%
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
tf.config.run_functions_eagerly(False)

# %%
USE_BT = True


DATASET_ROOT = "data/"
TRAIN_LOCATIONS_FILE = "SignatureLocs_altered.csv"
GENERATED_TRAIN_LOCATIONS_FILE = "SignatureLocs_altered_generated.csv"
TEST_LOCATIONS_FILE = "TestLocs_altered.csv"
TRAIN_STRENGTHS_FILE = "P_Signatures.csv" if USE_BT else "P_SA_Signatures.csv"
GENERATED_TRAIN_STRENGTHS_FILE = "P_Signatures_generated.csv" if USE_BT else "P_SA_Signatures_generated.csv"
TEST_STRENGTHS_FILE = "P_Tests.csv" if USE_BT else "P_SA_Tests.csv"
NUMBER_OF_BEACONS = 57 if USE_BT else 11

# %%
df_train_strengths = pd.read_csv(
    DATASET_ROOT+TRAIN_STRENGTHS_FILE, sep=';', names=[x for x in range(NUMBER_OF_BEACONS)])
df_train_locs = pd.read_csv(
    DATASET_ROOT+TRAIN_LOCATIONS_FILE, sep=';', names=['x', 'y'], dtype=float)
df_generated_train_strengths = pd.read_csv(
    DATASET_ROOT+TRAIN_STRENGTHS_FILE, sep=';', names=[x for x in range(NUMBER_OF_BEACONS)])
df_generated_train_locs = pd.read_csv(
    DATASET_ROOT+TRAIN_LOCATIONS_FILE, sep=';', names=['x', 'y'], dtype=float)
df_test_strengths = pd.read_csv(
    DATASET_ROOT+TEST_STRENGTHS_FILE, sep=';', names=[x for x in range(NUMBER_OF_BEACONS)])
df_test_locs = pd.read_csv(
    DATASET_ROOT+TEST_LOCATIONS_FILE, sep=';', names=['x', 'y'], dtype=float)

train_features = df_train_strengths
train_target = df_train_locs
generated_train_features = df_train_strengths
generated_train_target = df_train_locs
test_features = df_test_strengths
test_target = df_test_locs

normalization_values = np.array(train_features)

# %% [markdown]
# ### Define all models
# Currently:
# - 3 hidden layer mixed with dropout layers (basic_model) (deep-learn-paper)

# %%


def gen_basic_model():
    normalizer = layers.Normalization()
    normalizer.adapt(normalization_values)

    model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(500, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(500, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='relu'),
        layers.Dropout(0.5)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=optimizer,
                  loss=loss_function)

    return model


# %%
# Define more models
def gen_simple_model():
    normalizer = layers.Normalization()
    normalizer.adapt(normalization_values)

    model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(math.ceil((2//3) * NUMBER_OF_BEACONS), activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(math.ceil((2//3) * NUMBER_OF_BEACONS), activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='relu'),
        layers.Dropout(0.5)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=optimizer,
                  loss=loss_function)

    return model

# %%
# Paper: Improved Indoor Geomagnetic Field Fingerprinting for Smartwatch Localization Using Deep Learning


def homayani_conv_model():
    normalizer = layers.Normalization()
    normalizer.adapt(normalization_values)

    model = tf.keras.models.Sequential([
        normalizer,
        layers.Reshape((NUMBER_OF_BEACONS, 1)),
        layers.Conv1D(16, 3, activation='relu'),
        layers.Conv1D(32, 3, activation='relu'),
        layers.Dense(603, activation='relu'),  # Equal to measured locations
        layers.Flatten(),
        layers.Dense(2, activation='relu')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=optimizer,
                  loss=loss_function)

    return model

# %%
# Paper: A Comparison Analysis of BLE-Based Algorithms forLocalization in Industrial Environments


def cannizzaro_mlp():
    normalizer = layers.Normalization()
    normalizer.adapt(normalization_values)

    model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(8),  # Input after normalization
        layers.Dense(8, activation='relu'),
        layers.Dense(6, activation='relu'),
        layers.Dense(2, activation='relu'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=optimizer,
                  loss=loss_function)

    return model

# %%
# Paper: CNNLoc: Deep-Learning Based Indoor Localization with WiFi Fingerprinting


def cnn_loc():
    normalizer = layers.Normalization()
    normalizer.adapt(normalization_values)

    model = tf.keras.models.Sequential([
        normalizer,
        layers.Dense(math.ceil(NUMBER_OF_BEACONS * 2.5), activation='elu'),
        layers.Dense(math.ceil(NUMBER_OF_BEACONS * 1.25), activation='elu'),
        layers.Reshape((math.ceil(NUMBER_OF_BEACONS * 1.25), 1)),
        layers.Dropout(0.7),
        layers.Conv1D(math.ceil(NUMBER_OF_BEACONS * 0.5),
                      math.ceil(NUMBER_OF_BEACONS * 0.12)),
        layers.Conv1D(math.ceil(NUMBER_OF_BEACONS * 0.4),
                      math.ceil(NUMBER_OF_BEACONS * 0.12)),
        layers.Conv1D(math.ceil(NUMBER_OF_BEACONS * 0.3),
                      math.ceil(NUMBER_OF_BEACONS * 0.12)),
        layers.Flatten(),
        layers.Dense(2, activation='elu')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=optimizer,
                  loss=loss_function)

    return model

# %% [markdown]
# ### Train and evaluate all models in the same way


# %%
model_generators = [
    (gen_basic_model, 'Basic model, 3 hidden layers (500) with dropout (0.5)'),
    (gen_simple_model,
     'Simple model, 3 hidden layers 2/3 * #_{APs} with dropout (0.5)'),
    (homayani_conv_model, 'Basic 1D convolutional network, 2 conv layers and 1 dense'),
    (cannizzaro_mlp, 'Simple MLP, 3 hidden layers (8, 8, 6)'),
    (cnn_loc, 'CNNLoc, SAE + Conv layers'),
]
EPOCHS = 30000
BATCH_SIZE = 8


models_default = [m[0]() for m in model_generators]
models_generated = [m[0]() for m in model_generators]

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000)
for i, model in enumerate(models_default):
    print(f"Training model '{model_generators[i][1]}' on original data")
    model.fit(train_features, train_target, epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[es])

for i, model in enumerate(models_generated):
    print(f"Training model '{model_generators[i][1]}' on generated data")
    model.fit(train_features, train_target, epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[es])

# %%
for i, model in enumerate(models_default):
    print(f"Evaluating original data model '{model_generators[i][1]}'")
    model.evaluate(test_features, test_target)

for i, model in enumerate(models_generated):
    print(f"Evaluating generated data model '{model_generators[i][1]}'")
    model.evaluate(test_features, test_target)
