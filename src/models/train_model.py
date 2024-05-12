import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from src.data.make_dataset import read_image, read_mask

# Define Callbacks
CALLBACKS = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False),
]


def model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape, name="input_image")

    encoder = MobileNetV2(
        input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35
    )
    skip_connection_names = [
        "input_image",
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
    ]
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names) + 1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    return model


def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def compile_model(model, learning_rate=1e-4):
    opt = tf.keras.optimizers.Nadam(learning_rate)
    metrics = [dice_coef, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)
    return model


def train_model(
    model,
    train_dataset,
    valid_dataset,
    epochs,
    train_steps,
    valid_steps,
    callbacks=CALLBACKS,
):
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
    )
