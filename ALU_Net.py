#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/domschl/ALU_Net/blob/main/ALU_Net.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # A neural net that tries to become an ALU (arithmetic logic unit)
# 
# This notebook can run
# 
# - on local jupyter instances with a local graphics card
# - on Mac M1 with local jupyter instance and [Apple's tensorflow-plugin](https://developer.apple.com/metal/tensorflow-plugin/)
# - on Google Colab instances with either GPU or TPU runtime. The colab version uses a Google Drive account to cache data and model state within a Google Drive directory `My Drive/Colab Notebooks/ALU_Net`.

# ## 1. Configuration and setup

import tensorflow as tf

use_keras_project_versions=False
# Namespaces, namespaces
if use_keras_project_versions is False:
    # print("Importing Keras from tensorflow project (it won't work otherwise with TPU)")
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks, metrics, optimizers
else:
    # print("Importing Keras from keras project (which had recently declared independence [again]) -- as recommended")
    use_keras_project_versions=True
    import keras
    from keras import layers, regularizers, callbacks, metrics, optimizers

import ALU_Tools

def model_large(inputs, regu1=1e-7, regu2=1e-7, regu3=1e-7, neurons=1024, filters=16, strides=2, kernel_size=3):  # neurons must be divisible by 4 for the residual below
    # Input goes parallel into 3 streams which will be combined at the end:
    # Stream 1: convolutions
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs)  # x0)
    d1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    x1 = d1(rinp)
    filters=filters*2
    d2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    x2 = d2(x1)
    filters=filters*2
    d3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    x3 = d3(x2)
    filters=filters*2
    d4 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xcvl = d4(x3)
    flatter = layers.Flatten()
    xf = flatter(xcvl)
    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xe1 = de1(xf)

    # Stream 2: simple dense layers
    df1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xf1 = df1(inputs)
    df2 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xf2 = df2(xf1)
    df3 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xf3 = df3(xf2)

    # Stream3: dense layers with residual pathway
    dfa1 = layers.Dense(neurons/4, kernel_regularizer=regularizers.l2(
        regu3), activation="relu")
    xfa1 = dfa1(inputs)
    dfa2 = layers.Dense(neurons/4, kernel_regularizer=regularizers.l2(
        regu3), activation="relu")
    con0 = layers.Concatenate()
    xfa2 = con0([dfa2(xfa1),xfa1])
    dfa3 = layers.Dense(neurons/2, kernel_regularizer=regularizers.l2(regu3), activation="relu")
    con1 = layers.Concatenate()
    xfa3 = con1([dfa3(xfa2),xfa2])

    # Concat of stream1,2,3
    con = layers.Concatenate()
    xcon = con([xe1, xf3, xfa3])
    dc1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xc1 = dc1(xcon)

    # Use sigmoid to map to bits 0..1
    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xc1)
    return outputs

def model_medium(inputs, regu1=1e-7, regu2=1e-7, neurons=256, lstm_neurons=128, filters=64, kernel_size=3, strides=2):
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs) # xf1)
    d1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x1 = d1(rinp)
    filters = 2*filters
    d2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x2 = d2(x1)
    filters = 2*filters
    d3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x3 = d3(x2)

    flatter = layers.Flatten()
    xf = flatter(x3)

    r1 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr1 = r1(rinp)
    r2 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr2 = r2(xr1)
    r3 = layers.LSTM(lstm_neurons)
    xr3 = r3(xr2)

    cc = layers.Concatenate()
    xc = cc([xf, xr3])

    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    xe1 = de1(xc)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xe1)
    return outputs

def model_minimal(inputs, neurons=64, regu1=1e-7):
    df1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xf1 = df1(inputs)
    df2 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xf2 = df2(xf1)
    df3 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xf3 = df3(xf2)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xf3)
    return outputs

def model_minimal_recurrent(inputs, neurons=64, lstm_neurons=128, regu1=1e-7):
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs)
    r1 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr1 = r1(rinp)
    r2 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr2 = r2(xr1)
    r3 = layers.LSTM(lstm_neurons, return_sequences=True)
    xr3 = r3(xr2)

    flatter = layers.Flatten()
    xf = flatter(xr3)

    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xe1 = de1(xf)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xe1)
    return outputs

def model_conv1d_recurrent(inputs, neurons=512, lstm_neurons=386, filters=128, kernel_size=3, strides=2, regu0=1e-7, regu1=1e-7, regu2=1e-7):
    shaper = layers.Reshape(target_shape=(36, 1,), input_shape=(36,))
    rinp = shaper(inputs)

    r1 = layers.LSTM(lstm_neurons, return_sequences=True, kernel_regularizer=regularizers.l2(
        regu0), recurrent_regularizer=regularizers.l2(
        regu0))
    xr1 = r1(rinp)
    
    d1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(
        regu2), activation="relu")
    x1 = d1(xr1)
    
    r2 = layers.LSTM(lstm_neurons, return_sequences=True, kernel_regularizer=regularizers.l2(
        regu0), recurrent_regularizer=regularizers.l2(
        regu0))
    xr2 = r2(x1)

    d2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(regu2), activation="relu")
    x2 = d2(xr2)

    r3 = layers.LSTM(lstm_neurons, return_sequences=True, kernel_regularizer=regularizers.l2(
        regu0), recurrent_regularizer=regularizers.l2(
        regu0))
    xr3 = r3(x2)

    d3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(regu2), activation="relu")
    x3 = d3(xr3)

    flatter = layers.Flatten()
    xf = flatter(x3)

    de1 = layers.Dense(neurons, kernel_regularizer=regularizers.l2(
        regu1), activation="relu")
    xe1 = de1(xf)

    de2 = layers.Dense(32, activation="sigmoid")
    outputs = de2(xe1)
    return outputs

def create_load_model(save_path=None, model_variant=None):
    """ Create or load a model """
    if save_path is None or not os.path.exists(save_path): #or is_tpu is True:
        print("Initializing new model...")
        inputs = keras.Input(shape=(36,))  # depends on encoding of op-code!
        if model_variant not in model_variants:
            print('Unkown model type')
            return None
        outputs = model_variants[model_variant](inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name="maths_"+model_variant)
        print(f"Compiling new model of type {model_variant}")
        if use_keras_project_versions is False: 
            opti = keras.optimizers.Adam(learning_rate=0.001)
        else:
            opti = optimizers.Adam(learning_rate=0.001)
        model.compile(loss="mean_squared_error", optimizer=opti, metrics=[metrics.MeanSquaredError(), 'accuracy'])
    else:
        print(f"Loading standard-format model of type {model_variant} from {model_save_dir}")
        model = tf.keras.models.load_model(save_path)
        print("Continuing training from existing model")
    model.summary()
    return model

def get_model(ml_env, save_path=None, on_tpu=False, model_variant=None, import_weights=False):
    if on_tpu is True:
        if ml_env.tpu_is_init is False:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=ml_env.tpu_address)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)    
            ml_env.tpu_is_init=True
        with tpu_strategy.scope():
            print("Creating TPU-scope model")
            model=create_load_model(save_path=None, model_variant=model_variant)
        if ml_env.weights_file is not None and os.path.exists(ml_env.weights_file):
            print("Injecting saved weights into TPU model, loading...")
            temp_model = create_load_model(save_path=None, model_variant=model_variant)
            temp_model.load_weights(ml_env.weights_file)
            print("Injecting...")
            model.set_weights(temp_model.get_weights())
            print("Updated TPU weights from saved model")
        return model
    else:
        print("Creating standard-scope model")
        model = create_load_model(save_path=save_path, model_variant=model_variant)
        if import_weights is True and ml_env.weights_file is not None and os.path.exists(ml_env.weights_file):
            print("Injecting saved weights into model, loading...")        
            model.load_weights(ml_env.weights_file)
            imported_weights_file = ml_env.weights_file+'-imported'
            os.rename(ml_env.weights_file, imported_weights_file)
            print(f"Renamed weights file {ml_env.weights_file} to {imported_weights_file} to prevent further imports!")
        return model


def math_train(mlenv:MLEnv, model, dataset, validation, batch_size=8192, epochs=5000, steps_per_epoch=2000, log_path="./logs"):
    """ Training loop """
    interrupted = 2
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=log_path,
        histogram_freq=1,
        update_freq='batch'
        )
    lambda_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end = ml_env.epoch_time_func
    )
    try:
        if ml_env.is_tpu:
            model.fit(dataset, validation_data=validation, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[tensorboard_callback, lambda_callback])
            interrupted=0
        else:
            model.fit(dataset, validation_data=validation, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard_callback, lambda_callback])
            interrupted=0
    except KeyboardInterrupt:
        print("")
        print("")
        print("---------INTERRUPT----------")
        print("")
        print("Training interrupted")
        interrupted = 1 # user stopped runtime
    except Exception as e:
        interruped = 2  # Bad: something crashed.
        print(f"INTERNAL ERROR")
        print(f"Exception {e}")
    finally:
        return interrupted

def instantiate_models(ml_env:MLEnv, save_path, model_variant, import_weights=True):
    if ml_env.is_tpu:
        # Generate a second CPU model for testing:
        test_model = get_model(ml_env, save_path=None, on_tpu=False, model_variant=model_variant)
        math_model = get_model(ml_env, save_path=save_path, on_tpu=True, model_variant=model_variant)
    else:
        test_model = None
        math_model = get_model(ml_env, save_path=save_path, on_tpu=False, model_variant=model_variant, import_weights=import_weights)
    return math_model, test_model

def do_training(mlenv:MLEnv, math_model, training_dataset, validation_dataset, math_data, epochs_per_cycle, model_save_dir=None, 
                weights_file=None, test_model=None, cycles=100, steps_per_epoch=1000, reweight_size=1000, valid_ops=None, regenerate_data_after_cycles=0, data_func=None,
                log_path='./logs'):
    # Training
    for mep in range(0, cycles):
        print()
        print()
        print(f"------ Meta-Epoch {mep+1}/{cycles} ------")
        print()
        if regenerate_data_after_cycles!=0 and data_func is not None:
            if mep>0 and (mep+1)%regenerate_data_after_cycles==0:
                math_data, training_dataset, validation_dataset = data_func()
        if mep==0 and ml_env.is_tpu is True:
            print("There will be some warnings by Tensorflow, documenting some state of internal decoherence, currently they can be ignored.")
        interrupted = math_train(ml_env, math_model, training_dataset, validation=validation_dataset, epochs=epochs_per_cycle, steps_per_epoch=steps_per_epoch, log_path=log_path)
        if interrupted <2:
            if ml_env.is_tpu:
                if test_model is None:
                    print("Fatal: tpu-mode needs test_model on CPU")
                    return False
                print("Injecting weights into test_model:")
                test_model.set_weights(math_model.get_weights())
                if weights_file is not None:
                    print(f"Saving test-model weights to {weights_file}")
                    test_model.save_weights(weights_file)
                    print("Done")
                print(f"Checking {reweight_size} datapoints for accuracy...")
                math_data.check_results(test_model, samples=reweight_size, short_math=False, valid_ops=valid_ops, verbose=False)
            else:
                if model_save_dir is not None:
                    print("Saving math-model")
                    math_model.save(model_save_dir)
                    print("Done")
                print(f"Checking {reweight_size} datapoints for accuracy...")
                math_data.check_results(math_model, samples=reweight_size, short_math=False, valid_ops=valid_ops, verbose=False)
        if interrupted>0:
            break


model_variants = {"large": model_large,
                  "medium": model_medium,
                  "minimal": model_minimal,
                  "minimal_recurrent": model_minimal_recurrent,
                  "conv_recurrent": model_conv1d_recurrent
            }

model_variant='minimal'  # see: model_variants definition.
epochs_per_cycle=250
cycles = 100  # perform 100 cycles, each cycle trains with epochs_per_cycle epochs.
regenerate_data_after_cycles=2  # if !=0, the training data will be created anew after each number of 
                                # regenerace_data_after_cycles cycles. Disadvantage: when training TPU, 
                                # Google might use the time it takes to regenerate to training data to 
                                # terminate your session :-/
samples=2000000  # Number training data examples
batch_size=20000 
valid_ops=None  # Default: None (all ops), or list of ops, e.g. ['*', '/'] trains only multiplication and division.
steps_per_epoch=samples//batch_size  # TPU stuff

ml_env=MLEnv()
math_data=ALU_Dataset(ml_env)

model_save_dir, weights_file, cache_stub, log_path = ml_env.init_paths("ALU_Net", "math_model", model_variant=model_variant, log_to_gdrive=True)

train, val = math_data.get_datasets(pre_weight=True, samples=samples, validation_samples=50000, batch_size=batch_size, short_math=False, 
                                     valid_ops=valid_ops, cache_file_stub=cache_stub, use_cache=True, regenerate_cached_data=False)
math_model, test_model = instantiate_models(ml_env, model_save_dir, model_variant, import_weights=True)


try:
    # use the python variable log_path:
    get_ipython().run_line_magic('tensorboard', '--logdir "{log_path}"')
except:
    pass

do_training(ml_env, math_model, train, val, math_data, epochs_per_cycle, model_save_dir=model_save_dir, 
            weights_file=weights_file, test_model=test_model, cycles=cycles, steps_per_epoch=steps_per_epoch, valid_ops=valid_ops, 
            regenerate_data_after_cycles=regenerate_data_after_cycles, data_func=None, log_path=log_path)

