if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend import tensorflow_backend as K

    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    from nevermind import configurations

    configurations.train_atari(render=False)
