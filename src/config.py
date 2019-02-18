#!/usr/bin/env python3

class image_properties(object):
    depth = 3
    height = 256
    width = 256
    compressed_dims = [1, height/16, width/16, 10]
    
class config_train(object):
    train_fraction = 0.9
    mode = 'gan-train'
    num_epochs = 40
    batch_size = 4
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 50

    perceptual_coeff = 0.2

    # Compression
    lambda_X = 12   # Distortion Penalty
    channel_bottleneck = 10
    sample_noise = False
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = True
    feature_matching_weight = 10
    use_conditional_GAN = False

class config_test(object):
    mode = 'gan-test'
    num_epochs = 512
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 256

    perceptual_coeff = 0.2

    # Compression
    lambda_X = 12
    channel_bottleneck = 10
    sample_noise = False
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = True
    feature_matching_weight = 10
    use_conditional_GAN = False

class directories(object):
    train = 'data/paths_train.d5'
    test = 'data/paths_test.d5'
    val = 'data/paths_validation.d5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    samples = 'output/'

