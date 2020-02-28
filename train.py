import glob
import argparse
import warnings
import numpy as np
from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from generator import celebGenerator


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def construct_model(input_size, latent_dim):

    def vae_loss(input_img, decoded):
        reconstruction_loss = K.sum(K.square(decoded - input_img))
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    K.clear_session()

    input_img = Input(shape=input_size)  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    intermediate_shape = x._keras_shape[1:]
    intermediate_dim = intermediate_shape[0] * intermediate_shape[1] * intermediate_shape[2]
    x = Flatten()(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')

    latent_img = Input(shape=(latent_dim,))
    x = Dense(intermediate_dim)(latent_img)
    x = Reshape(intermediate_shape)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

    decoder = Model(latent_img, decoded, name='decoder')

    vae = Model(input_img, decoder(encoder(input_img)[2]), name='vae_mlp')
    vae.compile(optimizer="adam", loss=vae_loss)

    return encoder, decoder, vae


def prepare_data(input_size, batch_size, celebA_path, selfie_path, split=0.75):
    imgs = np.array(glob.glob(celebA_path + '/*.jpg'))
    if len(imgs) == 0:
        warnings.warn("celebA path yielded no images")

    if selfie_path is not None:
        selfie_imgs = np.array(glob.glob(selfie_path + '/*.jpg'))
        if len(selfie_imgs) == 0:
            warnings.warn("selfie path yielded no images")
        imgs = np.hstack([imgs, selfie_imgs])

    train_indices = np.random.choice(len(imgs), int(len(imgs) * split), replace=False)
    test_indices = list(set(range(len(imgs))) - set(train_indices))

    train_imgs = imgs[train_indices]
    test_imgs = imgs[test_indices]

    train_generator = celebGenerator(train_imgs, batch_size, input_size)
    test_generator = celebGenerator(test_imgs, batch_size, input_size)

    return train_generator, test_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=128,
                        help='image size, assuming square images. Default argument is 128')
    parser.add_argument('--latent-dim', type=int, default=512,
                        help="latent dimension")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of workers")
    parser.add_argument('--model-dir', type=str, default='models',
                        help="path to model folder")
    parser.add_argument('--celeb-dir', type=str, default='/data/celebA',
                        help="path to celebA images")
    parser.add_argument('--selfie-dir', type=str, default=None,
                        help="path to selfie images")

    args = parser.parse_args()
    input_size = (args.image_size, args.image_size, 1)

    encoder, decoder, variational_autoencoder = construct_model(input_size, args.latent_dim)

    print(encoder.summary())
    print(decoder.summary())
    train, test = prepare_data(input_size, args.batch_size, args.celeb_dir, args.selfie_dir)
    
    variational_autoencoder.fit_generator(generator=train,
                                          validation_data=test,
                                          epochs=args.epochs,
                                          shuffle=True,
                                          use_multiprocessing=True,
                                          workers=args.workers)

    encoder.save(args.out_dir + '/vae_encoder.h5')
    decoder.save(args.out_dir + '/vae_decoder.h5')

