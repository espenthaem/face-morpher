from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Morpher:
    def __init__(self):
        self.encoder = load_model('models/vae_encoder.h5')
        self.decoder = load_model('models/vae_decoder.h5')

        self.input_shape = self.encoder.layers[0].input_shape[1:]
        self.latent_dim = self.decoder.layers[-1].input_shape[1]

    def prepare_input(self, input_path):

        input_img = cv2.imread(input_path, 0)
        input_img = cv2.resize(input_img, self.input_shape[:-1])
        input_img = input_img.reshape(self.input_shape) / 127.5 - 1

        return input_img

    def sample(self, loc=0.0, std=0.5):

        z = np.random.normal(loc, std, size=(1, self.latent_dim))

        return z

    def map(self, image):

        assert image.shape == self.input_shape

        return self.encoder.predict(np.array([image]))[2]

    def generate_face(self, latent_img):

        return self.decoder.predict(latent_img)

    def morph_face(self, latent_img, morph_value=0.10, axis=0):

        latent_img[0, axis] += morph_value

        return self.generate_face(latent_img)


if __name__ == '__main__':
    from PIL import Image


    morpher = Morpher()
    example = morpher.prepare_input('examples/celebA_7.jpg')

    latent_img = morpher.map(example)
    decoded_face = morpher.generate_face(latent_img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.gray()
    axes[0].imshow(decoded_face.reshape(256, 256))
    axes[1].imshow(example.reshape(256, 256))

    morph_frames = []
    n = 25
    morph_axis = np.random.randint(latent_img.shape[1])
    morph_constant = 7.5
    for i in range(n):
        latent_img[:,morph_axis] += i/n * morph_constant
        morphed_face = morpher.generate_face(latent_img)
        scaled = (morphed_face.reshape(256, 256) + 1) * 127.5
        rgb_array = cv2.cvtColor(scaled.astype('uint8'), cv2.COLOR_GRAY2RGB)
        morph_frames.append(Image.fromarray(rgb_array))

    fps = 5
    morph_frames[0].save('examples/morphing.gif',
                         format='GIF',
                         append_images=morph_frames[1:],
                         save_all=True,
                         duration=1000 / fps,
                         loop=0)
