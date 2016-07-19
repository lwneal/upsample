from __future__ import print_function
import sys
import os
from PIL import Image
import numpy as np

def train(directory):
    files = sorted(os.listdir(directory))
    prev_frame = np.zeros((1, 3, 360,640))
    target_frame = np.zeros((1, 3, 360,640))
    model = build_model()
    for i, filename in enumerate(files):
        next_frame = load_image(os.path.join(directory, filename))
        model_input = np.concatenate([prev_frame, next_frame], axis=1)
        model.train_on_batch(model_input, target_frame)
        model.predict(model_input)
        print('Trained on frame {}'.format(filename))
        prev_frame = target_frame
        target_frame = next_frame
    model.save_weights('weighty.h5')

def test(input_directory, output_directory, upsample=True):
    files = sorted(os.listdir(input_directory))
    prev_frame = np.zeros((1, 3, 360, 640))
    model = build_model()
    model.load_weights('weighty.h5')
    i = 0
    for filename in files:
        next_frame = load_image(os.path.join(input_directory, filename))
        model_input = np.concatenate([prev_frame, next_frame], axis=1)
        output = model.predict(model_input)
        save_image(output, '{}/{:05d}.jpg'.format(output_directory, i))
        save_image(next_frame, '{}/{:05d}.jpg'.format(output_directory, i+1))
        prev_frame = next_frame
        i += 2
    print("Finished")

def load_image(filepath):
    img = Image.open(filepath)
    img.load()
    return np.array(img).transpose(2,0,1).reshape((1,3,360,640))

def save_image(img, filepath):
    pixels = img.reshape(3,360,640).transpose(1,2,0).clip(0,255)
    Image.fromarray(pixels.astype(np.uint8)).save(filepath)
    print('Saved image {}'.format(filepath))


def build_model():
    import keras
    from keras.models import Sequential
    from keras.layers import ZeroPadding2D, Convolution2D
    from keras.layers import Activation
    model = keras.models.Sequential()
    # Two stacked frames
    model.add(ZeroPadding2D(padding=(2,2), input_shape=(6, 360, 640)))
    model.add(Convolution2D(3, 5, 5))
    model.add(Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'test':
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        test(input_dir, output_dir)
    elif cmd == 'train':
        train(sys.argv[2])
