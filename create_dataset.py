import sys
import wave
import os
import random
from copy import deepcopy
from typing import List, Tuple, Any

import librosa.effects

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from pedalboard import Pedalboard, Reverb
import mfe.dsp as dsp

from tqdm import tqdm

sample_rate = 16000

my_pedalboard = Pedalboard()
my_pedalboard.append(Reverb())


def time_shift(sample: np.ndarray, time: int):
    rolled_sample = np.roll(sample, time)
    return rolled_sample


def stretch(sample: np.ndarray, rate: float):
    input_length = sample_rate
    sample_as_float = sample.astype(float)
    stretched_sample = librosa.effects.time_stretch(sample_as_float, rate=rate)

    if len(stretched_sample) > input_length:
        data = stretched_sample[:input_length]
    else:
        data = np.pad(stretched_sample, (0, max(0, input_length - len(stretched_sample))), "constant")

    data_as_int16 = data.astype(np.int16)
    return data_as_int16


def add_noise(sample: np.ndarray, noise_amplitude: float = 0.005):
    wn = np.random.randn(len(sample))
    sample_wn = sample + noise_amplitude * wn

    return sample_wn


def augment(sample: np.ndarray, shift_value: int, stretch_value: float, noise_amplitude: float, add_reverb=False):
    copied_sample = deepcopy(sample)
    copied_sample = time_shift(copied_sample, shift_value)
    copied_sample = stretch(copied_sample, stretch_value)
    copied_sample = add_noise(copied_sample, noise_amplitude=noise_amplitude)

    if add_reverb:
        copied_sample = my_pedalboard(copied_sample, sample_rate=sample_rate)

    return copied_sample


def load_wav(path: str) -> np.ndarray:
    wavefile = wave.open(path, 'r')
    samples = wavefile.getnframes()
    channels = wavefile.getnchannels()

    audio = wavefile.readframes(samples)
    if channels == 2:
        print('Audio has 2 channels')
        audio = (audio[0] + audio[1])/2

    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    if len(audio_as_np_int16) > sample_rate:
        audio_as_np_int16 = audio_as_np_int16[:sample_rate]
    elif len(audio_as_np_int16) < sample_rate:
        audio_as_np_int16 = np.pad(audio_as_np_int16, (0, sample_rate - len(audio_as_np_int16)))

    return audio_as_np_int16


def save_wav(path: str, samplerate: int, audio: np.array) -> None:
    with wave.open(path, 'w') as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)
        wavfile.setframerate(samplerate)
        wavfile.writeframes(audio.tobytes())


def process_audio(audio: np.ndarray) -> np.ndarray:
    features = dsp.generate_features(4, False, audio, '0', sample_rate, 0.02, 0.01, 40, 512, 125, 7500, 101, -52)
    flat_mfe = np.array(features['features'])

    return flat_mfe


def process_label(label_path: str, label_array: List[float]) -> list[list[ndarray | list[float]]]:
    file_list = [os.path.join(label_path, file_name)
                 for file_name in tqdm(os.listdir(label_path), desc=f'Loading files', file=sys.stdout)]

    sample_list = [load_wav(path) for path in file_list]

    shift_values = np.arange(-1600, 1600, 1600 * 2 // 5)
    stretch_values = np.arange(0.9, 1.1, (1.1 - 0.9) / 5)
    noise_values = np.arange(0, 0.005, 0.005 / 5)

    random.shuffle(shift_values)
    random.shuffle(stretch_values)
    random.shuffle(noise_values)

    values = zip(shift_values, stretch_values, noise_values)
    augmented_list = list()

    for count, parameters in enumerate(values):
        shift_value, stretch_value, noise_value = parameters
        augmented_samples = [augment(sample, shift_value, stretch_value, noise_value, add_reverb=(count % 2 == 1))
                             for sample in tqdm(sample_list, desc=f'Augment pass {count}', file=sys.stdout)]
        augmented_list.extend(augmented_samples)

    sample_list.extend(augmented_list)

    label = label_path.split('/')[-1]
    output_path = 'output'

    if not os.path.exists(os.path.join(output_path, label)):
        os.mkdir(os.path.join(output_path, label))

    for count, audio in enumerate(sample_list):
        save_wav(os.path.join(output_path, label, f'{label}_{count}.wav'), sample_rate, audio)

    spectrogram_list = [[process_audio(sample), label_array]
                        for sample in tqdm(sample_list, desc=f'Processing samples', file=sys.stdout)]

    return spectrogram_list


def process_raw_dataset(dataset_name: str) -> tuple[list[list[Any]], list[list[Any]]]:
    dataset_folder = os.path.join('data', dataset_name)
    labels = os.listdir(dataset_folder)
    training_list = list()
    testing_list = list()

    for pos, label in enumerate(labels):
        print(f'Processing {label}')
        label_array = [0.] * len(labels)
        label_array[pos] = 1.

        label_path = os.path.join(dataset_folder, label)

        spectrogram_list = process_label(label_path, label_array)

        test_train_separation = int(len(spectrogram_list)*(80/100))

        training_list.extend(spectrogram_list[:test_train_separation])
        testing_list.extend(spectrogram_list[test_train_separation:])

    return training_list, testing_list


def process_divided_dataset(dataset_folder: str) -> tuple[list[list[Any]], list[list[Any]]]:
    labels_testing = os.listdir(os.path.join(dataset_folder, 'training'))
    labels_training = os.listdir(os.path.join(dataset_folder, 'testing'))

    if labels_training != labels_testing:
        raise ValueError('Labels in testing and training folders do not match')

    labels = labels_training

    training_list = list()
    for pos, label in enumerate(labels):
        print(f'Processing {label} for training')
        label_array = [0.] * len(labels)
        label_array[pos] = 1.

        label_path = os.path.join(dataset_folder, 'training', label)

        spectrogram_list = process_label(label_path, label_array)
        training_list.extend(spectrogram_list)

    testing_list = list()
    for pos, label in enumerate(labels):
        print(f'Processing {label} for testing')
        label_array = [0.] * len(labels)
        label_array[pos] = 1.

        label_path = os.path.join(dataset_folder, 'testing', label)

        spectrogram_list = process_label(label_path, label_array)
        testing_list.extend(spectrogram_list)

    return training_list, testing_list


def main_function(dataset_name: str):
    dataset_folder = os.path.join('data', dataset_name)
    folders = os.listdir(dataset_folder)
    if len(folders) == 2 and 'testing' in folders and 'training' in folders:
        print('Found a manually separated dataset')
        training_list, testing_list = process_divided_dataset(dataset_folder)

    else:
        print('Found an non separated dataset, will perform automatic separation into training and testing')
        training_list, testing_list = process_raw_dataset(dataset_name)

    x_train_save_path = os.path.join('data', 'X_split_train.npy')
    y_train_save_path = os.path.join('data', 'Y_split_train.npy')
    x_test_save_path = os.path.join('data', 'X_split_test.npy')
    y_test_save_path = os.path.join('data', 'Y_split_test.npy')

    random.shuffle(training_list)
    random.shuffle(testing_list)

    x_split_train = [sample[0] for sample in training_list]
    y_split_train = [sample[1] for sample in training_list]
    x_split_test = [sample[0] for sample in testing_list]
    y_split_test = [sample[1] for sample in testing_list]

    np.save(x_train_save_path, np.asarray(x_split_train).astype('float32'))
    np.save(y_train_save_path, np.asarray(y_split_train).astype('float32'))
    np.save(x_test_save_path, np.asarray(x_split_test).astype('float32'))
    np.save(y_test_save_path, np.asarray(y_split_test).astype('float32'))


if __name__ == '__main__':
    main_function('xpons_dataset')
    x_train = np.load('data/X_split_train.npy')
    y_train = np.load('data/Y_split_train.npy')
    x_test = np.load('data/X_split_test.npy')
    y_test = np.load('data/Y_split_test.npy')
    print(f'x_train shape: {x_train.shape}, dtype: {x_train.dtype}')
    print(f'y_train shape: {y_train.shape}, dtype: {y_train.dtype}')
    print(f'x_test shape: {x_test.shape}, dtype: {x_test.dtype}')
    print(f'y_test shape: {y_test.shape}, dtype: {y_test.dtype}')

    '''edge_impulse_test = np.load('edge_impulse_test.npy')
    edge_impulse_train = np.load('edge_impulse_train.npy')
    print(f'edge_impulse_test shape: {edge_impulse_test.shape}, dtype: {edge_impulse_test.dtype}')

    image_1 = x_train[0]
    restored_image_1 = image_1.reshape(99, 40).T
    ei_image_1 = edge_impulse_test[0]
    ei_restored_image_1 = ei_image_1.reshape(99, 40).T

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(restored_image_1)
    axs[0].set_title('custom')
    axs[1].imshow(ei_restored_image_1)
    axs[1].set_title('edge impulse')
    plt.tight_layout()
    plt.show()'''

    '''image_2 = x_train[1]
    restored_image_2 = image_2.reshape(99, 40).T
    ei_image_2 = edge_impulse_test[0]
    ei_restored_image_2 = ei_image_2.reshape(99, 40).T

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(restored_image_2)
    axs[1].imshow(ei_restored_image_2)
    plt.show()'''
