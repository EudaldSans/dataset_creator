import copy
import csv
import os.path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import chirp

from create_dataset import load_wav, filter_audio, process_audio


def test_ei():
    x_train = np.load('data/X_split_train.npy')
    y_train = np.load('data/Y_split_train.npy')
    x_test = np.load('data/X_split_test.npy')
    y_test = np.load('data/Y_split_test.npy')
    ei_marvin_file = np.load('resources/test_mfe/marvin.npy')
    ei_happy_file = np.load('resources/test_mfe/happy.npy')
    ei_eight_file = np.load('resources/test_mfe/eight.npy')

    print(f'x_train shape: {x_train.shape}, dtype: {x_train.dtype}')
    print(f'y_train shape: {y_train.shape}, dtype: {y_train.dtype}')
    print(f'x_test shape: {x_test.shape}, dtype: {x_test.dtype}')
    print(f'y_test shape: {y_test.shape}, dtype: {y_test.dtype}')
    print(f'marvin: {ei_marvin_file.shape}, dtype: {ei_marvin_file.dtype}')
    print(f'happy: {ei_happy_file.shape}, dtype: {ei_happy_file.dtype}')
    print(f'eight: {ei_eight_file.shape}, dtype: {ei_eight_file.dtype}')

    '''flat_marvin = ei_marvin_file[0]
    marvin = flat_marvin.reshape(99, 40).T
    plt.imshow(marvin)
    plt.title('Our Marvin')
    plt.tight_layout()
    plt.show()

    flat_happy = ei_happy_file[0]
    happy = flat_happy.reshape(99, 40).T
    plt.imshow(happy)
    plt.title('Our happy')
    plt.tight_layout()
    plt.show()

    flat_eight = ei_eight_file[0]
    eight = flat_eight.reshape(99, 40).T
    plt.imshow(eight)
    plt.title('Our eight')
    plt.tight_layout()
    plt.show()'''

    '''image_2 = x_train[0]
    restored_image_2 = image_2.reshape(99, 40).T
    fig, axs = plt.subplots(2, 1)

    axs[0].imshow(marvin)
    axs[0].set_title('ei')
    axs[1].imshow(restored_image_2)
    axs[1].set_title('ours')
    plt.tight_layout()
    plt.show()

    image_2 = x_train[1]
    restored_image_2 = image_2.reshape(99, 40).T
    fig, axs = plt.subplots(2, 1)

    axs[0].imshow(marvin)
    axs[0].set_title('ei')
    axs[1].imshow(restored_image_2)
    axs[1].set_title('ours')
    plt.tight_layout()
    plt.show()

    image_2 = x_test[0]
    restored_image_2 = image_2.reshape(99, 40).T
    fig, axs = plt.subplots(2, 1)

    axs[0].imshow(marvin)
    axs[0].set_title('ei')
    axs[1].imshow(restored_image_2)
    axs[1].set_title('ours')
    plt.tight_layout()
    plt.show()'''

    flat_apaga = x_test[0]
    apaga = flat_apaga.reshape(99, 40).T
    plt.imshow(apaga)
    plt.title('I2CAT Apaga')
    plt.tight_layout()
    plt.show()

    np.savetxt("i2cat_apaga.csv", flat_apaga, delimiter=',')


def test_filter():
    filter_df = pd.read_excel(os.path.join('resources', 'filter_coefficients.xlsx'))
    # seconds_of_tone = 1

    # waveform_time = np.linspace(0, seconds_of_tone, 16000 * seconds_of_tone)

    # original_wav = chirp(waveform_time, f1=8000, f0=100, t1=seconds_of_tone, method='linear') * 100
    # original_wav = np.sin(2 * np.pi * waveform_time * 2000 / 16000) * 1000

    original_wav = load_wav(os.path.join('resources', 'pink_noise.wav'))
    audacity_wav = load_wav(os.path.join('resources', 'filtered_pink_noise.wav'))

    filtered_wav = copy.deepcopy(original_wav)
    filtered_wav = filter_audio(filtered_wav, 16000, filter_df)

    original_spectrogram = process_audio(original_wav)
    filtered_spectrogram = process_audio(filtered_wav)
    audacity_spectrogram = process_audio(audacity_wav)

    original_spectrogram = np.flip(original_spectrogram.reshape(99, 40).T, axis=1)
    filtered_spectrogram = np.flip(filtered_spectrogram.reshape(99, 40).T, axis=1)
    audacity_spectrogram = np.flip(audacity_spectrogram.reshape(99, 40).T, axis=1)

    fig, axs = plt.subplots(1, 3)

    axs[0].specgram(original_wav, NFFT=1024, Fs=16000)
    axs[0].set_title('original')
    axs[1].specgram(filtered_wav, NFFT=1024, Fs=16000)
    axs[1].set_title('i2cat filter')
    axs[2].specgram(audacity_wav, NFFT=1024, Fs=16000)
    axs[2].set_title('audacity filter')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(original_spectrogram, interpolation='nearest', aspect='auto')
    axs[0].set_title('original')
    axs[1].imshow(filtered_spectrogram, interpolation='nearest', aspect='auto')
    axs[1].set_title('i2cat filter')
    axs[2].imshow(audacity_spectrogram, interpolation='nearest', aspect='auto')
    axs[2].set_title('audacity filter')

    plt.tight_layout()
    plt.show()


def convert_filter_to_audacity():
    filter_df = pd.read_excel(os.path.join('resources', 'filter_coefficients.xlsx'))
    filter_freq = filter_df.frequency.tolist()
    narrow_switch = filter_df.narrow_switch.tolist()
    wide_switch = filter_df.wide_switch.tolist()

    if filter_freq[0] == 0:
        filter_freq[0] = 1

    wide_switch_max = max(wide_switch)
    narrow_switch_max = max(narrow_switch)

    frequencies = [f'f{count}="{freq}"' for count, freq in enumerate(filter_freq)]
    wide_switch_values = [f'v{count}="{value - wide_switch_max}"' for count, value in enumerate(wide_switch)]
    narrow_switch_values = [f'v{count}="{value - narrow_switch_max}"' for count, value in enumerate(narrow_switch)]

    frequencies_string = ' '.join(frequencies)
    wide_switch_string = ' '.join(wide_switch_values)
    narrow_switch_string = ' '.join(narrow_switch_values)

    wide_switch_filter = ('FilterCurve:' +
                          frequencies_string +
                          ' FilterLength="8191" InterpolateLin="1" InterpolationMethod="B-spline" ' +
                          wide_switch_string)

    narrow_switch_filter = ('FilterCurve:' +
                            frequencies_string +
                            ' FilterLength="8191" InterpolateLin="1" InterpolationMethod="B-spline" ' +
                            narrow_switch_string)

    with open(os.path.join('resources', 'wide_switch_filter.txt'), 'w') as filter_file:
        filter_file.write(wide_switch_filter)

    with open(os.path.join('resources', 'narrow_switch_filter.txt'), 'w') as filter_file:
        filter_file.write(narrow_switch_filter)


if __name__ == '__main__':
    test_ei()
