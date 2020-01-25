
import argparse
import os
import numpy as np
import scipy
import scipy.io.wavfile
from matplotlib import pyplot as plt


def main(args):
    
    wav_files              = find_wav_fpaths(args.dir_path)
    wav_files              = append_labels(wav_files)
    test_set, training_set = split_train_test(wav_files)

    ## training stage
    params                 = read_wavfile_wtih_scipy(training_set)
    
    plt.plot(params)
    plt.show()


def find_wav_fpaths(dir_path):
    """
    returns a list of paths for wav files
    """

    results = []

    for root, subdir, fnames in os.walk(dir_path):

        for fname in fnames:

            if os.path.splitext(fname)[-1] == '.wav':
                results.append(os.path.join(root, fname))
        
    return results


def append_labels(wav_files):

    results = []
    
    for wav_file in wav_files:

        if 'voiced' in wav_file:
            results.append((wav_file, 1))
        elif 'silence' in wav_file:
            results.append((wav_file, 0))
        else:
            print("Couldn't correctly split data to voiced-silence", file=stderr)
            sys.exit(1)
    
    return results


def split_train_test(wav_files):

    # shuffle silence and voiced wavs
    arr = np.asarray(wav_files)
    np.random.shuffle(arr)
    wav_files = arr.tolist()

    # split
    test_size = int(0.2 * len(wav_files))
    
    test  = wav_files[:test_size]
    train = wav_files[test_size:]
    
    return test, train


def read_wavfile_wtih_scipy(training_set):

    results = []

    for wav_file in training_set:

        # TO-DO set variable name properly

        input_data = scipy.io.wavfile.read(wav_file[0])[1]
        results.append(np.mean(input_data))

    return results


def parser():

    parser = argparse.ArgumentParser(description="Separating silent waves from louder waves")
    parser.add_argument('-p', '--dir_path', type=str, required=True, help="give path to the file with waves.")
    return parser.parse_args()


if __name__ == '__main__':

    args = parser()
    main(args)