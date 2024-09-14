from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from lgdet.util import util_audio as audio


class Hparams:
    rescaling = False
    rescaling_max = 0.999


hparams = Hparams()


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'transcript/aishell_transcript_v0.8.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.strip().split(' ', 1)
            name = tmp[0]
            id = int(name[7:11])
            wav_path = os.path.join(in_dir, 'wav/train/' + name[6:11], name + '.wav')
            if id == 345:
                text = tmp[1].strip().replace(' ', '')
                futures.append(executor.submit(
                    partial(_process_utterance, out_dir, index, wav_path, text, id)))
                index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, id):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'lgspeech-spec-%05d.npy' % index
    mel_filename = 'lgspeech-mel-%05d.npy' % index
    # if os.path.isfile(os.path.join(out_dir, spectrogram_filename)) and os.path.isfile(os.path.join(out_dir, mel_filename)):
    #     print('isfile')
    # else:
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
