"""Input audio file and output pictures with mouth shape drawings
"""
import argparse
import logging
import os
import subprocess

from pathlib import Path

from python_speech_features import logfbank
from scipy.io import wavfile


LOGGER = logging.getLogger(__name__)

AUDIO_WINDOW_STEP_SIZE = .01  # in seconds
NFFT = 4800


def get_audio_features(path):
    path = Path(path)
    LOGGER.debug(f"normalizing audio signal from file {path}")
    (rate, sig) = normalize_audio(path)
    LOGGER.debug(f"samples / second = {rate}")
    LOGGER.debug(f"data points = {len(sig)}")
    LOGGER.debug(f"extracting features from audio signal using a window step "
                 f"size of {AUDIO_WINDOW_STEP_SIZE} seconds")
    audio = logfbank(sig, rate, winstep=AUDIO_WINDOW_STEP_SIZE, nfft=NFFT)
    LOGGER.debug(f"remaining signal points = {audio.shape[0]}")
    LOGGER.debug(f"number of features = {audio.shape[1]}")
    return audio


def normalize_audio(path):

    # create filename for normalized version of the audio
    path_normalized = path.parents[0].joinpath(
        '-'.join([path.stem, 'normalized.wav'])
    )

    LOGGER.debug(f"applying EBU R128 loudness normalization to audio {path} "
                 "using ffmpeg-normalize")
    try:
        # EBU R128 loudness normalization
        # save normalized audio to file and overwrite
        subprocess.check_output(
            f"ffmpeg-normalize -f {path} -o {path_normalized}",
            shell=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode())

    (rate, signal) = wavfile.read(path_normalized)

    os.remove(path_normalized)
    return rate, signal


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", help="path to audio wav file")
    args = parser.parse_args()
    get_audio_features(args.audio_file)
