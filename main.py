import logging
import threading
import time
import numpy as np
import os
import random
import pyaudio as pa
import pyrubberband as pyrb
import scipy.io.wavfile
import scipy.signal
import sys

from collections import deque
from multiprocessing import Value, Process


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


CHUNK = 1000
RATE = 16000


def audio_filter(mode_state, mode_param):
    audio_logger = logger.getChild('audio')

    p = pa.PyAudio()

    input_queue = deque(maxlen=CHUNK)
    output_queue = deque(maxlen=CHUNK)

    def get_audio_index_by_name(name, input_device=False, output_device=False):
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if input_device and device_info['maxInputChannels'] == 0:
                continue
            if output_device and device_info['maxOutputChannels'] == 0:
                continue
            if device_info['name'] == name:
                audio_logger.info(f'device found = {device_info}')
                return i
        return None

    beep_freq, beep_signal = scipy.io.wavfile.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beep.wav'))
    assert beep_freq == RATE and len(beep_signal.shape) == 1
    beep_queue = None

    def play_beep(signal):
        nonlocal beep_queue
        if beep_queue is None:
            beep_queue = beep_signal

        if len(signal) < len(beep_queue):
            signal += beep_queue[:len(signal)] * mode_param.value
            beep_queue = beep_queue[len(signal):]
        else:
            signal[:len(beep_queue)] += beep_queue * mode_param.value
            beep_queue, mode_state.value = None, 0

        return signal

    def convert_on_state(signal):
        if mode_state.value == 0:
            audio_logger.debug('normal')
            return signal
        elif mode_state.value == 1:
            audio_logger.debug(f'pitch: {mode_param.value}')
            return pyrb.pitch_shift(signal / 32768.0, RATE, mode_param.value) * 32768.0
        elif mode_state.value == 2:
            audio_logger.debug(f'volume: {mode_param.value}')
            return signal * mode_param.value
        elif mode_state.value == 3:
            audio_logger.debug(f'beep: {mode_param.value}')
            return play_beep(signal)
        else:
            raise NotImplementedError

    def recorder():
        index = get_audio_index_by_name('Soundflower (2ch)', input_device=True)

        stream = p.open(format=pa.paInt16, channels=1, rate=RATE,
                        frames_per_buffer=CHUNK, input_device_index=index,
                        input=True, output=False)

        while stream.is_active():
            data = stream.read(CHUNK)
            input_queue.extend(np.frombuffer(data, dtype=np.int16))

    def converter():
        prev_input = np.zeros(CHUNK, dtype=np.int16)
        prev_output = np.zeros(CHUNK, dtype=np.int16)
        window = scipy.signal.get_window('hann', CHUNK * 2, fftbins=True)

        while True:
            if not len(input_queue):
                time.sleep(0.01)
                continue

            data = np.array(input_queue, dtype='int16').astype(np.float64)
            input_queue.clear()

            input_signal = np.concatenate((prev_input, data))
            prev_input = data

            output_signal = convert_on_state(input_signal * window).astype(np.int16)
            output_queue.extend(prev_output + output_signal[:CHUNK])
            prev_output = output_signal[CHUNK:]

    def player():
        stream = p.open(format=pa.paInt16, channels=1, rate=RATE, input=False, output=True)

        while True:
            if not len(output_queue):
                time.sleep(0.01)
                continue

            data = np.array(output_queue, dtype=np.int16)
            output_queue.clear()
            stream.write(data.tobytes())


    th_record = threading.Thread(target=recorder)
    th_convert = threading.Thread(target=converter)
    th_play = threading.Thread(target=player)

    th_record.start()
    th_convert.start()
    th_play.start()


def local_switch(mode_state, mode_param):
    while True:
        mode_state.value = random.randint(0, 3)
        if mode_state.value:
            mode_param.value = random.random() + 1.5
        else:
            mode_param.value = 0

        time.sleep(5)


def main():
    mode_state = Value('b', 0)
    mode_param = Value('d', 0)

    audio_process = Process(target=audio_filter, args=[mode_state, mode_param])
    audio_process.start()

    local_switch_process = Process(target=local_switch, args=[mode_state, mode_param])
    local_switch_process.start()

    audio_process.join()
    local_switch_process.join()


if __name__ == '__main__':
    main()
