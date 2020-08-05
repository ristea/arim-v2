import torch.utils.data
from scipy import signal
import numpy as np


class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        allData = np.load(self.config['arim-v2-test'], allow_pickle=True)

        self.sb_raw = allData[()]['sb']
        self.amplitudes = allData[()]['amplitudes']

        self.sb0_fft = np.fft.fft(allData[()]['sb0'], config['no_fft_points']) / config['no_points']

        self.sb0_fft_real = np.real(self.sb0_fft)
        self.sb0_fft_imag = np.imag(self.sb0_fft)

    def __getitem__(self, index):
        spectrogram = signal.stft(self.sb_raw[index], nfft=self.config['no_fft_points'], fs=self.config['fs'],
                                  nperseg=self.config['nperseg'], noverlap=self.config['noverlap'],
                                  window=self.config['window_type'], return_onesided=False, padded=False,
                                  boundary=None)[2]
        spectrogram = spectrogram / 40.0

        spec_real = np.expand_dims(np.real(spectrogram), 0)
        spec_imag = np.expand_dims(np.imag(spectrogram), 0)
        spec_abs = np.expand_dims(np.abs(spectrogram), 0)
        spect = np.concatenate((spec_real, spec_abs, spec_imag), 0)

        sb0 = np.concatenate((np.expand_dims(self.sb0_fft_real[index], 0),
                              np.expand_dims(np.abs(self.sb0_fft[index]), 0),
                              np.expand_dims(self.sb0_fft_imag[index], 0)), 0)

        # Positions
        position_real = np.expand_dims(np.real(self.amplitudes[index]), 0)
        position_imag = np.expand_dims(np.imag(self.amplitudes[index]), 0)
        position_abs = np.expand_dims(np.abs(self.amplitudes[index]), 0)
        position = np.concatenate((position_real, position_abs, position_imag), 0)

        return [spect, sb0, position]

    def __len__(self):
        return len(self.sb_raw)

    def __repr__(self):
        return self.__class__.__name__
