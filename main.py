import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from FConvNet import FConv3Net
from dataset import RadarDataset


def process():
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load NN model
    network = FConv3Net().to(config['device'])
    checkpoint = torch.load(config['classic'], map_location=config['device'])
    network.load_state_dict(checkpoint['model_weights'])

    # Load data
    dataset = RadarDataset(config)

    for i in range(0, len(dataset)):
        spect, sb0_fft, amplitudes = dataset.__getitem__(i)

        in_spect = torch.tensor(np.expand_dims(spect, 0)).to(config['device']).float()
        fft_ml = network(in_spect)
        fft_ml = fft_ml.cpu().detach().numpy()[0]

        plt.figure('Beat signal spectrum')
        plt.plot(20 * np.log10(abs(sb0_fft[1, :])), 'b', label='Label')
        plt.plot(20 * np.log10(abs(fft_ml[1, :])), 'r', label='ML Prediction')
        plt.show()


if __name__ == '__main__':
    process()
