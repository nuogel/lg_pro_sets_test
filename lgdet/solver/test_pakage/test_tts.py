"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
from lgdet.util.util_time_stamp import Time
from lgdet.util.util_audio.util_audio import Audio
from lgdet.util.util_audio.util_tacotron_audio import TacotronSTFT
from lgdet.solver.test_pakage._test_base import TestBase
import torch


class Test_TTS(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_TTS, self).__init__(cfg, args, train)
        self.audio = Audio(cfg)
        self.stft = TacotronSTFT(cfg.TRAIN)

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)
        timer = Time()
        for i in range(DataSet.__len__()):
            test_data = next(loader)
            timer.time_start()
            test_data = self.DataFun.to_devce(test_data)
            predicted = self.model(input_x=test_data[0], input_data=test_data, is_training=False)
            _, mels, _, _ = predicted
            for i, mel in enumerate(mels):
                # mel = mel.cpu()[..., :mel_lengths[i]]
                wav_txt = test_data[-1][i][-1]
                torch.save(mel, 'output/tested/' + wav_txt + '.wav.pt')
                waveform = self.stft.in_mel_to_wav(mel)
                # wav_path = 'output/tested/' + wav_txt + '.wav'
                # self.audio.write_wav(waveform, wav_path)
