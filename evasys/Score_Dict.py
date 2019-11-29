from evasys.Score_OBD_F1 import F1Score
from evasys.ASR_WER.wer import ASR_SCORE
from evasys.Score_SR import SR_SCORE


Score = {'OBD': F1Score,
         'ASR': ASR_SCORE,
         'SR': SR_SCORE

         }