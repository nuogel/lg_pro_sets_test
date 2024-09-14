sample_rate = 16000  # Sample rate.
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
allow_clipping_in_normalization = True
hop_length = int(sample_rate * frame_shift)  # samples.
win_length = int(sample_rate * frame_length)  # samples.
n_fft = win_length  # fft points (samples)
n_mels = 80  # Number of Mel banks to generate
power = 1.5  # Exponent for amplifying the predicted magnitude
n_iter = 50  # Number of inversion iterations
preemphasis = .97  # or None
max_db = 100
min_db = -100
ref_db = 20
top_db = 15

fmin = 50  # Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
fmax = 7600

trim_fft_size = 512
trim_hop_size = 128
trim_top_db = 50
