import librosa
import numpy as np
from scipy.fftpack import dct

# If you want to see the spectrogram picture
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# def plot_spectrogram(spec, note,file_name):
#     """Draw the spectrogram picture
#         :param spec: a feature_dim by num_frames array(real)
#         :param note: title of the picture
#         :param file_name: name of the file
#     """ 
#     fig = plt.figure(figsize=(20, 5))
#     heatmap = plt.pcolor(spec)
#     fig.colorbar(mappable=heatmap)
#     plt.xlabel('Time(s)')
#     plt.ylabel(note)
#     plt.tight_layout()
#     plt.savefig(file_name)


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=16kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    feats=np.zeros((int(fft_len/2+1), num_filter))
    #compute boundary points of filterfbanks
    fmel_low_boundary = hz2mel(0)
    fmel_high_boundary = hz2mel(fs/2)
    mel_filter_points = np.linspace(fmel_low_boundary,fmel_high_boundary,num_filter+2)
    hz_filter_points = mel2hz(mel_filter_points) 
    #compute serial number of each filterfbanks: formula:f_nanlyse = m/N*fs
    serial_number_filter_points = np.floor((fft_len+1)*hz_filter_points/fs)
    #compute value of array feats
    for filter_number in range(0, num_filter):
        start_point_this_filter = int(serial_number_filter_points[filter_number])
        mid_point_this_filter = int(serial_number_filter_points[filter_number+1])
        end_point_this_filter = int(serial_number_filter_points[filter_number+2])
        for i in range(start_point_this_filter,mid_point_this_filter):
            feats[i,filter_number] = (i - start_point_this_filter) / (mid_point_this_filter - start_point_this_filter)
        for i in range(mid_point_this_filter, end_point_this_filter):
            feats[i,filter_number] = (end_point_this_filter - i) / (end_point_this_filter - mid_point_this_filter)
    #shape of spectrum: [num_frames,fft_len/2 + 1]
    #shape of feats:  [fft_len/2 + 1, num_filter]
    fbank = np.dot(spectrum, feats)
    log_fbank = np.log(fbank)
    return log_fbank

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    feats = np.zeros((fbank.shape[0],num_mfcc))
    feats = dct(fbank,axis=1)[:,:num_mfcc]
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()

def mel2hz(f_mel):
    return 700*(10**(f_mel/2595.0)-1)

def hz2mel(f_hz):
    return 2595*np.log10(1+f_hz/700.0)

def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    print(mfcc_feats)
    print(mfcc_feats.shape)
    # plot_spectrogram(fbank_feats, 'Filter Bank','fbank.png')
    #write_file(fbank_feats,'./test.fbank')
    # plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    #write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()
