import socket
import struct
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
from pymongo import MongoClient
from os.path import exists
from scipy.io.wavfile import write
import pyaudio

# uri = "mongodb+srv://Local_Trial_Client:dvTxiBTJZAf1ZDcH@clusterlearning.uebi9hb.mongodb.net/?retryWrites=true&w=majority"
# client = MongoClient(uri)
# db = client.get_database("Trial_1")
# rec = db.BPM

# uri = "mongodb+srv://abel:65r7PNLqDF8gHUQ0@testing.iylygdv.mongodb.net/?retryWrites=true&w=majority"
# client = MongoClient(uri)
# dp = client.get_database("One")
# rec = dp.Beats

PatientID = 4817

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=8000, output=True, frames_per_buffer=4096)

peakY = []
peakX = []

def butter_lowpass_filter(sig, cutoff, order, Fs):
    # Get the filter coefficients 
    b, a = butter(order, Wn = cutoff, fs = Fs, btype='low', analog=False)
    y = filtfilt(b, a, sig)
    return y

def butter_highpass_filter(sig, cutoff, order, Fs):
    # Get the filter coefficients 
    b, a = butter(order, Wn = cutoff, fs = Fs, btype='high', analog=False)
    y = filtfilt(b, a, sig)
    return y

UDP_IP =  "192.168.157.251"
# "172.20.10.2"    192.168.249.251
UDP_PORT = 11211

sock = socket.socket(socket.AF_INET, # Internet
                      socket.SOCK_DGRAM) # UDP

sock.bind((UDP_IP, UDP_PORT))

startRecording = input("Start Recording (y/n)?")

if(startRecording == 'y'):

#----------------DATA ACQUISITION---------------
    startRecording = 'n'
    while(startRecording == 'n'):
        print("This is the audio that the mic is hearing:")

        for i in range(20):
            outData = []
            emg_data_list = []  # Initialize a list to store EMG data
            txData = bytes()

            data, addr = sock.recvfrom(131072) # buffer size is 4096 bytes
            # 131072
            y = []
            i = 0  # Initialize index outside the loop
            while i < len(data):
                a = float(struct.unpack('<f', data[i: i+4])[0])
                i += 4
                if a == -1:  # Check for the -1 indicator
                    emg_data = float(struct.unpack('<f', data[i: i+4])[0])
                    i += 4
                    emg_data_list.append(emg_data)
                    print(emg_data)  # Print the EMG data to the terminal
                else:
                    y.append(a)

            for i in y[:2048]:
                outData.append(i)

            for i in outData:
                txData += bytes(struct.pack('<f', i))

            stream.write(txData)
        startRecording = input("Start Recording (y/n)?")

    outData = []
    emg_data_list = []  # Initialize a list to store EMG data
    print("Recording in progress")

    for i in range(118):

        print(str(i) + " out of 118")

        data, addr = sock.recvfrom(131072) # buffer size is 4096 bytes

        y_temp = []
        i = 0  # Initialize index outside the loop
        while i < len(data):
            a = float(struct.unpack('<f', data[i : i+4])[0])
            i += 4
            if a == -1:  # Check for the -1 indicator
                emg_data = float(struct.unpack('<f', data[i: i+4])[0])
                i += 4
                emg_data_list.append(emg_data)
                print(f"EMG Data: {emg_data}")  # Print the EMG data to the terminal
            else:
                y_temp.append(a)

        for i in y_temp[:2048]:
            outData.append(i)
            
#----------------DATA PROCESSING--------------

    y = outData # Should be recorded data
    x = [i for i in range(len(y))]
    N = len(y)
    fs = 8000.0
    T = 1.0/8000.0 # Time period of recording

    yf_prev = fft(y)
    xf_prev = fftfreq(N, T)[:N//2]

    lpf_y = butter_lowpass_filter(y, 5, 7, 8000)
    bpf_y = butter_highpass_filter(lpf_y, 1.5, 7, 8000)

    bpf_y = [10*i for i in bpf_y]
    bpf_y = np.array(bpf_y)

    yf = fft(bpf_y)
    xf = fftfreq(N, T)[:N//2]

    minY = min(bpf_y)
    posY = [(i + (minY * -1) + 1.0) for i in bpf_y]

    log_y = [math.log(i) for i in posY]

    minY = min(log_y)
    negY = [(i - minY) for i in bpf_y]
    # negY = [(i - minY) for i in log_y]

    lpf_log_y = butter_lowpass_filter(negY, 50, 7, 8000)
    # lpf_log_y = butter_lowpass_filter(log_y, 50, 7, 8000)

    e_lpf_log_y = [math.exp(i) for i in lpf_log_y]

    yf_after = fft(e_lpf_log_y)
    xf_after = fftfreq(N, T)[:N//2]

    meanY = sum(e_lpf_log_y)/len(e_lpf_log_y)
    medianY = sorted(e_lpf_log_y)[(len(e_lpf_log_y) - 1)//2]
    normalized_data = [(i - medianY)/meanY for i in e_lpf_log_y]

    yf_norm = fft(normalized_data)
    xf_norm = fftfreq(N, T)[:N//2]

    Gaussian_signal = gaussian_filter1d(normalized_data, 500)

    meanY = sum(Gaussian_signal)/len(Gaussian_signal)
    medianY = sorted(Gaussian_signal)[(len(Gaussian_signal) - 1)//2]
    Gaussian_signal = [(i - medianY)/meanY for i in Gaussian_signal]

    yf_gauss = fft(Gaussian_signal)
    xf_gauss = fftfreq(N, T)[:N//2]

    # threshold_height = sorted(Gaussian_signal)[math.ceil((len(Gaussian_signal)-1)/2)]
    threshold_height = sum(Gaussian_signal)/len(Gaussian_signal)
    peaks, _ = find_peaks(Gaussian_signal)

    peakPltY = []
    peakPltX = []
    for i in range(len(peaks)):
        peakPltY.append(Gaussian_signal[peaks[i]])
        peakPltX.append(x[peaks[i]])

    BPM = (len(peaks)/(2.0*len(Gaussian_signal)/8000.0))*60.0

    print("BPM: ", BPM)
    print("len(peaks): ", len(peaks))

    print("Sending the BPM data to the cloud...")

    # if(rec.find_one({"ID":PatientID}) == None):
    #     print("Record doesn't exist, creating a record")
    #     rec.insert_one({
    #         "ID":PatientID,
    #         "BPM":0
    #     })
    # else:
    #     print("Record found, Updating")

    # rec.update_one({"ID":PatientID}, {"$set": {"BPM" : BPM}})

    #----------------SAVE THE INITIAL CSV-----------------

    file_name = "audioData_BackendGG_"
    counter = 0
    while(exists(file_name + str(counter) + ".csv")):
        counter = counter + 1
    file = open (file_name + str(counter) + ".csv", 'w')
    for i in outData:
        file.write(str(i) + "\n")
    file.close()

    #------------------SAVE THE AUDIO----------------------

    file_name = "Backend_AudioGG_"
    counter = 0
    while(exists(file_name + str(counter) + ".wav")):
        counter = counter + 1
    write(file_name + str(counter)+ ".wav", 8000, bpf_y.astype(np.float32))

    #----------------PLOT AND SAVE THE RESULTS-----------------

    fig = plt.figure()
    fig.suptitle("Peaks, Heartrate = %f BPM" % BPM, fontsize=15)
    plt.plot(x, Gaussian_signal)
    plt.plot(peakPltX, peakPltY,'x')
    plt.plot([threshold_height for i in range(len(x))], linestyle = '--')

    fig.savefig("Peaks_Heartrate_" + str(counter) + ".png")  # Save figure with counter

    rows = 4
    columns = 1

    fig1 = plt.figure()
    fig1.suptitle("Data Visualization", fontsize=15)

    ax1 = fig1.add_subplot(rows, columns, 1)
    ax1.plot(x, y, label='Original Signal')
    ax1.legend()

    ax2 = fig1.add_subplot(rows, columns, 2)
    ax2.plot(x, bpf_y, label='Bandpass Filtered Signal')
    ax2.legend()

    ax3 = fig1.add_subplot(rows, columns, 3)
    ax3.plot(x, Gaussian_signal, label='Gaussian Filtered Signal')
    ax3.plot(peakPltX, peakPltY,'x')
    ax3.plot([threshold_height for i in range(len(x))], linestyle = '--')
    ax3.legend()

    ax4 = fig1.add_subplot(rows, columns, 4)
    ax4.plot(emg_data_list, label='EMG Data')
    ax4.legend()

    fig1.savefig("Data_Visualization_" + str(counter) + ".png")  # Save figure with counter

    plt.show()

