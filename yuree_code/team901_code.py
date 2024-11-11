import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from scipy.fftpack import fft, fftfreq
from scipy.stats import skew, kurtosis
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import math

# --------- 데이터 저장 및 전처리 ---------
def load_mat_files(data_dir):
    all_signals = []
    all_fs = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mat'):
            file_path = os.path.join(data_dir, file_name)
            mat_data = loadmat(file_path)
            signal_data = mat_data['signal'].flatten()
            fs = mat_data['fs'][0][0]

            all_signals.append(signal_data)
            all_fs.append(fs)

    return all_signals, all_fs

def create_fault_dataframe(fault_type, data_dir):
    all_signals, all_fs = load_mat_files(data_dir)
    df_signals = pd.DataFrame(all_signals)
    df_signals['fs'] = all_fs
    df_signals['fault_type'] = fault_type
    return df_signals

def concatenate_fault_data(fault_types, base_dir):
    data_frames = []
    for fault in fault_types:
        data_dir = os.path.join(base_dir, fault)
        df_signals = create_fault_dataframe(fault, data_dir)
        data_frames.append(df_signals)
    final_df = pd.concat(data_frames, ignore_index=True)
    return final_df

def label_fault_type(df):
    label_mapping = {
        '내륜결함': 1,
        '롤러결함': 2,
        '외륜결함': 3,
        '정상': 0
    }
    df['fault_type'] = df['fault_type'].replace(label_mapping)
    return df.drop(columns=['fs'])

# --------- 주파수 특성 함수 ---------
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_top_frequencies(data, fs, lowcut=10, highcut=5000, n_top_frequencies=10):
    df_filtered = data.iloc[:, :-1]
    rows = []

    for i in range(df_filtered.shape[0]):
        signal = df_filtered.iloc[i, :].values
        signal = signal - np.mean(signal)
        filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)

        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        envelope = envelope - np.mean(envelope)

        fft_envelope = fft(envelope)
        freqs = fftfreq(len(envelope), d=1/fs)
        
        half_n = len(freqs) // 2
        freqs = freqs[:half_n]
        amplitude_spectrum = np.abs(fft_envelope[:half_n])

        top_indices = np.argsort(amplitude_spectrum)[-n_top_frequencies:][::-1]
        top_frequencies = freqs[top_indices]

        row_data = {}
        for j in range(n_top_frequencies):
            row_data[f'Frequency_{j+1}'] = top_frequencies[j]
        rows.append(row_data)

    result_df = pd.DataFrame(rows)
    return result_df

def compute_frequency_features(data, fs):
    df_filtered = data.iloc[:, :-1]
    features = []

    for i in range(df_filtered.shape[0]):
        signal = df_filtered.iloc[i, :].values
        fft_signal = fft(signal)
        freqs = fftfreq(len(signal), d=1/fs)

        half_n = len(freqs) // 2
        freqs = freqs[:half_n]
        amplitude_spectrum = np.abs(fft_signal[:half_n])

        mean_frequency = np.sum(freqs * amplitude_spectrum) / np.sum(amplitude_spectrum)
        max_amplitude_frequency = freqs[np.argmax(amplitude_spectrum)]
        spectral_std_dev = np.sqrt(np.sum((freqs - mean_frequency) ** 2 * amplitude_spectrum) / np.sum(amplitude_spectrum))
        spectral_energy = np.sum(amplitude_spectrum ** 2)
        spectral_skewness = skew(amplitude_spectrum)

        feature_data = {
            'Mean Frequency': mean_frequency,
            'Max Amplitude Frequency': max_amplitude_frequency,
            'Spectral Std Dev': spectral_std_dev,
            'Spectral Energy': spectral_energy,
            'Spectral Skewness': spectral_skewness
        }
        features.append(feature_data)

    features_df = pd.DataFrame(features)
    return features_df

# --------- 시간 특성 함수 ---------
def compute_signal_features(data):
    stt_features_df = pd.DataFrame(index=data.index)
    data_filtered = data.iloc[:, :-1]
    
    stt_features_df['mean'] = data_filtered.mean(axis=1)
    stt_features_df['absolute_mean'] = data_filtered.abs().mean(axis=1)
    stt_features_df['absolute_max'] = data_filtered.abs().max(axis=1)
    stt_features_df['std'] = data_filtered.std(axis=1)
    stt_features_df['variance'] = data_filtered.var(axis=1)
    stt_features_df['peak_to_peak'] = data_filtered.max(axis=1) - data_filtered.min(axis=1)
    stt_features_df['peak'] = data_filtered.max(axis=1)
    stt_features_df['skewness'] = data_filtered.apply(lambda row: skew(row), axis=1)
    stt_features_df['kurtosis'] = data_filtered.apply(lambda row: kurtosis(row), axis=1)
    stt_features_df['rms'] = np.sqrt((data_filtered ** 2).mean(axis=1))
    
    stt_features_df['crest_factor'] = stt_features_df['peak'] / stt_features_df['rms']
    stt_features_df['impulse_factor'] = stt_features_df['absolute_max'] / stt_features_df['absolute_mean']
    stt_features_df['margin'] = stt_features_df['peak'] - stt_features_df['mean']
    stt_features_df['power'] = stt_features_df['rms'] ** 2
    stt_features_df['shape_factor'] = stt_features_df['rms'] / stt_features_df['absolute_mean']
    stt_features_df['clearance_factor'] = stt_features_df['absolute_max'] / (stt_features_df['rms'] ** 2)

    return stt_features_df

# --------- BPFO, BPFI 계산 함수 ---------
def calculate_bpfo_bpfi(rpm, ball_count, contact_angle, ball_diameter, pitch_diameter):
    rotation_frequency = rpm / 60.0
    contact_angle_rad = math.radians(contact_angle)
    bpfo = (rotation_frequency * ball_count) / 2 * (1 - (ball_diameter / pitch_diameter) * math.cos(contact_angle_rad))
    bpfi = (rotation_frequency * ball_count) / 2 * (1 + (ball_diameter / pitch_diameter) * math.cos(contact_angle_rad))
    
    print(f"Calculated BPFO: {bpfo:.2f} Hz, BPFI: {bpfi:.2f} Hz")
    
    return bpfo, bpfi

# --------- 주파수에 따른 진폭 함수 ---------
def calculate_bpfo_bpfi_magnitude(data_group, fs, rpm, ball_count, contact_angle, ball_diameter, pitch_diameter, bandwidth=30):
    # BPFO와 BPFI 주파수 계산
    bpfo_freq, bpfi_freq = calculate_bpfo_bpfi(rpm, ball_count, contact_angle, ball_diameter, pitch_diameter)
    
    fft_magnitudes = []
    frequencies = np.fft.fftfreq(data_group.shape[1] - 1, d=1/fs)
    
    for index, signal in data_group.iloc[:, :-1].iterrows():
        fft_values = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_values)[:len(fft_values)//2]
        pos_frequencies = frequencies[:len(frequencies)//2]

        group_magnitudes = {}
        
        for freq, label in zip([bpfo_freq, bpfi_freq], ["BPFO_data", "BPFI_data"]):
            low_bound = max(0, freq - bandwidth)
            high_bound = freq + bandwidth
            freq_range = (pos_frequencies >= low_bound) & (pos_frequencies <= high_bound)
            magnitude_in_range = np.sum(fft_magnitude[freq_range])
            
            
            group_magnitudes[label] = magnitude_in_range
        
        fft_magnitudes.append(group_magnitudes)
    
    result_df = pd.DataFrame(fft_magnitudes)
    result_df['Hz'] = fs
    result_df['rpm'] = rpm
    return result_df

# --------- 모델 평가 함수 ---------
def evaluate_model_with_scaling(test_dataframe, model_path='team901_model.pkl'):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
        expected_columns = data['columns']
    
    X_test = test_dataframe.drop(columns=['fault_type'])
    y_test = test_dataframe['fault_type']
    X_test = X_test[expected_columns]

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
        
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report_df
    }


# --------- 분석 실행 ---------
if __name__ == "__main__":
    fault_types = ['내륜결함', '롤러결함', '외륜결함', '정상']
    base_dir = r'C:\Users\medici\901\학습용 데이터1'  # 데이터 파일 폴더의 경로를 적어주세요.
    
    df = concatenate_fault_data(fault_types, base_dir)
    df = label_fault_type(df)
    df.to_csv(r'.\vibrate_data.csv', index=False, encoding='utf-8-sig') # 경로 수정이 필요하시면 수정해주세요.

    fs = 25600
    rpm = 1800
    ball_count = 12
    contact_angle = 0
    ball_diameter = 7.5
    pitch_diameter = 34

    top_frequencies_df = extract_top_frequencies(data=df, fs=fs)
    frequency_features_df = compute_frequency_features(data=df, fs=fs)
    signal_features_df = compute_signal_features(data=df.iloc[:, :-1])

    fault_magnitude = calculate_bpfo_bpfi_magnitude(df, fs, rpm, ball_count, contact_angle, ball_diameter, pitch_diameter)
    print("Fault Magnitude:", fault_magnitude)

    final_df = pd.concat([top_frequencies_df, frequency_features_df, signal_features_df, fault_magnitude], axis=1)
    final_df['fault_type'] = df['fault_type'].values
    final_df.to_csv(r'.\final_data.csv', index=False, encoding='utf-8-sig')
    # 경로 수정이 필요하시면 수정해주세요.
    
    evaluation_results = evaluate_model_with_scaling(final_df)
    print("Evaluation Results:", evaluation_results)

