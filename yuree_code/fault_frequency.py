import math
import numpy as np
import pandas as pd
import os

# BPFO 및 BPFI를 계산하는 함수
def calculate_bpfo_bpfi(rpm, ball_count, contact_angle, ball_diameter, pitch_diameter):
    """
    RPM 값을 사용하여 BPFO 및 BPFI를 계산하는 함수

    Args:
        rpm (float): 회전수 (단위: RPM)
        ball_count (int): 볼의 개수
        contact_angle (float): 접촉각 (단위: 도, degree)
        ball_diameter (float): 볼의 직경 (단위: mm)
        pitch_diameter (float): 피치 직경 (단위: mm)
        
    Returns:
        tuple: (BPFO, BPFI) 계산된 결함 주파수들
    """
    # 회전 주파수 계산 (RPM에서 Hz로 변환)
    rotation_frequency = rpm / 60.0

    # 접촉각을 라디안으로 변환
    contact_angle_rad = math.radians(contact_angle)
    
    # BPFO 계산
    bpfo = (rotation_frequency * ball_count) / 2 * (1 + (ball_diameter / pitch_diameter) * math.cos(contact_angle_rad))
    
    # BPFI 계산
    bpfi = (rotation_frequency * ball_count) / 2 * (1 - (ball_diameter / pitch_diameter) * math.cos(contact_angle_rad))
    
    return bpfo, bpfi

# 주파수와 진폭의 세기를 계산하는 함수
def calculate_bpfo_bpfi_magnitude(data_group, sampling_rate, rpm, ball_count, contact_angle, ball_diameter, pitch_diameter, bandwidth=30):
    """
    BPFO와 BPFI 주파수 대역의 진폭 합을 계산하는 함수

    Args:
        data_group (pd.DataFrame): 신호 데이터
        sampling_rate (float): 샘플링 주파수 (단위: Hz)
        rpm (float): 회전수 (단위: RPM)
        ball_count (int): 볼의 개수
        contact_angle (float): 접촉각 (단위: 도, degree)
        ball_diameter (float): 볼의 직경 (단위: mm)
        pitch_diameter (float): 피치 직경 (단위: mm)
        bandwidth (float): 주파수 대역폭 (기본값: 30Hz)

    Returns:
        pd.DataFrame: BPFO 및 BPFI 주파수 대역의 진폭 크기를 담은 데이터프레임
    """
    # BPFO와 BPFI 주파수 계산
    bpfo_freq, bpfi_freq = calculate_bpfo_bpfi(rpm, ball_count, contact_angle, ball_diameter, pitch_diameter)
    
    fft_magnitudes = []
    frequencies = np.fft.fftfreq(data_group.shape[1], d=1/sampling_rate)
    
    for _, signal in data_group.iterrows():
        # FFT 변환
        fft_values = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_values)[:len(fft_values)//2]  # 양수 주파수만 선택
        pos_frequencies = frequencies[:len(frequencies)//2]  # 양수 주파수

        # 주파수 범위 및 진폭 합산
        group_magnitudes = {}
        
        for freq, label in zip([bpfo_freq, bpfi_freq], ["BPFO", "BPFI"]):
            # 주파수 대역 설정
            low_bound = max(0, freq - bandwidth)  # 음수가 나오지 않도록 처리
            high_bound = freq + bandwidth

            # 대역 내 주파수 범위 설정
            freq_range = (pos_frequencies >= low_bound) & (pos_frequencies <= high_bound)
            # 대역 내 진폭 합산
            magnitude_in_range = np.sum(fft_magnitude[freq_range])
            group_magnitudes[label] = magnitude_in_range
        
        fft_magnitudes.append(group_magnitudes)
    
    # DataFrame으로 변환하고 Hz 열 추가
    result_df = pd.DataFrame(fft_magnitudes)
    result_df['Hz'] = sampling_rate  # Hz 열 추가 및 값 할당
    result_df['rpm'] = rpm
    return result_df

# 샘플링 속도와 베어링 기기 정보를 입력해주세요.
sampling_rate = 25600      # 예시 샘플링 주파수
rpm = 1800                 # 회전수 (단위: RPM)
ball_count = 13             # 볼 개수
contact_angle = 0         # 접촉각 (단위: 도)
ball_diameter = 9         # 볼 직경 (단위: mm)
pitch_diameter = 46.5        # 피치 직경 (단위: mm)


# 예시 데이터프레임 inner를 사용해 계산
fault_magnitude = calculate_bpfo_bpfi_magnitude(final_df, sampling_rate, rpm, ball_count, contact_angle, ball_diameter, pitch_diameter)


