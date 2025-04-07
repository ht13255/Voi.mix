import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import tempfile
import os
import shutil
from spleeter.separator import Separator
from scipy.signal import fftconvolve, butter, lfilter

st.title("프로덕션급 AI Advanced Remix Studio")
st.write("두 곡을 업로드하면 AI가 자동으로 분석하여 다양한 요소(보컬/반주 분리, 템포·키 보정, 리버브, 필터, 딜레이, 사이드체인, 코러스, 패닝, 컴프레션 등)를 적용한 리믹스를 생성합니다!")

# ── 파일 업로드 ───────────────────────────────────────────
song1 = st.file_uploader("첫 번째 곡 업로드", type=["mp3", "wav", "ogg", "flac"])
song2 = st.file_uploader("두 번째 곡 업로드", type=["mp3", "wav", "ogg", "flac"])

# ── 사이드바: 기본 믹스 설정 ───────────────────────────────
st.sidebar.header("믹스 설정")
vocal_volume = st.sidebar.slider("보컬 볼륨", 0.0, 2.0, 1.0, 0.1)
inst_volume  = st.sidebar.slider("반주 볼륨", 0.0, 2.0, 1.0, 0.1)
master_volume = st.sidebar.slider("마스터 볼륨", 0.0, 2.0, 1.0, 0.1)

compression_threshold = st.sidebar.slider("컴프레션 임계치", 0.5, 1.0, 0.8, 0.05)
compression_ratio = st.sidebar.slider("컴프레션 비율", 1.0, 10.0, 4.0, 0.5)
tempo_tolerance = st.sidebar.slider("템포 허용 오차 (BPM)", 0, 10, 5, 1)
apply_key_correction = st.sidebar.checkbox("키 보정 적용", value=True)

# ── 사이드바: 리버브 / 필터 ──────────────────────────────────
st.sidebar.subheader("리버브 설정")
apply_reverb = st.sidebar.checkbox("리버브 적용", value=True)
reverb_decay = st.sidebar.slider("리버브 감쇠 시간 (초)", 0.5, 5.0, 2.0, 0.1)
reverb_mix = st.sidebar.slider("리버브 믹스 비율", 0.0, 1.0, 0.3, 0.05)

st.sidebar.subheader("고역 필터 설정")
apply_hp_filter = st.sidebar.checkbox("고역 필터 적용", value=True)
hp_cutoff = st.sidebar.slider("고역 필터 컷오프 (Hz)", 20, 2000, 200, 10)

st.sidebar.subheader("로우패스 필터 설정")
apply_lp_filter = st.sidebar.checkbox("로우패스 필터 적용", value=False)
lp_cutoff = st.sidebar.slider("로우패스 컷오프 (Hz)", 1000, 10000, 8000, 100)

# ── 사이드바: 딜레이 / 코러스 / 사이드체인 ─────────────────────
st.sidebar.subheader("딜레이 효과")
apply_delay = st.sidebar.checkbox("딜레이 적용", value=False)
delay_time = st.sidebar.slider("딜레이 시간 (초)", 0.1, 1.0, 0.3, 0.05)
delay_feedback = st.sidebar.slider("딜레이 피드백", 0.0, 0.9, 0.4, 0.05)
delay_mix = st.sidebar.slider("딜레이 믹스 비율", 0.0, 1.0, 0.3, 0.05)

st.sidebar.subheader("사이드체인(덕킹) 효과")
apply_sidechain = st.sidebar.checkbox("사이드체인 적용", value=False)
sidechain_strength = st.sidebar.slider("사이드체인 강도", 0.0, 1.0, 0.5, 0.05)

st.sidebar.subheader("코러스 효과")
apply_chorus_effect = st.sidebar.checkbox("코러스 적용", value=False)
chorus_depth = st.sidebar.slider("코러스 깊이 (초)", 0.001, 0.01, 0.003, 0.001)
chorus_rate = st.sidebar.slider("코러스 속도 (Hz)", 0.5, 3.0, 1.5, 0.1)
chorus_mix = st.sidebar.slider("코러스 믹스 비율", 0.0, 1.0, 0.3, 0.05)

# ── 사이드바: 자동 설정 적용 ────────────────────────────────
st.sidebar.subheader("자동 설정")
auto_settings = st.sidebar.checkbox("자동 설정 적용", value=False)

remix_mode = st.radio("리믹스 모드 선택", ["기본 오버레이 믹스", "AI 자동 고도화 리믹스"])

# ── 안전한 임시 디렉토리 삭제 함수 ─────────────────────────────
def safe_rmtree(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        st.error(f"임시 디렉토리 삭제 오류: {e}")

# ── 헬퍼 함수 ───────────────────────────────────────────────
def separate_stems(file_bytes):
    """
    Spleeter를 사용해 오디오 파일을 보컬과 반주로 분리합니다.
    분리된 결과와 임시 출력 디렉토리 경로를 반환합니다.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(file_bytes)
            temp_audio_file_path = temp_audio_file.name

        temp_output_dir = tempfile.mkdtemp()
        separator = Separator('spleeter:2stems', multiprocess=False)
        separator.separate_to_file(temp_audio_file_path, temp_output_dir)

        base_name = os.path.splitext(os.path.basename(temp_audio_file_path))[0]
        vocals_path = os.path.join(temp_output_dir, base_name, "vocals.wav")
        accomp_path = os.path.join(temp_output_dir, base_name, "accompaniment.wav")

        vocals, sr_vocals = librosa.load(vocals_path, sr=None)
        accomp, sr_accomp = librosa.load(accomp_path, sr=None)

        os.remove(temp_audio_file_path)  # 임시 파일 삭제
        return vocals, accomp, sr_vocals, temp_output_dir

    except Exception as e:
        st.error(f"Spleeter 분리 오류: {e}")
        raise

def detect_key(audio, sr):
    """
    librosa의 크로마 스펙트럼을 활용해 단순 키 추정을 수행합니다.
    반환값: (key_index, key_name) – key_index: 0~11, key_name: 'C', 'C#', ...
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_index = np.argmax(chroma_mean)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return key_index, notes[key_index]

def semitone_difference(vocal_key, inst_key):
    """
    두 키 간 반음 차이를 계산 (범위: -6 ~ +6)
    """
    diff = vocal_key - inst_key
    diff = (diff + 6) % 12 - 6
    return diff

def compress_audio(audio, threshold=0.8, ratio=4.0):
    """
    간단한 다이나믹 컴프레션 적용.
    amplitude가 threshold 초과 시 ratio에 따라 압축.
    """
    abs_audio = np.abs(audio)
    over_threshold = abs_audio > threshold
    compressed = audio.copy()
    compressed[over_threshold] = np.sign(audio[over_threshold]) * (threshold + (abs_audio[over_threshold] - threshold) / ratio)
    return compressed

def match_tempo(audio, sr, target_tempo, current_tempo):
    """
    현재 템포(current_tempo)를 target_tempo로 맞추기 위한 타임 스트레칭.
    """
    rate = target_tempo / current_tempo
    return librosa.effects.time_stretch(audio, rate)

def apply_reverb(audio, sr, decay=2.0, mix=0.3):
    """
    간단한 리버브 효과: 지수 감쇠 임펄스 응답을 생성 후 FFT convolution 적용.
    """
    ir_length = int(sr * decay)
    t = np.linspace(0, decay, ir_length)
    impulse_response = np.exp(-t)
    impulse_response /= np.max(np.abs(impulse_response) + 1e-6)
    wet = fftconvolve(audio, impulse_response, mode='full')[:len(audio)]
    return (1 - mix) * audio + mix * wet

def apply_highpass_filter(audio, sr, cutoff=200):
    """
    Butterworth 고역 필터 적용.
    """
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(2, norm_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio)

def apply_lowpass_filter(audio, sr, cutoff=8000):
    """
    Butterworth 로우패스 필터 적용.
    """
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(2, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio)

def apply_delay_effect(audio, sr, delay_time=0.3, feedback=0.4, mix=0.3):
    """
    간단한 딜레이 효과 적용.
    delay_time: 지연 시간(초), feedback: 피드백 비율, mix: 원신호와 딜레이 신호 혼합 비율.
    """
    delay_samples = int(sr * delay_time)
    delayed = np.zeros(len(audio) + delay_samples)
    delayed[:len(audio)] = audio
    for i in range(len(audio)):
        if i + delay_samples < len(delayed):
            delayed[i + delay_samples] += audio[i] * feedback
    delayed = delayed[:len(audio)]
    return (1 - mix) * audio + mix * delayed

def apply_sidechain_ducking(vocals, inst, strength=0.5, window_size=1024):
    """
    보컬의 에너지를 기반으로 반주 신호 덕킹(ducking) 처리.
    """
    envelope = np.abs(vocals)
    window = np.ones(window_size) / window_size
    envelope_smooth = np.convolve(envelope, window, mode='same')
    envelope_smooth = envelope_smooth / (np.max(envelope_smooth) + 1e-6)
    ducked = inst * (1 - strength * envelope_smooth)
    return ducked

def apply_chorus(audio, sr, depth=0.003, rate=1.5, mix=0.3):
    """
    간단한 코러스 효과 적용.
    depth: 지연 깊이(초), rate: 변조 속도(Hz), mix: 원신호와 코러스 신호 혼합 비율.
    """
    n_samples = len(audio)
    t = np.arange(n_samples) / sr
    mod = depth * np.sin(2 * np.pi * rate * t)
    chorus_audio = np.zeros_like(audio)
    for i in range(n_samples):
        delay = int(mod[i] * sr)
        idx = i - delay
        if 0 <= idx < n_samples:
            chorus_audio[i] = audio[idx]
    return (1 - mix) * audio + mix * chorus_audio

def apply_panning(mono_audio, pan=0.0):
    """
    단일 채널 오디오에 스테레오 패닝 적용.
    pan: -1 (왼쪽) ~ 1 (오른쪽), 0: 중앙.
    """
    left_gain = np.cos((pan + 1) * np.pi / 4)
    right_gain = np.sin((pan + 1) * np.pi / 4)
    stereo = np.vstack((mono_audio * left_gain, mono_audio * right_gain))
    return stereo

# ── 메인 처리 ──────────────────────────────────────────────
try:
    if song1 is not None and song2 is not None:
        if st.button("리믹스 시작"):
            if remix_mode == "기본 오버레이 믹스":
                st.write("단순 오버레이 믹스 진행 중...")
                song1.seek(0)
                song2.seek(0)
                y1, sr1 = librosa.load(song1, sr=None, mono=True)
                y2, sr2 = librosa.load(song2, sr=None, mono=True)
                if sr1 != sr2:
                    y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
                min_len = min(len(y1), len(y2))
                y1, y2 = y1[:min_len], y2[:min_len]
                mixed = (y1 + y2) / 2.0
                mixed = mixed / (np.max(np.abs(mixed)) + 1e-6)
                output_filename = "remix_overlay.wav"
            else:
                st.write("AI 자동 고도화 리믹스 진행 중...")
                st.write("1) 보컬/반주 분리  2) 템포·키 보정  3) 필터, 리버브, 딜레이, 사이드체인, 코러스, 패닝, 컴프레션 적용")
                
                # 파일 바이트 읽기
                song1_bytes = song1.getvalue()
                song2_bytes = song2.getvalue()
                
                # Spleeter를 통한 보컬/반주 분리
                st.write("첫 번째 곡 분리 중...")
                vocals1, accomp1, sr1, temp_dir1 = separate_stems(song1_bytes)
                st.write("두 번째 곡 분리 중...")
                vocals2, accomp2, sr2, temp_dir2 = separate_stems(song2_bytes)
                
                # 샘플레이트 맞추기 (sr1 기준)
                if sr1 != sr2:
                    vocals2 = librosa.resample(vocals2, orig_sr=sr2, target_sr=sr1)
                    accomp2 = librosa.resample(accomp2, orig_sr=sr2, target_sr=sr1)
                    sr2 = sr1
                
                # 보컬 RMS 에너지 계산 → 보컬이 더 두드러진 쪽 선택
                rms_vocals1 = np.sqrt(np.mean(vocals1**2))
                rms_vocals2 = np.sqrt(np.mean(vocals2**2))
                st.write(f"보컬 RMS 에너지: 곡1 = {rms_vocals1:.3f}, 곡2 = {rms_vocals2:.3f}")
                if rms_vocals1 >= rms_vocals2:
                    st.write("첫 번째 곡 보컬과 두 번째 곡 반주 선택")
                    vocal_source = vocals1
                    inst_source = accomp2
                else:
                    st.write("두 번째 곡 보컬과 첫 번째 곡 반주 선택")
                    vocal_source = vocals2
                    inst_source = accomp1
                
                # 템포 분석 및 템포 맞추기
                tempo_voc, _ = librosa.beat.beat_track(y=vocal_source, sr=sr1)
                tempo_inst, _ = librosa.beat.beat_track(y=inst_source, sr=sr1)
                st.write(f"감지된 템포: 보컬 = {tempo_voc:.2f} BPM, 반주 = {tempo_inst:.2f} BPM")
                if np.abs(tempo_voc - tempo_inst) > tempo_tolerance:
                    st.write("템포 차이 존재 → 반주 트랙 템포 보컬에 맞춤")
                    inst_source = match_tempo(inst_source, sr1, tempo_voc, tempo_inst)
                
                # 키 분석 및 보정 (옵션)
                vocal_key_index, vocal_key = detect_key(vocal_source, sr1)
                inst_key_index, inst_key = detect_key(inst_source, sr1)
                st.write(f"감지된 키: 보컬 = {vocal_key}, 반주 = {inst_key}")
                if apply_key_correction:
                    diff = semitone_difference(vocal_key_index, inst_key_index)
                    if diff != 0:
                        st.write(f"반주 트랙의 키를 보컬에 맞추기 위해 {diff} 반음 조정")
                        inst_source = librosa.effects.pitch_shift(inst_source, sr1, n_steps=diff)
                    else:
                        st.write("키 차이 없음")
                else:
                    st.write("키 보정 미적용")
                
                # 사용자 지정 볼륨 적용
                vocal_source = vocal_source * vocal_volume
                inst_source = inst_source * inst_volume
                
                # ── 자동 설정 적용 (옵션) ─────────────────────────────
                if auto_settings:
                    st.write("자동 설정 적용: 분석 결과를 바탕으로 최적 파라미터로 조정합니다.")
                    reverb_decay = 2.5
                    reverb_mix = 0.35
                    if tempo_voc > 0:
                        delay_time = max(0.1, min(1.0, 60/tempo_voc/2))
                    else:
                        delay_time = 0.3
                    delay_feedback = 0.4
                    delay_mix = 0.3
                    hp_cutoff = 200
                    if apply_lp_filter:
                        lp_cutoff = 8000
                    sidechain_strength = 0.5
                    chorus_depth = 0.003
                    chorus_rate = 1.5
                    chorus_mix = 0.3
                    compression_threshold = 0.8
                
                # ── 효과 적용 ───────────────────────────────────────
                if apply_hp_filter:
                    vocal_source = apply_highpass_filter(vocal_source, sr1, hp_cutoff)
                    inst_source = apply_highpass_filter(inst_source, sr1, hp_cutoff)
                if apply_reverb:
                    vocal_source = apply_reverb(vocal_source, sr1, decay=reverb_decay, mix=reverb_mix)
                    inst_source = apply_reverb(inst_source, sr1, decay=reverb_decay, mix=reverb_mix)
                if apply_delay:
                    vocal_source = apply_delay_effect(vocal_source, sr1, delay_time, delay_feedback, delay_mix)
                    inst_source = apply_delay_effect(inst_source, sr1, delay_time, delay_feedback, delay_mix)
                if apply_lp_filter:
                    vocal_source = apply_lowpass_filter(vocal_source, sr1, cutoff=lp_cutoff)
                    inst_source = apply_lowpass_filter(inst_source, sr1, cutoff=lp_cutoff)
                if apply_sidechain:
                    inst_source = apply_sidechain_ducking(vocal_source, inst_source, strength=sidechain_strength)
                if apply_chorus_effect:
                    vocal_source = apply_chorus(vocal_source, sr1, depth=chorus_depth, rate=chorus_rate, mix=chorus_mix)
                    inst_source = apply_chorus(inst_source, sr1, depth=chorus_depth, rate=chorus_rate, mix=chorus_mix)
                
                # 두 트랙 길이 맞추기
                min_len = min(len(vocal_source), len(inst_source))
                vocal_source = vocal_source[:min_len]
                inst_source = inst_source[:min_len]
                
                # 믹스: 합산 후 컴프레션 및 정규화
                mixed_mono = vocal_source + inst_source
                mixed_mono = compress_audio(mixed_mono, threshold=compression_threshold, ratio=compression_ratio)
                mixed_mono = mixed_mono / (np.max(np.abs(mixed_mono)) + 1e-6)
                
                # 스테레오 패닝 적용 (여기서는 중앙 배치)
                vocal_stereo = apply_panning(vocal_source, pan=0.0)
                inst_stereo  = apply_panning(inst_source, pan=0.0)
                mixed_stereo = vocal_stereo + inst_stereo
                mixed_stereo = mixed_stereo * master_volume
                norm_factor = np.max(np.abs(mixed_stereo))
                if norm_factor > 0:
                    mixed_stereo = mixed_stereo / norm_factor
                
                mixed = mixed_stereo.T
                output_filename = "remix_ai_advanced.wav"
                
                # 임시 디렉토리 안전하게 삭제
                safe_rmtree(temp_dir1)
                safe_rmtree(temp_dir2)
            
            # ── 최종 출력: 메모리 내 WAV 파일 생성 ─────────────────────
            buffer = io.BytesIO()
            sf.write(buffer, mixed, sr1, format='WAV')
            buffer.seek(0)
            
            st.audio(buffer, format='audio/wav')
            st.download_button("리믹스 파일 다운로드", data=buffer, file_name=output_filename, mime="audio/wav")
except Exception as e:
    st.error(f"리믹스 처리 중 오류 발생: {e}")
