import time
import streamlit as st
import numpy as np
import librosa
from scipy.signal import coherence
from constants import NL_MEDIAN, NL_RANGE, ZCR_MEDIAN, HARMONIC_CONTENT, PIANO_KEYS, INDEX_TO_CAT, WEIGHTS

def ingest(file: np.ndarray, fs: float) -> list[float]:
    trim_sig = np.trim_zeros(file, trim='fb')
    x = np.array_split(trim_sig, 50)
    zcr = [librosa.feature.zero_crossing_rate(i).mean(axis=1) for i in x]
    noise = np.random.normal(0,1,len(trim_sig))
    _, cohere = coherence(trim_sig,noise,fs)
    real_fft = np.abs(np.fft.rfft(trim_sig, 65536))
    fft_freqs = np.linspace(0, 22050, len(real_fft), True)
    harmonic_energy = 0
    for frequency, value in zip(fft_freqs[:16000], real_fft[:16000]):
        for harmonic_freq in PIANO_KEYS:
            if 1.02*harmonic_freq > frequency > 0.98*harmonic_freq:
                harmonic_energy += value
    total_harmonic_energy = (harmonic_energy/np.sum(real_fft))*100
    return [np.median(cohere), np.max(cohere) - np.min(cohere), total_harmonic_energy, np.median(zcr)]

st.header('Statistical Audio Category Classifier', divider='rainbow')
st.write('This program uses predefined weighted probabilities that for a given statistic an audio file will belong to the 5 categories: Effects, Human, Music, Nature and Urban.')
uploaded_files = st.file_uploader("Upload audio files to classify", accept_multiple_files=True)
st.divider()
st.subheader('Results')
for uploaded_file in uploaded_files:
    filename = uploaded_file.name
    probabilities = [0,0,0,0,0]
    file_as_wav, fs = librosa.load(path=uploaded_file, sr=None)
    ingest_start_time = time.time()
    file_stats = ingest(file_as_wav, fs)
    ingest_end_time = time.time() - ingest_start_time
    classify_start_time = time.time()
    for stat, method, weight in zip(file_stats, [NL_MEDIAN, NL_RANGE, HARMONIC_CONTENT, ZCR_MEDIAN], WEIGHTS):
        probability_points = list(method.keys())
        if stat >= probability_points[-1]:
            probability = [0,0,0,0,0]
        elif stat <= probability_points[0]:
            probability = np.multiply(np.log(method[probability_points[0]]),[weight])
        else:
            for index, point in enumerate(probability_points[0:-2]):
                if probability_points[index+1] > stat > point:
                    probability = np.multiply(np.log(method[probability_points[index+1]]), [weight])
                    break
        probabilities = np.add(probabilities,probability)
    guessed_cat = INDEX_TO_CAT[np.argmax(probabilities)]
    classify_end_time = time.time() - classify_start_time
    st.write(f"{filename} is category: {guessed_cat}")
    st.write(f"The computed probabilities are {probabilities}")
    st.write(f"The time to calculate statistics was: {np.round(ingest_end_time, 2)}s")
    st.write(f"The time to classify was: {np.round(classify_end_time, 5)}s")
    st.write(f"The total time was: {np.round(classify_end_time+ingest_end_time, 3)}s")
    st.divider()
