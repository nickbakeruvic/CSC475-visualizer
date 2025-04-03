import tkinter as tk
from tkinter import filedialog
import pygame
import numpy as np
import wave
import pandas as pd
import os
import shutil
import librosa
import time
from multiprocessing import Process
import json

pygame.mixer.init()

# Pygame setup
WIDTH, HEIGHT = 1600, 800
BAR_WIDTH = 20  # Width of each bar
NUM_BARS = WIDTH // BAR_WIDTH  # Number of bars

# Colors
BACKGROUND_COLOR = (255, 255, 255)
BAR_COLOR = (0, 124, 124)

# Sampling Rate
SRATE = 44100

# Get function for files csv/dataframe
def get_files():
    os.makedirs("visualizer_runtime_data/audio_files", exist_ok=True)
    os.makedirs("visualizer_runtime_data/processed_audio", exist_ok=True)
    if os.path.isfile("visualizer_runtime_data/files.csv"):
        files = pd.read_csv("visualizer_runtime_data/files.csv", index_col=0)
    else:
        files = pd.DataFrame(columns=["file_path", "processed_path"])

    return files

def set_files(files):
    files.to_csv("visualizer_runtime_data/files.csv")

def process_upload(file_path, name):
    processed_path = f"visualizer_runtime_data/processed_audio/{name}.json"

    beats = get_beats(file_path)

    with open(processed_path, "w") as f:
        f.write(json.dumps({
            "beats": beats.tolist()
        }))

    files = get_files()
    files.loc[name, "processed_path"] = processed_path
    set_files(files)

def get_beats(file_path):
    y, sr = librosa.load(file_path, sr=SRATE)

    print("getting beats")

    # Only beat track on the percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Convert frames to seconds
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print("finished getting beats")

    return beat_times

def get_file_name(file_path):
    return file_path.split("/")[-1].split(".")[0]

def get_processed_data(file_path):
    files = get_files()
    with open(files.loc[get_file_name(file_path), "processed_path"], "r") as f:
        return json.loads(f.read())

# Called by "Play" button
def play_audio(file_path):
    try:
        processed_data = get_processed_data(file_path)
        beats = processed_data["beats"]
        print(f"Playing: {file_path}")

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=0, start=0.0)

        global start_time
        start_time = time.time()

        # Trigger the frequency visualization
        visualize_audio(file_path, beats)
    except Exception as e:
        print(f"Error playing the file: {e}")

# Extract audio data from the file and play the animation
def visualize_audio(file_path, beats):
    with wave.open(file_path, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        num_samples = wave_file.getnframes()
        signal = wave_file.readframes(num_samples)
        signal = np.frombuffer(signal, dtype=np.int16)

        if wave_file.getnchannels() == 2:
            signal = signal[::2] + signal[1::2]  # Convert stereo to mono

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Audio Visualizer")

        clock = pygame.time.Clock()
        chunk_size = 1024  # Samples per frame

        beat_index = 0 

        # pulsing circle parameters
        beat_pulse_radius = 5
        beat_max_radius = 300
        beat_min_radius = 0
        beat_pulse_decay = 8

        running = True
        while running and pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get current position in the song
            # audio_pos_ms = pygame.mixer.music.get_pos()  # Milliseconds
            # audio_pos_sec = pygame.mixer.music.get_pos() / 1000.0  # Convert ms → sec
            audio_pos_sec = time.time() - start_time  # time since playback started - changed to this for use with beat tracking, unsure if necessary but results seem better
            # frame = int(audio_pos_ms / 1000 * framerate)
            frame = int(audio_pos_sec * framerate)

            start = frame
            end = start + chunk_size
            if end > len(signal):
                end = len(signal)

            if beat_index < len(beats) and audio_pos_sec >= beats[beat_index]:
                if beat_index % 2 == 0:
                    beat_pulse_radius = beat_max_radius  # expand pulse
                beat_index += 1  # next beat

            beat_pulse_radius = max(beat_min_radius, beat_pulse_radius - beat_pulse_decay)  # shrink pulse


            # Process FFT in chunks
            spectrum = np.fft.fft(signal[start:end])
            freq_magnitudes = np.abs(spectrum[:len(spectrum) // 2])

            # Aggregate freqs into bars
            bin_size = len(freq_magnitudes) // NUM_BARS
            binned_freqs = [np.mean(freq_magnitudes[i * bin_size:(i + 1) * bin_size]) for i in range(NUM_BARS)]

            # Normalize values to fit on screen
            max_amplitude = max(binned_freqs) if max(binned_freqs) > 0 else 1
            heights = [int((val / max_amplitude) * HEIGHT) for val in binned_freqs]

            # Draw bars
            screen.fill(BACKGROUND_COLOR)
            for i in range(NUM_BARS):
                x = i * BAR_WIDTH
                y = HEIGHT - heights[i]
                pygame.draw.rect(screen, BAR_COLOR, (x, y, BAR_WIDTH - 2, heights[i]))

            # pygame.draw.circle(screen, (0, 255, 0), (WIDTH // 2, HEIGHT // 2), beat_pulse_radius) # opaque circle

            # draw circle that is transparent
            circle = pygame.Surface((beat_max_radius * 2, beat_max_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle, (124, 91, 176, 64), (beat_max_radius, beat_max_radius), beat_pulse_radius)
            screen.blit(circle, (WIDTH // 2 - beat_max_radius, HEIGHT // 2 - beat_max_radius))

            pygame.display.flip()
            clock.tick(30)  # 30 fps

        pygame.display.quit()
        pygame.mixer.music.stop()

# Uploading file
def upload():
    files = get_files()

    # Parse file path input
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    name = get_file_name(file_path)
    file_path = shutil.copyfile(file_path, "visualizer_runtime_data/audio_files/" + name + ".wav")

    # Parse song and add to csv's
    if file_path:
        files.loc[name, "file_path"] = file_path
        files.loc[name, "processed_path"] = "not processed"
        set_files(files)

        draw_play_buttons()

        p = Process(target=process_upload, args=(file_path, name))
        p.start()

def draw_play_buttons():
    files = get_files()
    play_buttons = {}
    for widget in play_buttons_frame.winfo_children():
        widget.destroy()
    for i in range(files.shape[0]):
        play_buttons[i] = tk.Button(play_buttons_frame, text=f"Play {files.index[i]}", command=lambda file_path=files.iloc[i]["file_path"]: play_audio(file_path))
        play_buttons[i].pack(pady=5)
        if files.iloc[i]["processed_path"] == "not processed":
            play_buttons[i]["state"] = "disabled"
        else:
            play_buttons[i]["state"] = "normal"

def close():
    shutil.rmtree("visualizer_runtime_data") # For now we don't save anything permanently, for testing
    quit()

def check_processing_status():
    draw_play_buttons()
    root.after(1000, check_processing_status)  # check every second

if __name__ == "__main__":

    # Draw upload / play window
    root = tk.Tk()
    root.title("File Upload and Play with Visualizer")
    root.geometry("500x300")

    # Make quit and upload buttons
    menu_frame = tk.Frame(root)
    upload_button = tk.Button(menu_frame, text="Quit", command=close)
    upload_button.pack(pady=0, side="right")

    upload_button = tk.Button(menu_frame, text="Upload a file", command=upload)
    upload_button.pack(pady=0, side="left")
    menu_frame.pack(pady=5)

    # Make play buttons
    play_buttons_frame = tk.Frame(root)
    play_buttons_frame.pack(pady=20)
    draw_play_buttons()

    # Run the Tkinter event loop
    root.after(1000, check_processing_status)
    root.mainloop()
