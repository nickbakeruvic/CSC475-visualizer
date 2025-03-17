import tkinter as tk
from tkinter import filedialog
import pygame
import numpy as np
import wave
import os
import shutil

pygame.mixer.init()

# Pygame setup
WIDTH, HEIGHT = 1600, 800
BAR_WIDTH = 20  # Width of each bar
NUM_BARS = WIDTH // BAR_WIDTH  # Number of bars

# Colors
BACKGROUND_COLOR = (255, 255, 255)
BAR_COLOR = (0, 124, 124)

# Get function for files csv/dataframe
def get_files():
    os.makedirs("visualizer_runtime_data/audio_files", exist_ok=True)
    return {}  # Using a dictionary for simplicity instead of CSV

# Called by "Play" button
def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=0, start=0.0)
        print(f"Playing: {file_path}")

        # Start visualization
        visualize_audio(file_path)
    except Exception as e:
        print(f"Error playing the file: {e}")

# Extract audio and visualize
def visualize_audio(file_path):
    with wave.open(file_path, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        num_samples = wave_file.getnframes()
        signal = wave_file.readframes(num_samples)
        signal = np.frombuffer(signal, dtype=np.int16)

        if wave_file.getnchannels() == 2:
            signal = signal[::2] + signal[1::2]  # Convert stereo to mono

        # Initialize pygame window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Reinitialize screen
        pygame.display.set_caption("Audio Visualizer")

        clock = pygame.time.Clock()
        chunk_size = 1024  # Number of samples per frame

        running = True
        while running and pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get current position in the song
            audio_pos_ms = pygame.mixer.music.get_pos()  # Milliseconds
            frame = int(audio_pos_ms / 1000 * framerate)

            start = frame
            end = start + chunk_size
            if end > len(signal):
                end = len(signal)

            # Perform FFT
            spectrum = np.fft.fft(signal[start:end])
            freq_magnitudes = np.abs(spectrum[:len(spectrum) // 2])  # Take first half

            # Normalize and bin frequencies
            bin_size = len(freq_magnitudes) // NUM_BARS
            binned_freqs = [np.mean(freq_magnitudes[i * bin_size:(i + 1) * bin_size]) for i in range(NUM_BARS)]

            # Normalize values
            max_amplitude = max(binned_freqs) if max(binned_freqs) > 0 else 1
            heights = [int((val / max_amplitude) * HEIGHT) for val in binned_freqs]

            # Draw bars
            screen.fill(BACKGROUND_COLOR)
            for i in range(NUM_BARS):
                x = i * BAR_WIDTH
                y = HEIGHT - heights[i]
                pygame.draw.rect(screen, BAR_COLOR, (x, y, BAR_WIDTH - 2, heights[i]))

            pygame.display.flip()
            clock.tick(30)  # Run at ~30 FPS

        pygame.display.quit()  # Close window, but keep pygame running

# Uploading file
def upload():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        name = os.path.basename(file_path).split(".")[0]
        file_path = shutil.copy(file_path, f"visualizer_runtime_data/audio_files/{name}.wav")
        files[name] = file_path
        draw_play_buttons()

def draw_play_buttons():
    for widget in play_buttons_frame.winfo_children():
        widget.destroy()
    for i, (name, file_path) in enumerate(files.items()):
        tk.Button(play_buttons_frame, text=f"Play {name}", command=lambda fp=file_path: play_audio(fp)).pack(pady=5)

def close():
    shutil.rmtree("visualizer_runtime_data")
    quit()

# GUI setup
files = get_files()
root = tk.Tk()
root.title("File Upload and Play with Visualizer")
root.geometry("500x300")

menu_frame = tk.Frame(root)
tk.Button(menu_frame, text="Quit", command=close).pack(side="right")
tk.Button(menu_frame, text="Upload a file", command=upload).pack(side="left")
menu_frame.pack(pady=5)

play_buttons_frame = tk.Frame(root)
play_buttons_frame.pack(pady=20)
draw_play_buttons()

root.mainloop()
