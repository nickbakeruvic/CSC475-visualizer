import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pygame
import numpy as np
from numpy.fft import fft
import wave
import pandas as pd
import os
import shutil
import librosa
import time
from multiprocessing import Process
import json
import colorsys
import scipy.signal
import random

pygame.mixer.init()

# Define constants
WIDTH, HEIGHT = 800, 400
BAR_WIDTH = 20 
NUM_BARS = WIDTH // BAR_WIDTH 
BACKGROUND_COLOR = (255, 255, 255)
BAR_COLOR = (0, 124, 124)
SRATE = 44100
MEL_SIZE = 20

# Visualizer selection variable
visualizer = 1

def set_visualizer(num):
    global visualizer
    visualizer = num
    draw_visualizer()

def get_visualizer():
    global visualizer
    return visualizer

def draw_visualizer():
    global visualizer
    set_mode_label.configure(text=f"Current Mode: {visualizer}")

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

# Process a file upload, getting beats and locations of various drums
def process_upload(file_path, name):
    processed_path = f"visualizer_runtime_data/processed_audio/{name}.json"

    beats = get_beats(file_path)
    kicks, snares, hihats = get_drum_hits(file_path)

    with open(processed_path, "w") as f:
        f.write(json.dumps({
            "beats": beats.tolist(),
            "kicks": kicks.tolist(),
            "snares": snares.tolist(),
            "hihats": hihats.tolist()
        }))

    files = get_files()
    files.loc[name, "processed_path"] = processed_path
    set_files(files)

# Get location in song of beats 
def get_beats(file_path):
    y, sr = librosa.load(file_path, sr=SRATE)

    # Only beat track on the percussive components
    _, y_percussive = librosa.effects.hpss(y)
    _, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Convert frames to seconds
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return beat_times

# Get location in song of drum hits
def get_drum_hits(file_path, SRATE=44100):
    y, sr = librosa.load(file_path, sr=SRATE)
    n_fft = 2048
    hop_length = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def detect_hits(S_db, freq_range, percentile=95, min_ms=100, smooth=5):
        if smooth <= 3:
            smooth = 5
        if smooth % 2 == 0:
            smooth += 1
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        energy = np.sum(S_db[mask, :], axis=0)
        smoothed = scipy.signal.savgol_filter(energy, window_length=smooth, polyorder=3)
        threshold = np.percentile(smoothed, percentile)
        min_distance = int(min_ms * sr / hop_length / 1000)
        peaks, _ = scipy.signal.find_peaks(smoothed, height=threshold, distance=min_distance)
        return librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # kick and snare with spectral energy method
    kicks = detect_hits(S_db, (20, 150), percentile=95, min_ms=100)
    snares = detect_hits(S_db, (150, 2500), percentile=97, min_ms=150)

    # hihats with onset strength method
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmax=15000, aggregate=np.median)
    threshold = np.percentile(onset_env, 93)
    min_distance = int(0.015 * sr / hop_length)  # ~20ms
    peaks, _ = scipy.signal.find_peaks(onset_env, height=threshold, distance=min_distance)
    hihats = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    return kicks, snares, hihats

# Parse file path to find the name of the song
def get_file_name(file_path):
    return file_path.split("/")[-1].split(".")[0]

# Return part of files csv related to the particular file path specified, as JSON
def get_processed_data(file_path):
    files = get_files()
    with open(files.loc[get_file_name(file_path), "processed_path"], "r") as f:
        return json.loads(f.read())

# Begin visualizer, triggered by "Play" buttons
def play_audio(file_path):
    processed_data = get_processed_data(file_path)
    beats = processed_data["beats"]
    kicks = processed_data["kicks"]
    snares = processed_data["snares"]
    hihats = processed_data["hihats"]
    print(f"Playing: {file_path}")

    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play(loops=0, start=0.0)

    global start_time
    start_time = time.time()

    if visualizer == 1:
        visualizer_1(file_path, beats, kicks, snares, hihats)
    elif visualizer == 2:
        visualizer_2(file_path, beats)
    elif visualizer == 3:
        visualizer_3(file_path, beats)
    elif visualizer == 4:
        visualizer_4(file_path, beats, kicks, snares, hihats)

# Visualizer 1 is the EQ graph
def visualizer_1(file_path, beats, kicks, snares, hihats):
    with wave.open(file_path, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        num_samples = wave_file.getnframes()
        signal = wave_file.readframes(num_samples)
        signal = np.frombuffer(signal, dtype=np.int16)

        if wave_file.getnchannels() == 2:
            signal = signal[::2] + signal[1::2]  # Convert stereo to mono

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Audio Visualizer 1")

        clock = pygame.time.Clock()
        chunk_size = 1024  # Samples per frame

        top_bar_colour_counter = 50
        bottom_bar_colour_counter = 20

        running = True
        while running and pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            top_bar_colour_counter += 0.1
            bottom_bar_colour_counter += 0.2

            # Get current position in the song
            audio_pos_sec = time.time() - start_time  # time since playback started - changed to this for use with beat tracking, unsure if necessary but results seem better
            frame = int(audio_pos_sec * framerate)

            start = frame
            end = start + chunk_size
            if end > len(signal):
                end = len(signal)

            # Process FFT in chunks
            spectrum = np.fft.fft(signal[start:end])
            freq_magnitudes = np.abs(spectrum[:len(spectrum) // 2])

            # Aggregate freqs into bars
            bin_size = len(freq_magnitudes) // NUM_BARS
            binned_freqs = [np.mean(freq_magnitudes[i * bin_size:(i + 1) * bin_size]) for i in range(NUM_BARS)]

            # Normalize values to fit on screen
            max_amplitude = max(binned_freqs) if max(binned_freqs) > 0 else 1
            heights = [int((val / max_amplitude) * (HEIGHT / 2)) for val in binned_freqs]

            # Draw bars
            screen.fill(BACKGROUND_COLOR)
            for i in range(NUM_BARS):
                x = i * BAR_WIDTH
                y = (HEIGHT / 2) - heights[i]
                r, g, b = colorsys.hsv_to_rgb((top_bar_colour_counter % 100) * 0.01, 0.5, 0.5)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                pygame.draw.rect(screen, (r, g, b), (x, y + (HEIGHT / 2), BAR_WIDTH - 2, heights[i]))

                r, g, b = colorsys.hsv_to_rgb((bottom_bar_colour_counter % 100) * 0.01, 0.5, 0.5)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                pygame.draw.rect(screen, (r, g, b), (x, 0, BAR_WIDTH - 2, heights[i]))

            pygame.display.flip()
            clock.tick(30)  # 30 fps

        pygame.display.quit()
        pygame.mixer.music.stop()

# From https://github.com/iranroman/musicinformationretrieval.com/blob/gh-pages/realtime_spectrogram.py
def compute_spectrogram(chunk):

    # Choose the frequency range of your log-spectrogram
    F_LO = librosa.note_to_hz('C2')
    F_HI = librosa.note_to_hz('C9')
    M = librosa.filters.mel(sr = SRATE, n_fft = len(chunk), n_mels = WIDTH / MEL_SIZE, fmin=F_LO, fmax=F_HI)

    # Compute real FFT
    x_fft = np.fft.rfft(chunk)

    # Compute mel spectrum
    melspectrum = M.dot(abs(x_fft))

    return melspectrum

# Visualizer 2 uses beat locations, volume information, and spectrogram calculations to create pulsing circle with spectrogram in the background
def visualizer_2(file_path, beats):
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
        chunk_size = 4096  # Samples per frame

        beat_index = 0 

        print(signal)

        # pulsing circle parameters
        beat_max_radius = 200

        circles = []
        spects = []
        color = 50

        running = True
        while running and pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get current position in the song
            audio_pos_sec = time.time() - start_time  # time since playback started - changed to this for use with beat tracking, unsure if necessary but results seem better
            frame = int(audio_pos_sec * framerate)

            start = frame
            end = start + chunk_size
            if end > len(signal):
                end = len(signal)

            # Set current frame and get information from it
            frame_signal = signal[start:end].astype(np.float32)
            if frame_signal.size == 0:
                volume = 0.1
            else:
                volume = np.sqrt(np.mean(frame_signal ** 2)) / 32767  
                spectrogram = compute_spectrogram(frame_signal)
                spects.insert(0, spectrogram)
            volume = float(volume)

            # Append a new circle to the circles list if 1 second has passed
            if pygame.time.get_ticks() % 1 == 0:
                circles.append(min((volume + 0.1) * 200, beat_max_radius))

            # If a beat hit then change colour of spectogram and circles
            if beat_index < len(beats) and audio_pos_sec >= beats[beat_index]:
                if beat_index % 2 == 0:
                    color += 10
                beat_index += 1  # next beat

            screen.fill(BACKGROUND_COLOR)

            # Clean up circles array to remove old circles
            while len(circles) > 10:
                del circles[0]

            # Compute colour RGB and draw spectrogram
            r, g, b = colorsys.hsv_to_rgb((color % 100) * 0.01, 0.5, 0.5)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            rectangle = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(rectangle, (255, 255, 255), pygame.Rect(0, 0, WIDTH, HEIGHT))
            for i in range(len(spects)):
                for j in range(len(spects[i])):
                    if spects[i][j] > 90000:
                        pygame.draw.rect(rectangle, (r, g, b, 50), pygame.Rect(j * MEL_SIZE, HEIGHT - (MEL_SIZE * i), MEL_SIZE, MEL_SIZE))
            screen.blit(rectangle, (0, 0))

            # Draw pulsing circles
            for i in range(len(circles)):
                circle = pygame.Surface((beat_max_radius * 2, beat_max_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(circle, (r, g, b, 50), (beat_max_radius, beat_max_radius), circles[len(circles) - 1 - i])
                screen.blit(circle, (WIDTH // 2 - beat_max_radius, HEIGHT // 2 - beat_max_radius))
                circles[i] -= 10

            pygame.display.flip()
            clock.tick(30)  # 30 fps

        pygame.display.quit()
        pygame.mixer.music.stop()

# Get the amplitudes of bass and treble in a given frame signal
def get_bass_treble_amplitudes(frame_signal):
    # Compute FFT on the audio chunk
    spectrum = np.fft.rfft(frame_signal)
    freq_magnitudes = np.abs(spectrum)
    
    # Generate an array of frequencies corresponding to the FFT bins
    freqs = np.fft.rfftfreq(len(frame_signal), d=1/SRATE)
    
    # Define frequency ranges (in Hz) for bass and treble
    bass_range = (20, 250)
    treble_range = (4000, 20000)
    
    # Find indices corresponding to the bass and treble ranges
    bass_indices = np.where((freqs >= bass_range[0]) & (freqs <= bass_range[1]))
    treble_indices = np.where((freqs >= treble_range[0]) & (freqs <= treble_range[1]))
    
    # Calculate the average amplitude for each range
    bass_amp = np.mean(freq_magnitudes[bass_indices])
    treble_amp = np.mean(freq_magnitudes[treble_indices])
    
    return bass_amp, treble_amp

# Visualizer 3 uses bass and treble amplitude, overall volume, and beat locations to draw pulsing sin waves
def visualizer_3(file_path, beats):

    # Open the audio file
    with wave.open(file_path, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        num_samples = wave_file.getnframes()
        signal = wave_file.readframes(num_samples)
        signal = np.frombuffer(signal, dtype=np.int16)
        
        # Convert stereo to mono if needed
        if wave_file.getnchannels() == 2:
            signal = signal[::2] + signal[1::2]

    # Set up the Pygame display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sine Wave Visualizer")
    clock = pygame.time.Clock()
    
    # Visualization parameters
    chunk_size = 1024  
    phase = 0.0     
    phase_2 = 0.0   
    base_frequency = 2  
    beat_index = 0 
    color = 50
    
    running = True
    while running and pygame.mixer.music.get_busy():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get the current position in the audio (in seconds)
        audio_pos_sec = time.time() - start_time
        frame = int(audio_pos_sec * framerate)
        start = frame
        end = start + chunk_size
        if end > len(signal):
            end = len(signal)

        # Extract the current chunk and compute its volume
        frame_signal = signal[start:end].astype(np.float32)
        if frame_signal.size > 0:
            volume = np.sqrt(np.mean(frame_signal ** 2)) / 32767.0
            bass_amp, treble_amp = get_bass_treble_amplitudes(frame_signal)
            bass_amp /= 10000000
            treble_amp /= 10000000
        else:
            bass_amp, treble_amp, volume = 0

        # Use the volume to modulate the amplitude of the sine wave
        amplitude = volume * (HEIGHT / 3)  # scale amplitude to screen height

        if beat_index < len(beats) and audio_pos_sec >= beats[beat_index]:
            if beat_index % 2 == 0:
                amplitude *= 1.5
            beat_index += 1  # next beat

        # Update the phase and colour of the sin waves
        phase_change = max(0.2 - bass_amp * 0.5, 0.02)
        phase += phase_change
        phase_2 += phase_change * 3
        color += 0.2

        # Generate x values (covering the whole screen)
        x_vals = np.linspace(0, WIDTH, num=800)
        x_vals_2 = np.linspace(0, WIDTH, num=800)
        # Compute the sine wave's y values 
        y_vals = (HEIGHT / 2) + amplitude * np.sin(2 * np.pi * base_frequency * (x_vals / WIDTH) + phase)
        y_vals_2 = (HEIGHT / 2) + amplitude * np.sin(2 * np.pi * base_frequency * (x_vals / WIDTH) + phase_2)

        # Draw the sine wave
        screen.fill(BACKGROUND_COLOR)
        points = [(x, y) for x, y in zip(x_vals, y_vals)]
        points_2 = [(x, y) for x, y in zip(x_vals_2, y_vals_2)]

        # Compute colour RGB and draw sin waves
        r, g, b = colorsys.hsv_to_rgb((color % 100) * 0.01, 0.5, 0.5)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        pygame.draw.lines(screen, (r, g, b), False, points, max(3, int(30 * bass_amp)))
        pygame.draw.lines(screen, (r, g, b), False, points_2, max(3, int(30 * bass_amp)))

        pygame.display.flip()
        clock.tick(30)  # maintain 30 fps

    pygame.display.quit()
    pygame.mixer.music.stop()

# Visualizer 4 uses the drum and beat locations to create a particle effect in time with the music
def visualizer_4(file_path, beats, kicks, snares, hihats):
    class Particle:
        def __init__(self, x, y, vx, vy, size, color, lifetime):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.size = size
            self.color = color
            self.lifetime = lifetime
            self.initial_lifetime = lifetime

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.lifetime -= 1

        def draw(self, screen):
            if self.lifetime > 0:
                alpha = max(0, int(255 * (self.lifetime / self.initial_lifetime)))
                particle_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, (*self.color, alpha), (self.size, self.size), self.size)
                screen.blit(particle_surface, (int(self.x - self.size), int(self.y - self.size)))

    # Loading audio file
    with wave.open(file_path, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        num_samples = wave_file.getnframes()
        signal = wave_file.readframes(num_samples)
        signal = np.frombuffer(signal, dtype=np.int16)

        if wave_file.getnchannels() == 2:
            signal = signal[::2] + signal[1::2]  # Convert stereo to mono

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Drum Beats Visualizer")
        clock = pygame.time.Clock()

        chunk_size = 1024
        kick_index = snare_index = hihat_index = beat_index = 0
        particles = []
        start_time = time.time()

        background_change_duration = 5 # frames
        background_change_current = 0

        # Separate / measure frequency bands for background changes
        def get_frequency_bands(signal, chunk_size):
            chunk = signal[:chunk_size]
            fft_result = fft(chunk)
            magnitude = np.abs(fft_result)[:chunk_size // 2]
            if np.max(magnitude) != 0:
                magnitude = magnitude / np.max(magnitude)

            low_energy = np.sum(magnitude[:50])
            mid_energy = np.sum(magnitude[50:300])
            high_energy = np.sum(magnitude[300:])

            return low_energy, mid_energy, high_energy

        running = True
        while running and pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            audio_pos_sec = time.time() - start_time
            frame = int(audio_pos_sec * framerate)
            start = frame
            end = min(start + chunk_size, len(signal))

            # Spawn particles at a random place not too close to the edges of the screen
            center_x, center_y = random.uniform(WIDTH // 4, WIDTH * 3 // 4), random.uniform(HEIGHT // 4, HEIGHT * 3 // 4)
            chunk = signal[start:end].astype(np.float32)

            # Measure RMS to scale background change visual amplitude
            if chunk.size > 0:
                rms = np.sqrt(np.mean(chunk**2)) / 32767  # Normalize to [0,1]
            else:
                rms = 0

            # Blend low, mid, and high colors together based on the amplitude
            # in those respective frequencies
            low_energy, mid_energy, high_energy = get_frequency_bands(signal[start:end], chunk_size)

            # Colors to be part of the final background. Low is mainly blue,
            # mid is mainly green, high is mainly red. The other non-main values
            # for each color were random, tweaked based on what looks best.
            low_color = np.array([105, 34, 255])
            mid_color = np.array([53, 255, 138])
            high_color = np.array([255, 59, 77])

            total_energy = low_energy + mid_energy + high_energy
            if total_energy == 0:
                weights = np.array([0, 0, 0])
            else:
                weights = np.array([
                    low_energy / total_energy,
                    mid_energy / total_energy,
                    high_energy / total_energy
                ])
            mixed_color = (low_color * weights[0] +
               mid_color * weights[1] +
               high_color * weights[2])

            brightness = min(0.5, rms * 2 * (0.1 + weights[1] + weights[2]))
            final_color = np.clip(mixed_color * brightness, 0, 255).astype(int)

            background_color = tuple(final_color)

            # Spawn different particles based on each drum type
            # Generate random angles and speeds for each particle (drums and snares)
            if kick_index < len(kicks) and abs(audio_pos_sec - kicks[kick_index]) < 0.05:
                for _ in range(15):
                    angle = random.uniform(0, 2 * np.pi)
                    speed = random.uniform(2, 6)
                    particles.append(Particle(center_x, center_y, speed * np.cos(angle),
                                              speed * np.sin(angle), random.randint(5, 10),
                                              (255, 50, 50), 30))
                kick_index += 1

            if snare_index < len(snares) and abs(audio_pos_sec - snares[snare_index]) < 0.05:
                for _ in range(30):
                    angle = random.uniform(0, 2 * np.pi)
                    radius = random.uniform(30, 70)
                    speed = random.uniform(2, 6)
                    particles.append(Particle(center_x + radius * np.cos(angle),
                                              center_y + radius * np.sin(angle),
                                              speed * np.cos(angle), speed * np.sin(angle),
                                              3,(100, 100, 255), 40))
                snare_index += 1

            if hihat_index < len(hihats) and abs(audio_pos_sec - hihats[hihat_index]) < 0.03:
                for _ in range(20):
                    particles.append(Particle(random.randint(0, WIDTH), random.randint(0, HEIGHT),
                                              0, 0, 2, (200, 255, 200), 10))
                hihat_index += 1

            # Flash a background color based on the blend above on each beat
            if beat_index < len(beats) and abs(audio_pos_sec - beats[beat_index]) < 0.05:
                background_change_current = 0
                beat_index += 1
            if background_change_current < background_change_duration:
                background_change_current += 1
                screen.fill(background_color)
            else:
                screen.fill((0, 0, 0))

            # Draw all the particles
            for p in particles[:]:
                p.update()
                p.draw(screen)
                if p.lifetime <= 0:
                    particles.remove(p)

            pygame.display.flip()
            clock.tick(30)

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

        # Reload play buttons now that we have the new song added to the files csv
        draw_play_buttons()

        # Start a subprocess which will process the new upload, getting beats and drum locations
        p = Process(target=process_upload, args=(file_path, name))
        p.start()

# Draw play buttons and song labels, keeping track of which files are ready to be played
def draw_play_buttons():
    files = get_files()
    song_frames = {}

    # Clear existing buttons and labels
    for widget in play_buttons_frame.winfo_children():
        widget.destroy()

    for i in range(files.shape[0]):
        song_frames[i] = tk.Frame(play_buttons_frame, background="white")
        song_frames[i].pack(pady=5, fill="x")

        song_label = ttk.Label(song_frames[i], text=f"{files.index[i]}", background="white")
        song_label.pack(side = "left")

        play_button = ttk.Button(song_frames[i], text="Play", command=lambda file_path=files.iloc[i]["file_path"]: play_audio(file_path), padding=0, width=5)
        play_button.pack(side = "right", padx=10)

        if files.iloc[i]["processed_path"] == "not processed":
            play_button["state"] = "disabled"
        else:
            play_button["state"] = "normal"

# Removes all stored data for the program
def close():
    shutil.rmtree("visualizer_runtime_data") # For now we don't save anything permanently, for testing
    quit()

# Continuously check if new files are ready to be played
def check_processing_status():
    draw_play_buttons()
    root.after(1000, check_processing_status)  # check every second

if __name__ == "__main__":

    # Draw upload / play window
    root = tk.Tk()
    root.configure(background='white')

    # Styling
    style = ttk.Style(root)
    style.theme_use('clam')
    
    root.title("File Upload and Play with Visualizer")
    root.geometry("700x400")

    # Make quit and upload buttons
    menu_frame = tk.Frame(root)
    menu_frame.configure(background="white")
    upload_button = ttk.Button(menu_frame, text="Quit", command=close)
    upload_button.pack(pady=0, side="right")

    upload_button = ttk.Button(menu_frame, text="Upload a file", command=upload)
    upload_button.pack(pady=0, side="left")
    menu_frame.pack(pady=5)

    # Make play buttons
    play_buttons_frame = tk.Frame(root)
    play_buttons_frame.configure(background="white", borderwidth=2, relief="sunken")
    play_buttons_frame.pack(pady=20, padx=20, fill="x")
    draw_play_buttons()

    # Mode buttons
    mode_buttons_frame = tk.Frame(root)
    mode_buttons_frame.configure(background="white")
    mode_label = ttk.Label(mode_buttons_frame, text=f"Choose Visualizer Mode:", background="white")
    mode_label.pack(side = "left", padx=10)
    play_button_1 = ttk.Button(mode_buttons_frame, command=lambda num=1: set_visualizer(num), text="1", padding=0, width=5)
    play_button_1.pack(side = "left", padx=10)
    play_button_2 = ttk.Button(mode_buttons_frame, command=lambda num=2: set_visualizer(num), text="2", padding=0, width=5)
    play_button_2.pack(side = "left", padx=10)
    play_button_3 = ttk.Button(mode_buttons_frame, command=lambda num=3: set_visualizer(num), text="3", padding=0, width=5)
    play_button_3.pack(side = "left", padx=10)
    play_button_4 = ttk.Button(mode_buttons_frame, command=lambda num=4: set_visualizer(num), text="4", padding=0, width=5)
    play_button_4.pack(side = "left", padx=10)
    mode_buttons_frame.pack(pady=20, padx=20, fill="x")

    # Mode label (currently selected mode)
    set_mode_label = ttk.Label(mode_buttons_frame, text=f"Current Mode: {get_visualizer()}", background="white")
    set_mode_label.pack(padx=10, side="right")

    # Run the Tkinter event loop
    root.after(1000, check_processing_status)
    root.mainloop()