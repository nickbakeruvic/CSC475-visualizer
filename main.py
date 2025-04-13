import tkinter as tk
from tkinter import ttk
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
import colorsys
import scipy.signal

pygame.mixer.init()

# Pygame setup
WIDTH, HEIGHT = 800, 400
BAR_WIDTH = 20  # Width of each bar
NUM_BARS = WIDTH // BAR_WIDTH  # Number of bars

# Colors
BACKGROUND_COLOR = (255, 255, 255)
BAR_COLOR = (0, 124, 124)

# Sampling Rate
SRATE = 44100

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
        kicks = processed_data["kicks"]
        snares = processed_data["snares"]
        hihats = processed_data["hihats"]
        print(f"Playing: {file_path}")

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=0, start=0.0)

        global start_time
        start_time = time.time()

        # Trigger the frequency visualization
        if visualizer == 1:
            visualizer_1(file_path, beats, kicks, snares, hihats)
        elif visualizer == 2:
            visualizer_2(file_path, beats)
        elif visualizer == 3:
            visualizer_3(file_path, beats)
    except Exception as e:
        print(f"Error playing the file: {e}")

# Extract audio data from the file and play the animation
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

        kick_index = 0
        snare_index = 0
        hihat_index = 0

        running = True
        while running and pygame.mixer.music.get_busy():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            top_bar_colour_counter += 0.1
            bottom_bar_colour_counter += 0.2

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

            # check if current time is near the next kick time
            draw_kick_square = False
            if kick_index < len(kicks) and abs(audio_pos_sec - kicks[kick_index]) < 0.05:
                draw_kick_square = True
                kick_index += 1  

            draw_snare_circle = False
            if snare_index < len(snares) and abs(audio_pos_sec - snares[snare_index]) < 0.05:
                draw_snare_circle = True
                snare_index += 1

            draw_hihat_triangle = False
            if hihat_index < len(hihats) and abs(audio_pos_sec - hihats[hihat_index]) < 0.05:
                draw_hihat_triangle = True
                hihat_index += 1


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

            
            if draw_kick_square:
                square_size = 50
                square_color = (255, 0, 0)
                square_pos = ((WIDTH - square_size) // 2, (HEIGHT - square_size) // 2)
                pygame.draw.rect(screen, square_color, (*square_pos, square_size, square_size))

            if draw_snare_circle:
                circle_radius = 50
                circle_color = (0, 0, 255)
                circle_pos = (WIDTH // 3, HEIGHT // 2)
                pygame.draw.circle(screen, circle_color, circle_pos, circle_radius)

            if draw_hihat_triangle:
                triangle_color = (0, 255, 0)
                triangle_pos = [(WIDTH * 2 // 3, HEIGHT // 2 - 50), (WIDTH * 2 // 3 - 50, HEIGHT // 2 + 50), (WIDTH * 2 // 3 + 50, HEIGHT // 2 + 50)]
                pygame.draw.polygon(screen, triangle_color, triangle_pos)


            pygame.display.flip()
            clock.tick(30)  # 30 fps

        pygame.display.quit()
        pygame.mixer.music.stop()

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
        chunk_size = 1024  # Samples per frame

        beat_index = 0 

        print(signal)

        # pulsing circle parameters
        beat_pulse_radius = 5
        beat_max_radius = 200
        beat_min_radius = 0
        beat_pulse_decay = 8

        circles = []
        color = 50

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

            frame_signal = signal[start:end].astype(np.float32)
            if frame_signal.size == 0:
                volume = 0.1
            else:
                volume = np.sqrt(np.mean(frame_signal ** 2)) / 32767  
            volume = float(volume)

            if pygame.time.get_ticks() % 1 == 0:
                circles.append(min((volume + 0.1) * 200, beat_max_radius))

            if beat_index < len(beats) and audio_pos_sec >= beats[beat_index]:
                if beat_index % 2 == 0:
                    color += 10
                beat_index += 1  # next beat

            # beat_pulse_radius = max(beat_min_radius, beat_pulse_radius - beat_pulse_decay)  # shrink pulse

            screen.fill(BACKGROUND_COLOR)

            while len(circles) > 10:
                del circles[0]

            # pygame.draw.circle(screen, (0, 255, 0), (WIDTH // 2, HEIGHT // 2), beat_pulse_radius) # opaque circle
            for i in range(len(circles)):
                circle = pygame.Surface((beat_max_radius * 2, beat_max_radius * 2), pygame.SRCALPHA)
                r, g, b = colorsys.hsv_to_rgb((color % 100) * 0.01, 0.5, 0.5)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                pygame.draw.circle(circle, (r, g, b, 50), (beat_max_radius, beat_max_radius), circles[len(circles) - 1 - i])
                # pygame.draw.circle(circle, (255, 255, 255), (beat_max_radius, beat_max_radius), (c - 5))
                screen.blit(circle, (WIDTH // 2 - beat_max_radius, HEIGHT // 2 - beat_max_radius))
                circles[i] -= 10

            pygame.display.flip()
            clock.tick(30)  # 30 fps

        pygame.display.quit()
        pygame.mixer.music.stop()

def get_bass_treble_amplitudes(chunk):
    # Compute FFT on the audio chunk
    spectrum = np.fft.rfft(chunk)
    freq_magnitudes = np.abs(spectrum)
    
    # Generate an array of frequencies corresponding to the FFT bins
    freqs = np.fft.rfftfreq(len(chunk), d=1/SRATE)
    
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
        chunk = signal[start:end].astype(np.float32)
        if chunk.size > 0:
            volume = np.sqrt(np.mean(chunk ** 2)) / 32767.0
            bass_amp, treble_amp = get_bass_treble_amplitudes(chunk)
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

        # Update the phase for smooth movement
        phase_change = max(0.2 - bass_amp * 0.5, 0.02)
        phase += phase_change
        phase_2 += phase_change * 3
        color += 0.2

        # Generate x values (covering the whole screen)
        x_vals = np.linspace(0, WIDTH, num=800)
        x_vals_2 = np.linspace(0, WIDTH, num=800)
        # Compute the sine wave's y values with modulation
        y_vals = (HEIGHT / 2) + amplitude * np.sin(2 * np.pi * base_frequency * (x_vals / WIDTH) + phase)
        y_vals_2 = (HEIGHT / 2) + amplitude * np.sin(2 * np.pi * base_frequency * (x_vals / WIDTH) + phase_2)

        # Draw the sine wave
        screen.fill(BACKGROUND_COLOR)
        points = [(x, y) for x, y in zip(x_vals, y_vals)]
        points_2 = [(x, y) for x, y in zip(x_vals_2, y_vals_2)]

        r, g, b = colorsys.hsv_to_rgb((color % 100) * 0.01, 0.5, 0.5)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        pygame.draw.lines(screen, (r, g, b), False, points, max(3, int(30 * bass_amp)))
        pygame.draw.lines(screen, (r, g, b), False, points_2, max(3, int(30 * bass_amp)))

        pygame.display.flip()
        clock.tick(30)  # maintain 30 fps

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
    song_frames = {}
    for widget in play_buttons_frame.winfo_children():
        widget.destroy()
    for i in range(files.shape[0]):
        song_frames[i] = tk.Frame(play_buttons_frame, background="white")
        song_frames[i].pack(pady=5, fill="x")
        # song_frames[i].grid_columnconfigure((0,1,2), weight=1)

        song_label = ttk.Label(song_frames[i], text=f"{files.index[i]}", background="white")
        song_label.pack(side = "left")

        play_button = ttk.Button(song_frames[i], text="Play", command=lambda file_path=files.iloc[i]["file_path"]: play_audio(file_path), padding=0, width=5)
        play_button.pack(side = "right", padx=10)

        # play_button_3 = ttk.Button(song_frames[i], text="3", command=lambda file_path=files.iloc[i]["file_path"]: play_audio(file_path, 3), padding=0, width=5)
        # play_button_3.pack(side = "right", padx=10)
        # play_button_2 = ttk.Button(song_frames[i], text="2", command=lambda file_path=files.iloc[i]["file_path"]: play_audio(file_path, 2), padding=0, width=5)
        # play_button_2.pack(side = "right", padx=10)
        # play_button_1 = ttk.Button(song_frames[i], text="1", command=lambda file_path=files.iloc[i]["file_path"]: play_audio(file_path, 1), padding=0, width=5)
        # play_button_1.pack(side = "right", padx=10)

        if files.iloc[i]["processed_path"] == "not processed":
            play_button["state"] = "disabled"
        else:
            play_button["state"] = "normal"

def close():
    shutil.rmtree("visualizer_runtime_data") # For now we don't save anything permanently, for testing
    quit()

def check_processing_status():
    draw_play_buttons()
    root.after(1000, check_processing_status)  # check every second

if __name__ == "__main__":

    # Draw upload / play window
    root = tk.Tk()
    root.configure(background='white')

    # root.tk.call('lappend', 'auto_path', 'awthemes-10.4.0')
    # root.tk.call('package', 'require', 'awdark')

    # Styling
    style = ttk.Style(root)
    style.theme_use('clam')
    # style.configure('Modern.TButton',
    #             foreground='white',
    #             background='#0078D7',  # This might be overridden by the native theme on some OSs
    #             padding=0)
    # style.configure()
    
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
    mode_buttons_frame.pack(pady=20, padx=20, fill="x")

    set_mode_label = ttk.Label(mode_buttons_frame, text=f"Current Mode: {get_visualizer()}", background="white")
    set_mode_label.pack(padx=10, side="right")

    # Run the Tkinter event loop
    root.after(1000, check_processing_status)
    root.mainloop()
