import tkinter as tk
from tkinter import filedialog
import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave

pygame.mixer.init()

# Called by "Play" button
def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=0, start=0.0)
        print(f"Playing: {file_path}")

        # Trigger the frequency visualization
        visualize_audio(file_path)
    except Exception as e:
        print(f"Error playing the file: {e}")

# Extract audio data from the file and play the animation
def visualize_audio(file_path):
    with wave.open(file_path, 'rb') as wave_file:
        framerate = wave_file.getframerate()
        num_samples = wave_file.getnframes()
        signal = wave_file.readframes(num_samples)
        signal = np.frombuffer(signal, dtype=np.int16)

        # Convert stereo to mono by averaging the two channels
        if wave_file.getnchannels() == 2:
            signal = signal[::2] + signal[1::2]

        # Initialize matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, framerate / 2, 512)  # Frequency range

        # Set chunk size for FFT
        chunk_size = 1024

        max_amplitude = 0
        for start in range(0, len(signal), chunk_size):
            end = start + chunk_size
            if end > len(signal):
                end = len(signal)

            spectrum = np.fft.fft(signal[start:end])  # Process in chunks
            freq = np.abs(spectrum)[:len(spectrum) // 2]
            max_amplitude = max(max_amplitude, np.max(freq))

        def update_plot(i):
            # Get the current position of the audio in milliseconds
            audio_pos_ms = pygame.mixer.music.get_pos()  # In milliseconds
            frame = int(audio_pos_ms / 1000 * framerate)  # Convert ms to frame position

            # Make sure the frame stays within bounds
            if frame >= len(signal):
                return False

            start = frame
            end = start + chunk_size
            if end > len(signal):
                end = len(signal)

            # Process FFT in chunks
            spectrum = np.fft.fft(signal[start:end])
            freq = np.abs(spectrum)[:len(spectrum) // 2]

            # Draw new frame (clear old one)
            ax.clear()
            ax.plot(x, freq)
            ax.set_ylim(0, max_amplitude * 1.1)  # Make sure the y-axis doesn't jump around

            ax.set_title("Frequency Spectrum")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")

            if pygame.mixer.music.get_busy():
                return True
            else:
                return False

        # Actually draw our visualizer (will need to do something better)
        ani = animation.FuncAnimation(fig, update_plot, interval=33, repeat=False)  # 30 FPS

        def on_close(event):
            pygame.mixer.music.stop()
            plt.close(fig)

        fig.canvas.mpl_connect('close_event', on_close)  # Bind the close event

        # Show frame
        plt.show()


# Uploading file
def upload():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    # Show and enable the play button after a file is selected
    if file_path:
        play_button.config(state=tk.NORMAL)
        play_button.config(command=lambda: play_audio(file_path))

# Draw upload / play window
root = tk.Tk()
root.title("File Upload and Play with Visualizer")
root.geometry("300x200")

# Creat buttons within above window
upload_button = tk.Button(root, text="Upload a file", command=upload)
upload_button.pack(pady=50)

play_button = tk.Button(root, text="Play", state=tk.DISABLED)  # Initially disabled
play_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
