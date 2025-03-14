import tkinter as tk
from tkinter import filedialog
import pygame

pygame.mixer.init()

def upload_and_play():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    if file_path:
        play_button.config(state=tk.NORMAL)
        play_button.config(command=lambda: play_audio(file_path))

def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print(f"Playing: {file_path}")
    except Exception as e:
        print(f"Error playing the file: {e}")

root = tk.Tk()
root.title("File Upload and Play")
root.geometry("300x200")

upload_button = tk.Button(root, text="Upload a file", command=upload_and_play)
upload_button.pack(pady=50)

play_button = tk.Button(root, text="Play", state=tk.DISABLED)
play_button.pack(pady=20)

root.mainloop()
