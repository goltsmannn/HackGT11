import tkinter as tk
from tkinter import messagebox
import asyncio
import threading
import time
import squat_analysis  # Import the squat analysis module

# Global state to track whether a reference squat has been recorded
reference_recorded = False

# Function to run asyncio tasks in a separate thread to avoid freezing the GUI
def run_asyncio_task(task, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task(*args))

# Function to start reference squat recording
def start_reference_squat():
    def reference_task():
        status_label.config(text="Recording reference squat... Get ready!")
        countdown_timer(3)  # 3-second countdown
        status_label.config(text="Recording in progress...")
        run_asyncio_task(squat_analysis.main, 'reference')
        global reference_recorded
        reference_recorded = True
        status_label.config(text="Reference squat recorded!")
        weighted_button.config(state='normal')  # Enable the weighted squat button

    # Start the recording in a separate thread
    threading.Thread(target=reference_task).start()

# Function to start weighted squat recording
def start_weighted_squat():
    if not reference_recorded:
        messagebox.showerror("Error", "You must record a reference squat first!")
        return

    def weighted_task():
        status_label.config(text="Recording weighted squat... Get ready!")
        countdown_timer(3)  # 3-second countdown
        status_label.config(text="Recording in progress...")
        run_asyncio_task(squat_analysis.main, 'weighted')
        status_label.config(text="Weighted squat recorded!")

    # Start the recording in a separate thread
    threading.Thread(target=weighted_task).start()

# Countdown timer function to show a timer before starting squat recording
def countdown_timer(seconds):
    for i in range(seconds, 0, -1):
        status_label.config(text=f"Starting in {i}...")
        root.update()  # Ensure the GUI updates during the countdown
        time.sleep(1)

# Create the main application window
root = tk.Tk()
root.title("Squat Analysis")
root.geometry("400x300")

# Create UI elements
reference_button = tk.Button(root, text="Start Reference Squat", command=start_reference_squat, width=25, height=2)
reference_button.pack(pady=20)

weighted_button = tk.Button(root, text="Start Weighted Squat", command=start_weighted_squat, width=25, height=2, state='disabled')
weighted_button.pack(pady=20)

status_label = tk.Label(root, text="Press a button to start.")
status_label.pack(pady=20)

# Run the GUI event loop
root.mainloop()