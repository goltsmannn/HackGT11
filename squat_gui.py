import tkinter as tk
from tkinter import messagebox
import threading
import asyncio  # Import asyncio to use with async functions
import record_squats
import data_analysis

def start_reference_recording():
    def run_recording():
        # Use asyncio.run() to call the async function
        asyncio.run(record_squats.record_squat_set('reference.csv', mode='reference', num_reps=3))
        messagebox.showinfo("Info", "Reference squats recorded.")
    threading.Thread(target=run_recording).start()

def start_weighted_recording():
    def run_recording():
        # Use asyncio.run() to call the async function
        asyncio.run(record_squats.record_squat_set('weighted.csv', mode='weighted', num_reps=3))
        messagebox.showinfo("Info", "Weighted squats recorded.")
    threading.Thread(target=run_recording).start()

def view_analysis():
    data_analysis.analyze_squat_set('reference.csv', 'weighted.csv')

# Setup the GUI
root = tk.Tk()
root.title("Squat Analysis")
root.geometry("300x200")

reference_button = tk.Button(root, text="Record Reference Squats", command=start_reference_recording)
reference_button.pack(pady=10)

weighted_button = tk.Button(root, text="Record Weighted Squats", command=start_weighted_recording)
weighted_button.pack(pady=10)

analysis_button = tk.Button(root, text="View Analysis", command=view_analysis)
analysis_button.pack(pady=10)

root.mainloop()
