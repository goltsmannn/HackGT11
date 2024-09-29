import tkinter as tk
from tkinter import messagebox
import threading
import record_squats
#from data_analysis.comparing_graphs import compare_graphs

def start_reference_recording():
    def record_reference_squats():
        record_squats.record_squat_set('reference.csv', num_reps=3)
        messagebox.showinfo("Info", "Reference squats recorded.")
    record_reference_squats()

def start_weighted_recording():
    def record_weighted_squats():
        record_squats.record_squat_set('weighted.csv',  num_reps=3)
        messagebox.showinfo("Info", "Weighted squats recorded.")
    record_weighted_squats()

def view_analysis():
    pass
    #compare_graphs()

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
