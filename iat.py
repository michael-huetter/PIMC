import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from stats import _integrated_autocorrelation_time

import glob
import re

############ Hyperparams ################
max_lag = 50
max_corr_skip = 50

Kin_list = glob.glob('output/*_KinEnergyTrace.csv')
Pot_list = glob.glob('output/*_PotEnergyTrace.csv')
Pos_list = glob.glob('output/*_PositionTrace.csv')

Kin_E = []
Pot_E = []
Positions = []
T_ar = []
for Kin in Kin_list:
    Kin_E.append(np.loadtxt(Kin))
for Pot in Pot_list:
    T = float(Pot.split('\\')[1].split('_')[0])
    T_ar.append(T)
    Pot_E.append(np.loadtxt(Pot))
for Pos in Pos_list:
    Positions.append(np.loadtxt(Pos, delimiter=','))

def generate_data1(corr_skip, temp):
    data = []
    data.append(np.arange(max_lag) + 1)
    pot = Pot_E[T_ar.index(temp)]
    kin = Kin_E[T_ar.index(temp)]
    E = pot + kin
    data.append([])
    for lag in data[0]:
        data[1].append(_integrated_autocorrelation_time(E[::corr_skip], lag))
    return data

def generate_data2(lag, temp):
    data = []
    data.append(np.arange(max_corr_skip) + 1)
    pot = Pot_E[T_ar.index(temp)]
    kin = Kin_E[T_ar.index(temp)]
    E = pot + kin
    data.append([])
    for corr_skip in data[0]:
        data[1].append(_integrated_autocorrelation_time(E[::corr_skip], lag))
    return data

def generate_data3(coord, corr_skip, temp):
    data = []
    data.append(np.arange(max_lag) + 1)
    coords = Positions[T_ar.index(temp)][:, coord]
    data.append([])
    for lag in data[0]:
        try:
            iat = _integrated_autocorrelation_time(coords[::corr_skip], lag)
            data[1].append(iat)
        except Exception as e:
            print(f"Error computing IAT for coord {coord}, lag {lag}, skip {corr_skip}: {e}")
            data[1].append(np.nan)  # Handle errors gracefully
    return data

def generate_data4(coord, lag, temp):
    data = []
    data.append(np.arange(max_corr_skip) + 1)
    coords = Positions[T_ar.index(temp)][:, coord]
    data.append([])
    for corr_skip in data[0]:
        try:
            iat = _integrated_autocorrelation_time(coords[::corr_skip], lag)
            data[1].append(iat)
        except Exception as e:
            print(f"Error computing IAT for coord {coord}, lag {lag}, skip {corr_skip}: {e}")
            data[1].append(np.nan)  # Handle errors gracefully
    return data

def update_plot1(event=None):
    try:
        temp = float(temp_dropdown.get())
        corr_skip = int(entry1.get())
        ax1.clear()
        data = generate_data1(corr_skip, temp)
        ax1.bar(data[0], data[1], width=0.5)
        ax1.set_xlabel('lag')
        ax1.set_ylabel('Integrated Autocorrelation Time')
        ax1.set_title('Plot 1: lag vs iat(Energy)')
        ax1.grid(True)
        canvas1.draw()
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def update_plot2(event=None):
    try:
        temp = float(temp_dropdown.get())
        lag = int(entry2.get())
        ax2.clear()
        data = generate_data2(lag, temp)
        ax2.bar(data[0], data[1], width=0.5)
        ax2.set_xlabel('correlation skip')
        ax2.set_ylabel('Integrated Autocorrelation Time')
        ax2.set_title('Plot 2: correlation skip vs iat(Energy)')
        ax2.grid(True)
        canvas2.draw()
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def update_plot3(event=None):
    try:
        temp = float(temp_dropdown.get())
        corr_skip = int(entry3.get())
        coord = int(coord_dropdown.get())
        ax3.clear()
        data = generate_data3(coord, corr_skip, temp)
        ax3.bar(data[0], data[1], width=0.5)
        ax3.set_xlabel('lag')
        ax3.set_ylabel('Integrated Autocorrelation Time')
        ax3.set_title('Plot 3: lag vs iat(Position)')
        ax3.grid(True)
        canvas3.draw()
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def update_plot4(event=None):
    try:
        temp = float(temp_dropdown.get())
        lag = int(entry4.get())
        coord = int(coord_dropdown.get())
        ax4.clear()
        data = generate_data4(coord, lag, temp)
        ax4.bar(data[0], data[1], width=0.5)
        ax4.set_xlabel('correlation skip')
        ax4.set_ylabel('Integrated Autocorrelation Time')
        ax4.set_title('Plot 4: correlation skip vs iat(Position)')
        ax4.grid(True)
        canvas4.draw()
    except ValueError:
        print("Invalid input. Please enter valid numbers.")

def on_closing():
    root.destroy()
    plt.close('all')  # Close all matplotlib figures

root = tk.Tk()
root.title("Interactive Plot")

# Create fig, ax, canvas
fig1, ax1 = plt.subplots(figsize=(8, 4))
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, columnspan=4)

fig2, ax2 = plt.subplots(figsize=(8, 4))
canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2.get_tk_widget().grid(row=2, column=0, padx=10, pady=10, columnspan=4)

fig3, ax3 = plt.subplots(figsize=(8, 4))
canvas3 = FigureCanvasTkAgg(fig3, master=root)
canvas3.get_tk_widget().grid(row=0, column=4, padx=10, pady=10, columnspan=4)

fig4, ax4 = plt.subplots(figsize=(8, 4))
canvas4 = FigureCanvasTkAgg(fig4, master=root)
canvas4.get_tk_widget().grid(row=2, column=4, padx=10, pady=10, columnspan=4)

# First plot options
entry1_label = tk.Label(root, text="Correlation Skip:")
entry1_label.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

entry1 = tk.Entry(root)
entry1.insert(0, '1')  # Default value
entry1.grid(row=1, column=2, padx=10, pady=10, sticky='ew')
entry1.bind("<FocusOut>", update_plot1)  # Update plot on entry field change

# Second plot options
entry2_label = tk.Label(root, text="Lag:")
entry2_label.grid(row=3, column=0, padx=10, pady=10, sticky='ew')

entry2 = tk.Entry(root)
entry2.insert(0, '1')  # Default value
entry2.grid(row=3, column=1, padx=10, pady=10, sticky='ew')
entry2.bind("<FocusOut>", update_plot2)  # Update plot on entry field change

# Third plot options
entry3_label = tk.Label(root, text="Correlation Skip:")
entry3_label.grid(row=1, column=5, padx=10, pady=10, sticky='ew')

entry3 = tk.Entry(root)
entry3.insert(0, '1')  # Default value
entry3.grid(row=1, column=6, padx=10, pady=10, sticky='ew')
entry3.bind("<FocusOut>", update_plot3)  # Update plot on entry field change

# Fourth plot options
entry4_label = tk.Label(root, text="Lag:")
entry4_label.grid(row=3, column=5, padx=10, pady=10, sticky='ew')

entry4 = tk.Entry(root)
entry4.insert(0, '1')  # Default value
entry4.grid(row=3, column=6, padx=10, pady=10, sticky='ew')
entry4.bind("<FocusOut>", update_plot4)  # Update plot on entry field change


# Temperature and Coordinate
temp_label = tk.Label(root, text="Temperature:")
temp_label.grid(row=4, column=0, padx=10, pady=10, sticky='ew')

temp_options = T_ar
temp_dropdown = ttk.Combobox(root, values=temp_options)
temp_dropdown.set(temp_options[0])
temp_dropdown.grid(row=4, column=1, padx=10, pady=10, sticky='ew')
temp_dropdown.bind("<<ComboboxSelected>>", lambda event: (update_plot1(), update_plot2(), update_plot3(), update_plot4()))

coord_label = tk.Label(root, text="Coord: (only affects plots 3 and 4)")
coord_label.grid(row=4, column=2, padx=10, pady=10, sticky='ew')

coord_options = list(np.arange(len(Positions[0][0])))
coord_dropdown = ttk.Combobox(root, values=coord_options)
coord_dropdown.set(coord_options[0])
coord_dropdown.grid(row=4, column=3, padx=10, pady=10, sticky='ew')
coord_dropdown.bind("<<ComboboxSelected>>", lambda event: (update_plot3(), update_plot4()))

# Initialize the plots
update_plot1()
update_plot2()
update_plot3()
update_plot4()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()