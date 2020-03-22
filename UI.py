import tkinter as tk
from tkinter.filedialog import askdirectory
import os
from main import *

# from main import *

filenames = []
N = ''

def open_file():

	filename = askdirectory()
	if filename:
		filenames.append(filename)
		L1['text']= filenames[-1]

def show():
	if len(filenames):
		Augment(E1.get(),filenames[-1])
	else:
		print("Dataset Not chosen")
	root.destroy()

root = tk.Tk()

B1 = tk.Button(root, text='Open File', command=open_file)
B1.grid(row=0, column=1)
B1.pack()

L1 = tk.Label(root, text='')
L1.grid(row=0, column=2)
L1.pack()

L2 = tk.Label(root, text="Number of Images to Augment")
L2.grid(row=1, column=1)
L2.pack()

E1 = tk.Entry(root, bd =5)
E1.grid(row=1, column=2)
E1.pack()

B2 = tk.Button(root, text='Augment', command=show)
B2.grid(row=2, column=1)
B2.pack()

root.mainloop()