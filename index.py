from tkinter import *
from flask import Flask,redirect, url_for,render_template,request
import os
import tkinter as tk
import tkinter.font as tkfont

def d_dtcn():
	root = Tk()
	root.configure(background = "white")

	def function1(): 
		os.system("python sober.py --shape_predictor shape_predictor_68_face_landmarks.dat")
		exit()

	def function2(): 
		os.system("python android_cam.py --shape_predictor shape_predictor_68_face_landmarks.dat")
		exit()

	



		
	root.title("Scan")
	root.configure(bg="#333333")
	Label(root, text="SoberScan",font=("times new roman",20),fg="white",bg="black",height=2).grid(row=2,rowspan=2,columnspan=5,sticky=N+E+W+S,padx=5,pady=10)
	Button(root,text="Run using web cam",font=("times new roman",20),bg="#03f8fc",fg='white',command=function1).grid(row=5,columnspan=5,sticky=W+E+N+S,padx=5,pady=5)
	Button(root,text="Identify",font=("times new roman",20),bg="#03f8fc",fg='white',command=function2).grid(row=7,columnspan=5,sticky=W+E+N+S,padx=5,pady=5)
	Button(root,text="Exit",font=("times new roman",20),bg="#03f8fc",fg='white',command=root.destroy).grid(row=9,columnspan=5,sticky=W+E+N+S,padx=5,pady=5)

	root.mainloop()