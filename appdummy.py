import shutil
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from face_embeddings import generate_face_encodings
import pyttsx3
import customtkinter
from PIL import ImageTk, Image
import os

# Initialize Firebase
cred = credentials.Certificate(
    r"face-attendance-system-921a2-firebase-adminsdk-ynaoj-d384c89ae0.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-attendance-system-921a2-default-rtdb.firebaseio.com/"
})

# Initialize the attendance file path
attendance_file = 'Attendance/attendance.csv'
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

# Initialize attendance data
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    df.to_csv(attendance_file, index=False)
customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System")
        self.root.geometry("800x600")
        # Stack to manage pages
        self.pages = []

        # Initialize the home page
        self.home_page()
        self.engine = pyttsx3.init()
        self.recognized_people = set()
        # Load face encodings
        self.known_encodings = self.load_encodings()
        self.marked_ids = set()

    from PIL import Image, ImageTk
    import tkinter as tk

    def home_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the home page to the stack
        self.pages.append('home')

        # Load background image
        img = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        bg_label = tk.Label(self.root, image=img)
        bg_label.image = img  # Keep a reference to prevent garbage collection
        bg_label.pack(fill="both", expand=True)

        # Create a frame for layout
        layout_frame = tk.Frame(self.root, bg="#ffffff")
        layout_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Frame for the video feed with a fixed size
        self.video_frame = tk.Frame(layout_frame, bg="#000000", width=320, height=240)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Initialize a label to show the live video feed
        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        # Frame for the attendance label
        right_frame = tk.Frame(layout_frame, bg="#ffffff")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Label to display recognized name
        self.attendance_label = tk.Label(right_frame, text="", font=("Arial", 12), fg="green", bg="#ffffff")
        self.attendance_label.pack(pady=5)

        # Configure grid rows and columns to ensure the frames expand correctly
        layout_frame.grid_columnconfigure(0, weight=1)
        layout_frame.grid_columnconfigure(1, weight=1)
        layout_frame.grid_rowconfigure(0, weight=1)

        # Load icon images
        icon_default = ImageTk.PhotoImage(Image.open("Menu Button (1).png"))
        icon_hover = ImageTk.PhotoImage(Image.open("Menu Button.png"))

        # Create hamburger button with icon
        self.hamburger_button = tk.Button(self.root, image=icon_default, bg="#ffffff", borderwidth=0,
                                          command=self.login_page)
        self.hamburger_button.place(x=10, y=10)

        # Bind hover events to change button image
        self.hamburger_button.bind("<Enter>", lambda e: self.hamburger_button.config(image=icon_hover))
        self.hamburger_button.bind("<Leave>", lambda e: self.hamburger_button.config(image=icon_default))

        # Start video feed
        self.start_video()

    def login_page(self):
        # Stop video feed when navigating away
        self.stop_video()

        # Clear the current frame
        self.clear_frame()

        # Add the login page to the stack
        self.pages.append('login')

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for login form
        frame = customtkinter.CTkFrame(master=l1, width=320, height=360, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Add login form widgets
        l2 = customtkinter.CTkLabel(master=frame, text="Log into your Account", font=('Century Gothic', 20))
        l2.place(x=50, y=45)

        self.user_id_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Username')
        self.user_id_entry.place(x=50, y=110)

        self.password_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Password', show="*")
        self.password_entry.place(x=50, y=165)

        # Create custom login button
        login_button = customtkinter.CTkButton(master=frame, width=220, text="Login", command=self.admin_panel_page, corner_radius=6)
        login_button.place(x=50, y=240)

        # Create custom back button
        back_button = customtkinter.CTkButton(master=frame, width=220, text="Back", corner_radius=6, fg_color="#191970", hover_color="#4682B4", command=self.home_page)
        back_button.place(x=50, y=290)

    def admin_panel_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the admin panel page to the stack
        self.pages.append('admin_panel')

        # Load and set background image
        img11 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l11 = customtkinter.CTkLabel(master=self.root, image=img11)
        l11.image = img11  # Keep a reference to prevent garbage collection
        l11.pack(fill="both", expand=True)  # Set the background to fill the window

        # Create a frame for the admin panel on top of the background
        frame = customtkinter.CTkFrame(master=self.root, width=600, height=440, fg_color="#000000", corner_radius=0)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center the frame on the window

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Admin Panel", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Buttons for Admin Panel
        manage_members_button = customtkinter.CTkButton(master=frame, text="Manage Members", width=220,
                                                        command=self.manage_members_page, corner_radius=6,
                                                        fg_color="#00466a", hover_color="#003366")
        manage_members_button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        voice_message_button = customtkinter.CTkButton(master=frame, text="Voice Message", width=220,
                                                       command=self.show_voice_message_page, corner_radius=6,
                                                       fg_color="#00466a", hover_color="#003366")
        voice_message_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        download_data_button = customtkinter.CTkButton(master=frame, text="Download Data", width=220,
                                                       command=self.download_data, corner_radius=6, fg_color="#00466a",
                                                       hover_color="#003366")
        download_data_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        back_button = customtkinter.CTkButton(master=frame, text="Back", width=220, command=self.home_page,
                                              corner_radius=6, fg_color="#191970", hover_color="#4682B4")
        back_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def manage_members_page(self):
        # Clear the current frame
        self.clear_frame()

        # Load and set background image
        img12 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l12 = customtkinter.CTkLabel(master=self.root, image=img12)
        l12.image = img12  # Keep a reference to prevent garbage collection
        l12.pack(fill="both", expand=True)  # Set the background to fill the window

        # Create a frame for the manage members page on top of the background
        frame = customtkinter.CTkFrame(master=self.root, width=600, height=440, fg_color="#000000", corner_radius=0)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center the frame on the window

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Manage Members", font=("Century Gothic", 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Buttons for Manage Members
        add_member_button = customtkinter.CTkButton(master=frame, text="Add Member", width=220,
                                                    command=self.add_member_page,
                                                    corner_radius=6, fg_color="#00466a", hover_color="#003366")
        add_member_button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        remove_member_button = customtkinter.CTkButton(master=frame, text="Remove Member", width=220,
                                                       command=self.remove_member_page, corner_radius=6,
                                                       fg_color="#00466a", hover_color="#003366")
        remove_member_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        view_member_button = customtkinter.CTkButton(master=frame, text="View Member", width=220,
                                                     command=self.view_member_page, corner_radius=6,
                                                     fg_color="#00466a", hover_color="#003366")
        view_member_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        update_data_button = customtkinter.CTkButton(master=frame, text="Update Data", width=220,
                                                     command=self.manage_data_page, corner_radius=6,
                                                     fg_color="#00466a", hover_color="#003366")
        update_data_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        # Back button
        back_button = customtkinter.CTkButton(master=frame, text="Back", width=220, command=self.admin_panel_page,
                                              corner_radius=6, fg_color="#191970", hover_color="#4682B4")
        back_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    def add_member_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the add member page to the stack
        self.pages.append('add_member')

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for adding member form
        frame = customtkinter.CTkFrame(master=self.root, width=320, height=500, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Add Member", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Upload Image Dataset Button
        upload_button = customtkinter.CTkButton(master=frame, text="Upload Image Dataset", width=220,
                                                command=self.upload_image_dataset)
        upload_button.place(x=50, y=80)

        self.upload_folder_path = tk.StringVar()
        folder_label = customtkinter.CTkLabel(master=frame, textvariable=self.upload_folder_path, font=('Arial', 10))
        folder_label.place(x=50, y=120)

        # Employee ID Field
        self.employee_id_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text="Employee ID")
        self.employee_id_entry.place(x=50, y=160)

        # Name Field
        self.name_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text="Name")
        self.name_entry.place(x=50, y=210)

        # Role Field
        self.role_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text="Role")
        self.role_entry.place(x=50, y=260)

        # Add Member Button
        add_member_button = customtkinter.CTkButton(master=frame, width=220, text="Add Member", command=self.add_member,
                                                    corner_radius=6)
        add_member_button.place(x=50, y=320)

        # Back Button
        back_button = customtkinter.CTkButton(master=frame, width=220, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#5b5b5b", command=self.manage_members_page)
        back_button.place(x=50, y=380)

    def remove_member_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the remove member page to the stack
        self.pages.append('remove_member')

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for removing a member
        frame = customtkinter.CTkFrame(master=self.root, width=320, height=300, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Remove Member", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Employee ID Field
        self.remove_id_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text="Employee ID")
        self.remove_id_entry.place(x=50, y=100)

        # Buttons for OK and Back
        ok_button = customtkinter.CTkButton(master=frame, width=100, text="OK", corner_radius=6,
                                            command=self.show_member_details)
        ok_button.place(x=50, y=180)

        back_button = customtkinter.CTkButton(master=frame, width=100, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#4682B4", command=self.manage_members_page)
        back_button.place(x=170, y=180)

    def view_member_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the view member page to the stack
        self.pages.append('view_member')

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for viewing a member
        frame = customtkinter.CTkFrame(master=self.root, width=320, height=250, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="View Member", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Employee ID Field
        self.view_id_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text="Employee ID")
        self.view_id_entry.place(x=50, y=100)

        # Buttons for OK and Back
        ok_button = customtkinter.CTkButton(master=frame, width=100, text="OK", corner_radius=6,
                                            command=self.show_member_details1)
        ok_button.place(x=50, y=180)

        back_button = customtkinter.CTkButton(master=frame, width=100, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#4682B4", command=self.manage_members_page)
        back_button.place(x=170, y=180)

    def manage_data_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the manage data page to the stack
        self.pages.append('manage_data')

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for managing data
        frame = customtkinter.CTkFrame(master=self.root, width=320, height=280, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Manage Data", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Employee ID Field
        self.data_id_entry = customtkinter.CTkEntry(master=frame, width=220, placeholder_text="Employee ID")
        self.data_id_entry.place(x=50, y=100)

        # OK Button to View and Edit Details
        ok_button = customtkinter.CTkButton(master=frame, width=100, text="OK", corner_radius=6,
                                            command=self.view_and_edit_employee)
        ok_button.place(x=50, y=180)

        # Back Button
        back_button = customtkinter.CTkButton(master=frame, width=100, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#4682B4", command=self.manage_members_page)
        back_button.place(x=170, y=180)

    def show_voice_message_page(self):
        pass

    def handle_remove_member(self):
        employee_id = self.remove_id_entry.get()
        if employee_id:
            self.remove_member(employee_id)
        else:
            messagebox.showwarning("Input Error", "Please enter the Employee ID.")

    def upload_image_dataset(self):
        # Open a dialog to select the folder
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.upload_folder_path.set(f"Folder selected: {folder_selected}")
            self.folder_path = folder_selected
            print(f"Selected folder: {folder_selected}")

    def add_member(self):
        employee_id = self.employee_id_entry.get()
        name = self.name_entry.get()
        role = self.role_entry.get()

        if not all([employee_id, name, role, hasattr(self, 'folder_path')]):
            messagebox.showwarning("Input Error", "Please complete all fields and upload the image dataset.")
            return

        # Save images to 'Images' folder
        person_folder = os.path.join('Images', employee_id)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Move images to the 'Images' folder
        for image_name in os.listdir(self.folder_path):
            image_path = os.path.join(self.folder_path, image_name)
            if os.path.isfile(image_path):
                os.rename(image_path, os.path.join(person_folder, image_name))

        # Update face encodings
        generate_face_encodings()

        # Update Firebase with the new member details
        ref = db.reference('Attendance system')
        member_data = {
            "name": name,
            "Role": role,
            "total_attendance": 0,
            "last_attendance_time": "N/A"
        }
        ref.child(employee_id).set(member_data)

        # Show success message
        messagebox.showinfo("Success", "Member added successfully")

        # Navigate back to the manage members page
        self.manage_members_page()

    def remove_member(self, employee_id):
        # Remove from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        ref.delete()

        # Remove from local encodings
        encodings_path = os.path.join('Encodings', f'{employee_id}.npy')
        if os.path.exists(encodings_path):
            os.remove(encodings_path)
        images_folder_path = os.path.join('Images', employee_id)
        if os.path.exists(images_folder_path):
            shutil.rmtree(images_folder_path)

        messagebox.showinfo("Success", "Member removed successfully.")
        self.manage_members_page()

    def show_member_details(self):
        employee_id = self.remove_id_entry.get()

        if not employee_id:
            messagebox.showwarning("Input Error", "Please enter an Employee ID.")
            return

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            messagebox.showwarning("Not Found", "No member found with the provided ID.")
            return

        # Clear the current frame
        self.clear_frame()

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for displaying member details
        frame = customtkinter.CTkFrame(master=self.root, width=320, height=300, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Member Details", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Display member details
        details_text = f"Employee ID: {employee_id}\nName: {member_details['name']}\nRole: {member_details['Role']}"
        member_details_label = customtkinter.CTkLabel(master=frame, text=details_text, font=('Century Gothic', 14))
        member_details_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        # Remove Member Button
        remove_button = customtkinter.CTkButton(master=frame, width=100, text="Remove Member", corner_radius=6,
                                                command=lambda: self.remove_member(employee_id))
        remove_button.place(x=50, y=200)

        # Back Button
        back_button = customtkinter.CTkButton(master=frame, width=100, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#5b5b5b", command=self.manage_members_page)
        back_button.place(x=170, y=200)

    def display_member_details(self, details):
        # Display member details (e.g., in a new window or as labels)
        details_window = tk.Toplevel(self.root)
        details_window.title("Member Details")

        tk.Label(details_window, text=f"Name: {details.get('name', 'N/A')}").pack(pady=5)
        tk.Label(details_window, text=f"Role: {details.get('Role', 'N/A')}").pack(pady=5)
        tk.Label(details_window, text=f"Total Attendance: {details.get('total_attendance', 'N/A')}").pack(pady=5)
        tk.Label(details_window, text=f"Last Attendance Time: {details.get('last_attendance_time', 'N/A')}").pack(
            pady=5)

    def show_member_details1(self):
        employee_id = self.view_id_entry.get()

        if not employee_id:
            messagebox.showwarning("Input Error", "Please enter an Employee ID.")
            return

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            messagebox.showwarning("Not Found", "No member found with the provided ID.")
            self.view_member_page()  # Go back to view member page
            return

        # Clear the current frame
        self.clear_frame()

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Create custom frame for displaying member details
        frame = customtkinter.CTkFrame(master=self.root, width=320, height=300, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Member Details", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Display member details
        details_text = f"Employee ID: {employee_id}\nName: {member_details['name']}\nRole: {member_details['Role']}"
        member_details_label = customtkinter.CTkLabel(master=frame, text=details_text, font=('Century Gothic', 14))
        member_details_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        # Back Button
        back_button = customtkinter.CTkButton(master=frame, width=100, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#5b5b5b", command=self.manage_members_page)
        back_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    def display_member_details1(self, details):
        # Display member details (customize as needed)
        tk.Label(self.root, text=f"Employee ID: {details.get('employee_id', 'N/A')}").pack(pady=5)
        tk.Label(self.root, text=f"Name: {details.get('name', 'N/A')}").pack(pady=5)
        tk.Label(self.root, text=f"Role: {details.get('Role', 'N/A')}").pack(pady=5)
        tk.Label(self.root, text=f"Last Attendance Time: {details.get('last_attendance_time', 'N/A')}").pack(pady=5)

    def view_and_edit_employee(self):
        employee_id = self.data_id_entry.get()

        if not employee_id:
            messagebox.showwarning("Input Error", "Please enter an Employee ID.")
            return

        # Clear the current frame
        self.clear_frame()

        # Load background image
        img1 = ImageTk.PhotoImage(Image.open("gradient-tool-example1.jpg"))
        l1 = customtkinter.CTkLabel(master=self.root, image=img1)
        l1.image = img1  # Keep a reference to prevent garbage collection
        l1.pack(fill="both", expand=True)

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            messagebox.showwarning("Not Found", "No member found with the provided ID.")
            self.manage_data_page()  # Go back to manage data page
            return

        # Create custom frame for viewing and editing member details
        frame = customtkinter.CTkFrame(master=self.root, width=400, height=400, corner_radius=0, fg_color="#000000")
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Title
        title = customtkinter.CTkLabel(master=frame, text="Edit Member Details", font=('Century Gothic', 20))
        title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # Employee ID
        employee_id_label = customtkinter.CTkLabel(master=frame, text="Employee ID:", font=('Arial', 12))
        employee_id_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)
        employee_id_value = customtkinter.CTkLabel(master=frame, text=employee_id, font=('Arial', 12))
        employee_id_value.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

        # Name
        name_label = customtkinter.CTkLabel(master=frame, text="Name:", font=('Arial', 12))
        name_label.place(relx=0.5, rely=0.35, anchor=tk.CENTER)
        self.name_entry = customtkinter.CTkEntry(master=frame, width=200)
        self.name_entry.insert(0, member_details.get('name', ''))
        self.name_entry.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        # Role
        role_label = customtkinter.CTkLabel(master=frame, text="Role:", font=('Arial', 12))
        role_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.role_entry = customtkinter.CTkEntry(master=frame, width=200)
        self.role_entry.insert(0, member_details.get('Role', ''))
        self.role_entry.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

        # Update Button
        update_button = customtkinter.CTkButton(master=frame, width=120, text="Update", corner_radius=6,
                                                command=lambda: self.update_employee(employee_id))
        update_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        # Back Button
        back_button = customtkinter.CTkButton(master=frame, width=120, text="Back", corner_radius=6, fg_color="#191970",
                                              hover_color="#5b5b5b", command=self.manage_members_page)
        back_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    def update_employee(self, employee_id):
        name = self.name_entry.get()
        position = self.role_entry.get()

        if not name or not position:
            messagebox.showwarning("Input Error", "Please fill out all fields.")
            return

        # Update member details in Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        ref.update({
            'name': name,
            'position': position
        })

        messagebox.showinfo("Success", "Member details updated successfully!")
        self.manage_members_page()  # Go back to manage members page
    def download_data(self):
        pass

    def start_video(self):
        # Open webcam and start video feed
        self.cap = cv2.VideoCapture(0)
        self.video_thread = threading.Thread(target=self.update_video_feed)
        self.video_thread.start()

    def update_video_feed(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame and perform face recognition
            self.recognize_face_from_frame(frame)

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            # Update the label with the new image
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)

    def recognize_face_from_frame(self, frame):
        # Recognize faces in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        known_face_encodings = list(self.known_encodings.values())
        known_face_names = list(self.known_encodings.keys())

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                user_id = known_face_names[first_match_index]
                user_data = db.reference(f'Attendance system/{user_id}').get()
                if user_data:
                    name = user_data.get('name', user_id)
                    self.mark_attendance(name, user_id)
                    self.show_attendance_message(name)
                    self.marked_ids.add(user_id)
                    return name
        return None

    def mark_attendance(self, name, user_id):
        """
        Marks attendance by logging the name, date, and time in a CSV file
        and updates the Firebase database.
        """
        now = datetime.now()
        date_str = now.strftime('%d/%m/%Y')
        time_str = now.strftime('%H:%M:%S')

        # Load the current attendance data
        df = pd.read_csv(attendance_file)

        # Check if attendance is already marked for today
        if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
            new_entry = pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(attendance_file, index=False)
            print(f"Attendance marked for {name}")

            # Update the attendance in Firebase
            self.update_firebase_attendance(user_id)

    def show_attendance_message(self, name):
        # Check if the person has already been welcomed
        if name not in self.recognized_people:
            # Add the name to the set of recognized people
            self.recognized_people.add(name)

            # Update the label with the recognized name
            self.attendance_label.config(text=f"Welcome: {name}")

            # Speak the welcome message
            welcome_message = f"Welcome, {name}"
            self.engine.say(welcome_message)
            self.engine.runAndWait()

    def stop_video(self):
        # Stop video feed
        self.cap.release()
        cv2.destroyAllWindows()

    def clear_frame(self):
        # Clears the current frame from the root window
        for widget in self.root.winfo_children():
            widget.destroy()

    def back_to_home(self):
        # Navigates back to the home page
        self.clear_frame()
        self.home_page()

    def load_encodings(self):
        # Load face encodings from the 'Encodings' folder
        encodings = {}
        encodings_folder = 'Encodings'
        if os.path.exists(encodings_folder):
            for file in os.listdir(encodings_folder):
                if file.endswith('.npy'):
                    person_name = file.split('.')[0]
                    encoding_path = os.path.join(encodings_folder, file)
                    encodings[person_name] = np.load(encoding_path)
        return encodings


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()