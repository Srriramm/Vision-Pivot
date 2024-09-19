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
# Initialize Firebase
cred = credentials.Certificate(
    r"C:\Users\SRIRAM\PycharmProjects\fd\face-attendance-system-921a2-firebase-adminsdk-ynaoj-d384c89ae0.json")
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

    def home_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the home page to the stack
        self.pages.append('home')

        # Title
        title = tk.Label(self.root, text="Live Attendance Entries", font=("Arial", 18))
        title.pack(pady=20)

        # Frame for the video feed
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack()

        # Initialize a label to show the live video feed
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Label to display recognized name
        self.attendance_label = tk.Label(self.root, text="", font=("Arial", 12), fg="green")
        self.attendance_label.pack(pady=5)

        # Start video feed
        self.start_video()

        # Hamburger button for navigation
        hamburger_button = tk.Button(self.root, text="≡", font=("Arial", 16), command=self.login_page)
        hamburger_button.place(x=10, y=10)

    def login_page(self):
        # Stop video feed when navigating away
        self.stop_video()

        # Clear the current frame
        self.clear_frame()

        # Add the login page to the stack
        self.pages.append('login')

        # Title
        title = tk.Label(self.root, text="Login", font=("Arial", 18))
        title.pack(pady=20)

        # User ID
        user_id_label = tk.Label(self.root, text="User ID", font=("Arial", 12))
        user_id_label.pack(pady=10)
        self.user_id_entry = tk.Entry(self.root)
        self.user_id_entry.pack(pady=5)

        # Password
        password_label = tk.Label(self.root, text="Password", font=("Arial", 12))
        password_label.pack(pady=10)
        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.pack(pady=5)

        # Button Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        # Submit button
        submit_button = tk.Button(button_frame, text="Submit", command=self.admin_panel_page)
        submit_button.pack(side=tk.LEFT, padx=10)

        # Back button
        back_button = tk.Button(button_frame, text="Back", command=self.back_to_home)
        back_button.pack(side=tk.LEFT, padx=10)

    def admin_panel_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the admin panel page to the stack
        self.pages.append('admin_panel')

        # Title
        title = tk.Label(self.root, text="Admin Panel", font=("Arial", 18))
        title.pack(pady=20)

        # Buttons for Admin Panel
        manage_members_button = tk.Button(self.root, text="Manage Members", width=20, command=self.manage_members_page)
        manage_members_button.pack(pady=10)

        voice_message_button = tk.Button(self.root, text="Voice Message", width=20, command=self.show_voice_message_page)
        voice_message_button.pack(pady=10)

        download_data_button = tk.Button(self.root, text="Download Data", width=20, command=self.download_data)
        download_data_button.pack(pady=10)
        back_button = tk.Button(self.root, text="Back", command=self.home_page)
        back_button.pack(pady=10)

    def manage_members_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the manage members page to the stack
        self.pages.append('manage_members')

        # Title
        title = tk.Label(self.root, text="Manage Members", font=("Arial", 18))
        title.pack(pady=20)

        # Buttons for Manage Members
        add_member_button = tk.Button(self.root, text="Add Member", width=20, command=self.add_member_page)
        add_member_button.pack(pady=10)

        remove_member_button = tk.Button(self.root, text="Remove Member", width=20, command=self.remove_member_page)
        remove_member_button.pack(pady=10)

        view_member_button = tk.Button(self.root, text="View Member", width=20, command=self.view_member_page)
        view_member_button.pack(pady=10)

        update_data_button = tk.Button(self.root, text="Update Data", width=20, command=self.manage_data_page)
        update_data_button.pack(pady=10)

        # Back button
        back_button = tk.Button(self.root, text="Back", command=self.admin_panel_page)
        back_button.pack(pady=20)

    def add_member_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the add member page to the stack
        self.pages.append('add_member')

        # Title
        title = tk.Label(self.root, text="Add Member", font=("Arial", 18))
        title.pack(pady=20)

        # Upload Image Dataset
        upload_frame = tk.Frame(self.root)
        upload_frame.pack(pady=10)

        upload_button = tk.Button(upload_frame, text="Upload Image Dataset", command=self.upload_image_dataset)
        upload_button.pack()

        self.upload_folder_path = tk.StringVar()
        folder_label = tk.Label(upload_frame, textvariable=self.upload_folder_path)
        folder_label.pack(pady=5)

        # Enter Details
        details_frame = tk.Frame(self.root)
        details_frame.pack(pady=10)

        tk.Label(details_frame, text="Enter Details", font=("Arial", 14)).pack(pady=10)

        tk.Label(details_frame, text="Employee ID").pack(pady=5)
        self.employee_id_entry = tk.Entry(details_frame)
        self.employee_id_entry.pack(pady=5)

        tk.Label(details_frame, text="Name").pack(pady=5)
        self.name_entry = tk.Entry(details_frame)
        self.name_entry.pack(pady=5)

        tk.Label(details_frame, text="Role").pack(pady=5)
        self.role_entry = tk.Entry(details_frame)
        self.role_entry.pack(pady=5)

        add_member_button = tk.Button(self.root, text="Add Member", command=self.add_member)
        add_member_button.pack(pady=20)

        # Back button
        back_button = tk.Button(self.root, text="Back", command=self.manage_members_page)
        back_button.pack(pady=10)

    def remove_member_page(self):
        # Clear the current frame
        self.clear_frame()

        # Add the remove member page to the stack
        self.pages.append('remove_member')

        # Title
        title = tk.Label(self.root, text="Remove Member", font=("Arial", 18))
        title.pack(pady=20)

        # Enter Employee ID
        id_frame = tk.Frame(self.root)
        id_frame.pack(pady=10)

        tk.Label(id_frame, text="Employee ID", font=("Arial", 12)).pack(pady=5)
        self.remove_id_entry = tk.Entry(id_frame)
        self.remove_id_entry.pack(pady=5)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        ok_button = tk.Button(button_frame, text="OK", command=self.show_member_details)
        ok_button.pack(side=tk.LEFT, padx=10)

        back_button = tk.Button(button_frame, text="Back", command=self.manage_members_page)
        back_button.pack(side=tk.LEFT, padx=10)
    def view_member_page(self):
        # Clear the current page
        for widget in self.root.winfo_children():
            widget.destroy()

        # Page to enter Employee ID
        tk.Label(self.root, text="Enter Employee ID:").pack(pady=10)
        self.view_id_entry = tk.Entry(self.root)
        self.view_id_entry.pack(pady=5)

        ok_button = tk.Button(self.root, text="OK", command=self.show_member_details1)
        ok_button.pack(pady=10)

        back_button = tk.Button(self.root, text="Back", command=self.manage_members_page)
        back_button.pack(pady=10)

    def manage_data_page(self):
        # Clear the current page
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Manage Data").pack(pady=10)

        # Entry for employee ID
        tk.Label(self.root, text="Enter Employee ID:").pack(pady=5)
        self.data_id_entry = tk.Entry(self.root)
        self.data_id_entry.pack(pady=5)

        # OK button to view details
        ok_button = tk.Button(self.root, text="OK", command=self.view_and_edit_employee)
        ok_button.pack(pady=5)

        # Back button
        back_button = tk.Button(self.root, text="Back", command=self.manage_members_page)
        back_button.pack(pady=20)

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

        # Show member details
        self.display_member_details(member_details)

        # Add remove button
        remove_button = tk.Button(self.root, text="Remove Member", command=lambda: self.remove_member(employee_id))
        remove_button.pack(pady=20)

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

        # Clear the current page
        for widget in self.root.winfo_children():
            widget.destroy()

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            messagebox.showwarning("Not Found", "No member found with the provided ID.")
            self.view_member_page()  # Go back to view member page
            return

        # Display member details
        self.display_member_details1(member_details)

        # Back button
        back_button = tk.Button(self.root, text="Back", command=self.manage_members_page)
        back_button.pack(pady=20)

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

        # Clear the current page
        for widget in self.root.winfo_children():
            widget.destroy()

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            messagebox.showwarning("Not Found", "No member found with the provided ID.")
            self.manage_data_page()  # Go back to manage data page
            return

        # Display and edit member details
        tk.Label(self.root, text="Edit Member Details").pack(pady=10)

        tk.Label(self.root, text="Employee ID:").pack(pady=5)
        tk.Label(self.root, text=member_details.get('employee_id', 'N/A')).pack(pady=5)

        tk.Label(self.root, text="Name:").pack(pady=5)
        self.name_entry = tk.Entry(self.root)
        self.name_entry.insert(0, member_details.get('name', ''))
        self.name_entry.pack(pady=5)

        tk.Label(self.root, text="Role:").pack(pady=5)
        self.position_entry = tk.Entry(self.root)
        self.position_entry.insert(0, member_details.get('Role', ''))
        self.position_entry.pack(pady=5)

        # Update button
        update_button = tk.Button(self.root, text="Update", command=lambda: self.update_employee(employee_id))
        update_button.pack(pady=5)

        # Back button
        back_button = tk.Button(self.root, text="Back", command=self.manage_data_page)
        back_button.pack(pady=20)

    def update_employee(self, employee_id):
        name = self.name_entry.get()
        position = self.position_entry.get()

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