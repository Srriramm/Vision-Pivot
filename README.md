# VisionPivot Attendance System

A comprehensive face recognition-based attendance management system built with Python, PyQt5, and Firebase integration.

## 🚀 Quick Start Guide

### Step 1: Activate Virtual Environment

**For Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Setup Script

```bash
python setup.py
```

This will:
- Create necessary directories (Images, Encodings, Attendance)
- Install all required dependencies
- Check for configuration files

### Step 4: Start the Application

```bash
python start_app.py
```

**Default Login Credentials:**
- Username: `test`
- Password: `test`

---

## 📋 System Requirements

- Python 3.7 or higher
- Webcam or external camera
- Firebase project (optional - system works without it)
- Windows/Linux/macOS

## 🔧 Features Overview

### 🏠 Home Page
- **Start Attendance System**: Begin real-time face recognition
- **Admin Login**: Access administrative functions
- **Dual Camera Support**: Entry and exit camera monitoring

### 👤 Admin Panel
- **Member Management**: Add, remove, and edit employee information
- **Data Management**: View and export attendance data
- **Voice Messages**: Set custom voice announcements for employees
- **Camera Configuration**: Adjust camera settings

### 👥 Member Management

#### Adding New Members
1. Go to Admin Panel → Manage Members → Add Member
2. Fill in required details:
   - Employee ID
   - Name
   - Role
   - Email (optional)
3. **Upload Image Dataset**: Click "Upload Image Dataset" and select a folder containing **6-8 clear photos** of the person
4. Click "Add Member" to save

**Important Notes for Adding Members:**
- Each person needs 6-8 high-quality photos
- Photos should be clear, well-lit, and show the person's face clearly
- Different angles and expressions improve recognition accuracy
- Recommended image size: 300x300 to 500x500 pixels
- Supported formats: JPG, JPEG, PNG

#### Managing Existing Members
- **View Members**: See all registered employees with their details
- **Edit Members**: Update employee information
- **Remove Members**: Delete employee records and associated data

### 📊 Data Management

#### Attendance Tracking
- **Real-time Recognition**: Continuous face detection and recognition
- **Automatic Marking**: Entry and exit time recording
- **Duplicate Prevention**: Prevents multiple entries for the same person
- **Cloud Sync**: Automatic synchronization with Firebase (if configured)

#### Data Export
- **Daily Reports**: Export today's attendance data
- **Custom Date Range**: Export data for specific periods
- **Excel Format**: Professional Excel reports with formatting
- **CSV Backup**: Local CSV file storage

### 🎤 Voice & Notifications

#### Voice Announcements
- **Custom Messages**: Set personalized welcome messages for each employee
- **Text-to-Speech**: Automatic voice announcements when someone is recognized
- **Voice Settings**: Adjust speech rate, volume, and voice selection

#### Email Notifications
- **Automatic Emails**: Send attendance notifications to employees
- **Entry/Exit Alerts**: Notify when someone enters or leaves
- **Customizable Templates**: Personalize email content

### 📷 Camera Management

#### Camera Configuration
- **Entry Camera**: Configure camera for entry monitoring
- **Exit Camera**: Configure camera for exit monitoring
- **Multiple Sources**: Support for webcams and external cameras
- **Real-time Preview**: Live camera feed display

#### Camera Settings
- **Camera Selection**: Choose from available camera sources
- **Resolution Settings**: Optimize camera resolution
- **Performance Tuning**: Adjust settings for better recognition

## 🗂️ Project Structure

```
VisionPivot/
├── visionpivot_final.py          # Main application file
├── face_embeddings.py            # Face encoding generation
├── requirements.txt              # Python dependencies
├── setup.py                     # Setup script
├── start_app.py                 # Application launcher
├── install_dependencies.bat     # Windows installer
├── README.md                    # This file
├── Images/                      # Employee photos directory
│   ├── employee1/               # Individual employee folders
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── ... (6-8 photos)
│   └── employee2/
├── Encodings/                   # Generated face encodings
├── Attendance/                  # Attendance CSV files
├── vision-pivot-firebase-adminsdk-*.json  # Firebase config (optional)
└── Lawrencium.jpg              # Background image (optional)
```

## 🔥 Firebase Configuration (Optional)

The system works without Firebase, but for cloud features:

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firestore Database and Realtime Database
3. Generate a service account key:
   - Go to Project Settings > Service Accounts
   - Click "Generate new private key"
   - Download the JSON file
4. Rename the downloaded file to match the pattern in the code
5. Place it in the project root directory

## 🎯 Usage Instructions

### First Time Setup
1. Follow the Quick Start Guide above
2. Login with default credentials (test/test)
3. Add your first employee:
   - Go to Admin Panel → Manage Members → Add Member
   - Fill in details and upload 6-8 photos
   - Click "Add Member"

### Daily Usage
1. Start the application
2. Click "Start Attendance System"
3. The system will automatically recognize faces and mark attendance
4. View attendance data in Admin Panel → Manage Data

### Adding New Employees
1. Prepare 6-8 clear photos of the person
2. Create a folder with these photos
3. Go to Admin Panel → Manage Members → Add Member
4. Fill in employee details
5. Click "Upload Image Dataset" and select the photo folder
6. Click "Add Member"

## 🛠️ Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different camera indices in Admin Panel
   - Ensure camera is not being used by another application

2. **Face recognition not working**
   - Ensure face encodings are generated (run `python face_embeddings.py`)
   - Check image quality and lighting
   - Verify employee images are in correct folders
   - Ensure you have 6-8 clear photos per person

3. **Import errors**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility (3.7+)

4. **Application won't start**
   - Check if all dependencies are installed
   - Try running `python start_app.py`
   - Check for error messages in terminal

5. **Firebase connection issues**
   - Verify Firebase configuration file
   - Check internet connection
   - System works without Firebase (optional feature)

### Performance Optimization

- **Use GPU acceleration** if available (CUDA-compatible GPU)
- **Optimize image sizes** (recommended: 300x300 to 500x500 pixels)
- **Limit number of employees** for better performance (recommended: <100 employees)
- **Use good lighting conditions** for better face detection
- **Ensure clear, high-quality photos** for each employee

## 🔒 Security Notes

- **Default credentials**: username: `test`, password: `test`
- **Change default credentials** for production use
- **Secure Firebase configuration file** if using cloud features
- **Comply with local privacy laws** when implementing face recognition
- **Regular data backups** recommended
- **Employee consent** required for face recognition systems

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review error messages in the terminal
3. Ensure all dependencies are properly installed
4. Verify camera permissions and availability

## 📝 License

This project is for educational and commercial use. Please ensure compliance with local privacy laws when implementing face recognition systems.

---

**Note**: This system uses face recognition technology. Please ensure compliance with local privacy laws and obtain necessary permissions before deploying in production environments.