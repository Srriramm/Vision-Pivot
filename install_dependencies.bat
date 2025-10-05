@echo off
echo Installing VisionPivot Attendance System Dependencies...
echo =====================================================

echo.
echo 1. Installing Python packages...
pip install -r requirements.txt

echo.
echo 2. Creating directories...
if not exist "Images" mkdir Images
if not exist "Encodings" mkdir Encodings
if not exist "Attendance" mkdir Attendance

echo.
echo 3. Testing imports...
python test_imports.py

echo.
echo =====================================================
echo Installation completed!
echo.
echo Next steps:
echo 1. Add employee images to the 'Images' folder
echo 2. Run 'python face_embeddings.py' to generate face encodings
echo 3. Run 'python visionpivot_final.py' to start the application
echo.
pause
