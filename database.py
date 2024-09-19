import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate(r"face-attendance-system-921a2-firebase-adminsdk-ynaoj-d384c89ae0.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-attendance-system-921a2-default-rtdb.firebaseio.com/"
})

ref = db.reference('Attendance system')

data = {
    "321654":
        {
            "name": "Sriram",
            "Role": "Team Lead",
            "total_attendance": 0,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}
for key, value in data.items():
    ref.child(key).set(value)