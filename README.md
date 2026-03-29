# 🎓 Smart Attendance System

A **face-recognition-based attendance management system** built with Flask, FaceNet, MediaPipe, and PyTorch. It supports real-time multi-face detection, liveness checks, and role-based dashboards for faculty and students.

---

## ✨ Features

- 🔍 **Real-time Face Recognition** – Powered by FaceNet (InceptionResnetV1) + MTCNN for accurate, fast identification
- 👁️ **Liveness Detection** – Uses MediaPipe Face Mesh with Eye Aspect Ratio (EAR) to prevent photo spoofing
- 👥 **Multi-face Detection** – Marks attendance for multiple students in a single frame simultaneously
- 🏫 **Role-based Dashboards** – Separate interfaces for Faculty and Students
- 📊 **Attendance Reports** – Filter by year, branch, section, subject, period, and date
- 📥 **Excel Export** – Download attendance reports as `.xlsx` files
- 🔒 **IN/OUT Tracking** – Enforces a minimum 5-minute gap between IN and OUT marking
- ✏️ **Student Management** – Register, edit, and delete student profiles
- 🌐 **Web-based Interface** – Access from any device on the same network

---

## 🛠️ Tech Stack

| Layer       | Technology                                      |
|-------------|-------------------------------------------------|
| Backend     | Python 3.10, Flask 2.2.5                        |
| ML / Vision | PyTorch 2.0.1, FaceNet-PyTorch, MediaPipe 0.10.9 |
| Database    | SQLite3                                         |
| Frontend    | HTML, CSS, JavaScript (Vanilla)                 |
| Data        | Pandas, NumPy, OpenCV (headless)                |
| Deployment  | Gunicorn                                        |

---

## 📁 Project Structure

```
Smart-Attendance/
├── server.py               # Main Flask application & all API routes
├── main.py                 # Entry point / standalone runner
├── student_db.py           # Database utility functions
├── student_register.py     # Face registration helper
├── register_face.py        # CLI face registration script
├── delete_face.py          # CLI face deletion script
├── export_attendance.py    # Attendance export utility
├── embedding_accuracy.py   # Embedding quality check
├── check_db.py             # Database inspection tool
├── check_col.py            # Column checker utility
├── reset_db.py             # Database reset script
├── update_db.py            # Database migration helper
├── final_db_fix.py         # DB schema fix utility
├── gunicorn.conf.py        # Gunicorn configuration
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── data/
│   └── attendance1.db      # SQLite database (auto-created)
├── templates/              # HTML templates (Jinja2)
│   ├── login.html
│   ├── faculty_dashboard.html
│   ├── student_dashboard.html
│   ├── register.html
│   ├── manage_students.html
│   ├── reports.html
│   ├── secure_in.html
│   ├── secure_out.html
│   └── edit_student_profile.html
├── static/                 # CSS, JS, and static assets
└── venv_new/               # Recommended virtual environment
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- A working webcam
- macOS / Linux (Windows may need additional setup for MediaPipe)

### 1. Clone the Repository
```bash
git clone https://github.com/Hemalatha150/Smart-Attendance.git
cd Smart-Attendance
```

### 2. Create & Activate a Virtual Environment
```bash
python3.10 -m venv venv_new
source venv_new/bin/activate        # macOS / Linux
# venv_new\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is installed using the CPU-only index URL specified in `requirements.txt` to reduce memory usage.

---

## 🚀 Running the Application

```bash
source venv_new/bin/activate
python3 server.py
```

The server starts on:
- **Local:** http://127.0.0.1:5000
- **Network:** http://\<your-ip\>:5000

---

## 🔐 Login Credentials

### Faculty (Admin)
| Field    | Value       |
|----------|-------------|
| Username | `ADMIN`     |
| Password | `ADMIN123`  |

### Student
| Field    | Value                        |
|----------|------------------------------|
| Username | Student's Register Number    |
| Password | Same as Register Number      |

---

## 📋 How It Works

### 1. Student Registration (Faculty)
1. Log in as Faculty → **Register Student**
2. Enter student details (Name, Register No., Phone, Branch, Section, Year)
3. System captures **20 face embeddings** using FaceNet
4. Averaged embedding is stored in the SQLite database

### 2. Marking Attendance
1. Faculty navigates to **Mark IN** or **Mark OUT**
2. Faculty starts the attendance session
3. Students stand in front of the camera — the system detects and recognizes faces automatically
4. For **OUT**, a minimum gap of **5 minutes** from IN time is enforced

### 3. Reports
1. Faculty goes to **Attendance Reports**
2. Filters by Year, Branch, Section, Subject, Period, and Date
3. View Present/Absent summary and download as Excel

---

## 🗄️ Database Schema

### `faces` table
| Column    | Type  | Description                        |
|-----------|-------|------------------------------------|
| reg_no    | TEXT  | Primary key – Register Number      |
| name      | TEXT  | Student name                       |
| phone     | TEXT  | Contact number                     |
| branch    | TEXT  | Department (e.g., CSE, ECE)        |
| section   | TEXT  | Section (e.g., A, B)               |
| year      | TEXT  | Academic year (e.g., 1, 2, 3, 4)   |
| embedding | BLOB  | 512-dim FaceNet face embedding      |

### `attendance` table
| Column  | Type    | Description                          |
|---------|---------|--------------------------------------|
| id      | INTEGER | Auto-increment primary key           |
| reg_no  | TEXT    | Student Register Number              |
| name    | TEXT    | Student Name                         |
| branch  | TEXT    | Branch                               |
| section | TEXT    | Section                              |
| year    | TEXT    | Year                                 |
| subject | TEXT    | Subject name                         |
| date    | TEXT    | Date (YYYY-MM-DD)                    |
| time    | TEXT    | Time (HH:MM:SS)                      |
| status  | TEXT    | `IN` or `OUT`                        |

---

## 🌐 API Endpoints

| Method | Endpoint                          | Description                        |
|--------|-----------------------------------|------------------------------------|
| POST   | `/api/login`                      | User login                         |
| POST   | `/api/check_id`                   | Check if student is registered     |
| POST   | `/api/process_web_pose`           | Capture face & register student    |
| POST   | `/api/mark_attendance`            | Mark IN/OUT attendance             |
| GET    | `/api/toggle_attendance/<mode>/<action>` | Start/stop attendance session |
| POST   | `/api/update_student`             | Update student info                |
| DELETE | `/api/delete_student/<reg_no>`    | Delete a student                   |
| GET    | `/download_report`                | Download attendance as Excel       |
| GET    | `/warmup`                         | Warm up ML models                  |

---

## 🚢 Production Deployment (Gunicorn)

```bash
source venv_new/bin/activate
gunicorn -c gunicorn.conf.py server:app
```

---

## 📦 Key Dependencies

| Package              | Version   | Purpose                         |
|----------------------|-----------|---------------------------------|
| flask                | 2.2.5     | Web framework                   |
| torch                | 2.0.1     | Deep learning (CPU)             |
| facenet-pytorch      | 2.5.2     | Face detection & recognition    |
| mediapipe            | 0.10.9    | Face mesh & liveness detection  |
| opencv-python-headless | 4.8.0.76 | Image processing               |
| numpy                | 1.24.4    | Numerical computations          |
| pandas               | 1.5.3     | Data manipulation & Excel export |
| scikit-learn         | 1.3.2     | ML utilities                    |
| Pillow               | latest    | Image handling                  |
| gunicorn             | latest    | Production WSGI server          |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Built with ❤️ using Flask + FaceNet + MediaPipe
