from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import pickle
import numpy as np
import io
import base64
from datetime import datetime
from PIL import Image

app = Flask(__name__)
app.secret_key = "secret123"

# Load ML model
model = pickle.load(open("saved_models/model.pkl", "rb"))
selector = pickle.load(open("saved_models/selector.pkl", "rb"))

# Create DB
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    conn.commit()
    conn.close()

init_db()

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = u
            return redirect("/home")
        else:
            return "Invalid login"

    return render_template("login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        conn.close()

        return redirect("/")

    return render_template("register.html")

# ---------------- DASHBOARD ----------------
@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":

        # Basic Info
        name = request.form["student_name"]
        student_class = request.form["student_class"]
        syllabus = request.form["syllabus"]
        exam_date = request.form["exam_date"]

        # Inputs
        study = float(request.form["study"])
        attendance = float(request.form["attendance"])
        sleep = float(request.form["sleep"])
        marks = float(request.form["marks"])
        guidance = float(request.form["guidance"])
        internet = float(request.form["internet"])
        activities = float(request.form["activities"])

        # Convert syllabus fraction
        try:
            completed, total = syllabus.split("/")
            syllabus_ratio = float(completed) / float(total)
        except:
            syllabus_ratio = 0

        # Days left calculation
        today = datetime.today()
        exam = datetime.strptime(exam_date, "%Y-%m-%d")
        days_left = (exam - today).days

        # Model prediction
        features = np.array([[study, attendance, sleep, marks, guidance, internet, activities]])
        features = selector.transform(features)
        pred = model.predict(features)[0]

        # Score calculation
        score = min((study*5 + attendance*0.5 + sleep*3 + marks*0.5), 100)

        # Suggestions
        suggestions = []

        if study < 3:
            suggestions.append("Increase study hours")

        if attendance < 75:
            suggestions.append("Improve attendance")

        if sleep < 6:
            suggestions.append("Improve sleep schedule")

        if syllabus_ratio < 0.5 and days_left < 15:
            suggestions.append("⚠️ Urgent: Low syllabus & exam near")

        if days_left < 7:
            suggestions.append("Focus on revision and mock tests")

        return render_template(
            "index.html",
            name=name,
            student_class=student_class,
            syllabus=syllabus,
            exam_date=exam_date,
            days_left=days_left,
            result=pred,
            score=score,
            study=study,
            sleep=sleep,
            suggestions=suggestions
        )

    return render_template("index.html")

# ---------------- PDF DOWNLOAD ----------------
@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()
    image_data = data["image"].split(",")[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    pdf_buffer = io.BytesIO()
    image.save(pdf_buffer, format="PDF")
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name="report.pdf")

# ---------------- RUN APP ----------------
import os
if __name__=="__main__":
    port=int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)
    
