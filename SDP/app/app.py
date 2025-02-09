# app/app.py
import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "CHANGE_ME_TO_A_SECURE_RANDOM_STRING"

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.json")
LABELED_FILE = os.path.join(DATA_DIR, "labeled_dataset.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


### User Functions ###

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}  # return empty dict if file does not exist
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


### Dataset & Labeling Functions ###

def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_labels():
    if not os.path.exists(LABELED_FILE):
        return []
    with open(LABELED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_label(entry):
    labels = load_labels()
    labels.append(entry)
    with open(LABELED_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

def get_unlabeled_example(username):
    dataset = load_dataset()
    labels = load_labels()
    # Get all example IDs already labeled by this user
    labeled_ids = {item["example_id"] for item in labels if item.get("username") == username}
    # Return the first example not yet labeled by this user (or None)
    for example in dataset:
        if example["id"] not in labeled_ids:
            return example
    return None

### Routes ###

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("label"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        users = load_users()
        if username in users and users[username]["password"] == password:
            session["username"] = username
            flash("Logged in successfully!")
            return redirect(url_for("label"))
        else:
            error = "Invalid credentials. Please try again."
    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        users = load_users()
        if username in users:
            error = "Username already exists. Please choose a different one."
        else:
            users[username] = {"password": password}
            save_users(users)
            flash("Signup successful! Please log in.")
            return redirect(url_for("login"))
    return render_template("signup.html", error=error)

@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.")
    return redirect(url_for("login"))

@app.route("/label", methods=["GET", "POST"])
def label():
    if "username" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        # When user submits a label, store it.
        example_id = request.form.get("example_id")
        user_label = request.form.get("label")
        entry = {
            "id": str(uuid.uuid4()),
            "example_id": example_id,
            "username": session["username"],
            "label": user_label,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        save_label(entry)
        flash("Label saved successfully!")
        return redirect(url_for("label"))
    
    # GET: show an unlabeled example
    example = get_unlabeled_example(session["username"])
    if not example:
        return render_template("label.html", example=None, message="All examples have been labeled. Thank you!")
    return render_template("label.html", example=example)

if __name__ == "__main__":
    app.run(debug=True)
