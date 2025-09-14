from flask import Flask, request, render_template, redirect, url_for, flash, Response, session
from google import genai
from google.genai import types
import base64
import os
import re
import csv
import datetime
import uuid
import numpy as np
import sys
import glob
import random
import json
from rapidfuzz import fuzz
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

# Imports for Flask-Login and Session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_session import Session

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Configure server-side session storage (using filesystem)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Setup Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Initialize the Gemini client
#
# NOTE: Do not hard‑code your Google Cloud project identifiers directly in source code.
# For safety the project ID and region are read from environment variables.  When
# deploying this application you should set the ``GOOGLE_PROJECT_ID`` and
# ``GOOGLE_REGION`` environment variables to match your Google Cloud project and
# preferred region.  Defaults below can be replaced with placeholders as needed.
project_id = os.environ.get("GOOGLE_PROJECT_ID", "your‑project‑id")
location = os.environ.get("GOOGLE_REGION", "us‑central1")
client = genai.Client(
    vertexai=True,
    project=project_id,
    location=location,
)

# Model and config setup
model = "gemini-2.0-flash-lite-001"
generate_content_config = types.GenerateContentConfig(
    temperature=0,
    top_p=1,
    top_k=1,
    max_output_tokens=8192,
    response_modalities=["TEXT"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
)

def load_users(csv_filename):
    """Load users from a CSV file."""
    users = {}
    with open(csv_filename, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normalized = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}
            email = normalized.get("mit email")
            unique_code = normalized.get("unique code")
            if email and unique_code:
                users[email.lower()] = {"access_code": unique_code}
    return users

CSV_FILENAME = os.path.join(os.path.dirname(__file__), "userdatabase.csv")
users_db = load_users(CSV_FILENAME)

# Global cache for grades.csv records
grades_cache = None
def load_grades_cache():
    global grades_cache
    if os.path.exists("grades.csv"):
        with open("grades.csv", "r", newline="") as f:
            reader = csv.reader(f)
            grades_cache = list(reader)
    else:
        grades_cache = []
def get_grades_cache():
    global grades_cache
    if grades_cache is None:
        load_grades_cache()
    return grades_cache
def update_grades_cache():
    load_grades_cache()

class User(UserMixin):
    def __init__(self, email):
        self.id = email
@login_manager.user_loader
def load_user(user_id):
    if user_id in users_db:
        return User(email=user_id)
    return None

def count_attempts(user_email, problem_number):
    count = 0
    records = get_grades_cache()
    for row in records:
        if row and row[1].strip().lower() == user_email.lower() and int(row[6]) == problem_number:
            count += 1
    return count

def get_total_problems():
    rubric_pattern = os.path.join(os.path.dirname(__file__), "solution_with_rubric_*.txt")
    files = glob.glob(rubric_pattern)
    return len(files)

# --- Global Parameters ---
CORRECTNESS_THRESHOLD = 0.3
UNCERTAINTY_THRESHOLD = 0.3
GRADE_BOOLEAN_ACCEPTANCE = True
EVALUATION_RUNS = 1  # Single run as per original setup

evaluation_cache = {}
cache_lock = threading.Lock()

def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())

def get_approx_cached_evaluation(prompt: str, similarity_threshold: float = 95) -> str:
    normalized_prompt = normalize_prompt(prompt)
    with cache_lock:
        for cached_prompt, response in evaluation_cache.items():
            if fuzz.ratio(normalized_prompt, cached_prompt) >= similarity_threshold:
                return response
    return None

def store_approx_cached_evaluation(prompt: str, response: str):
    normalized_prompt = normalize_prompt(prompt)
    with cache_lock:
        evaluation_cache[normalized_prompt] = response

@app.template_filter('split_str')
def split_str(value, delimiter):
    try:
        return value.split(delimiter)
    except Exception:
        return []

def get_latest_record(user_email, problem_number):
    records = get_grades_cache()
    latest_record = None
    latest_time = None
    for row in records:
        if row and row[1].strip().lower() == user_email.lower() and int(row[6]) == problem_number:
            try:
                t = datetime.datetime.fromisoformat(row[5])
                if latest_time is None or t > latest_time:
                    latest_time = t
                    latest_record = row
            except Exception:
                continue
    return latest_record

@lru_cache(maxsize=None)
def get_rubric(problem_number):
    rubric_file = os.path.join(os.path.dirname(__file__), f"solution_with_rubric_{problem_number}.txt")
    try:
        with open(rubric_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def get_device_flags():
    is_mobile = "iphone" in request.user_agent.string.lower()
    accessibility = request.args.get("accessibility", "0") == "1"
    return is_mobile, accessibility

@app.route("/login", methods=["GET", "POST"])
def login():
    is_mobile, accessibility = get_device_flags()
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        code = request.form.get("code").strip()
        if not email.endswith("@mit.edu"):
            flash("Email must be an MIT email address.")
            return redirect(url_for("login"))
        user_data = users_db.get(email)
        if not user_data:
            flash("Email not found. Please use your MIT email.")
            return redirect(url_for("login"))
        if user_data["access_code"] == code:
            user = User(email=email)
            login_user(user)
            return redirect(url_for("welcome"))
        else:
            flash("Incorrect access code.")
            return redirect(url_for("login"))
    return render_template("login.html", is_mobile=is_mobile, accessibility=accessibility)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

def call_gemini_pixtral(content, temperature=0.0, max_tokens=8192):
    """Call the Gemini API for multimodal or text inputs."""
    try:
        response_stream = client.models.generate_content_stream(
            model=model,
            contents=content,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=1,
                top_k=1,
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
            ),
        )
        result = ''.join(chunk.text for chunk in response_stream)
        return result
    except Exception as e:
        logging.error(f"Error in Gemini call: {e}")
        return ""

def get_gemini_evaluation(prompt: str, temperature: float = 0.0) -> str:
    """Get evaluation for text-only prompts with caching."""
    cached_response = get_approx_cached_evaluation(prompt)
    if cached_response:
        return cached_response
    content = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    result = call_gemini_pixtral(content, temperature=temperature)
    store_approx_cached_evaluation(prompt, result)
    return result

def process_text(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

# Utility Functions
def parse_rubric_items(rubric_text):
    pattern = (
        r"Rubric Item\s*\d+:\s*(.*?)\s*\((\d+)\s*pts?\).*?"
        r"Partial\s*\((\d+)\s*pts?\)"
    )
    items = []
    for m in re.finditer(pattern, rubric_text, flags=re.DOTALL|re.IGNORECASE):
        name = m.group(1).strip()
        max_pts = int(m.group(2))
        partial_pts = int(m.group(3))
        items.append({"item": name, "max": max_pts, "partial": partial_pts})
    return items

def score_from_status(status: str, max_pts: int, partial_pts: int) -> int:
    m = re.search(r"\b(excellent|partial|unsatisfactory|missing)\b", status, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized status '{status}'")
    key = m.group(1).lower()
    return {'excellent': max_pts, 'partial': partial_pts, 'unsatisfactory': 0, 'missing': 0}[key]

def clean_json(raw_text):
    """Clean and fix JSON string, handling LaTeX escape sequences."""
    txt = raw_text.strip()
    # Remove code fences
    if txt.startswith("```"):
        m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", txt, flags=re.DOTALL)
        if m:
            txt = m.group(1).strip()
    elif not txt.startswith(("{", "[")):
        m = re.search(r"(\{.*\}|\[.*\])", txt, flags=re.DOTALL)
        if m:
            txt = m.group(1).strip()
    
    # Double all backslashes to ensure proper JSON escaping
    txt = txt.replace('\\', '\\\\')
    
    logging.debug(f"Cleaned JSON string: {txt}")
    return txt

def parse_raw_analysis_fallback(raw_text):
    """Fallback to extract graded items from raw text if JSON parsing fails."""
    graded_items = []
    # Extract JSON-like content between ```json ... ```
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, flags=re.DOTALL)
    if not m:
        logging.warning("No JSON-like content found in raw analysis for fallback.")
        return graded_items
    
    content = m.group(1)
    # Split into individual items
    item_pattern = r'\{\s*"item":\s*"([^"]+)",\s*"status":\s*"([^"]+)",\s*"justification":\s*"([^"]+)"\s*\}'
    for match in re.finditer(item_pattern, content, flags=re.DOTALL):
        item = match.group(1).strip()
        status = match.group(2).strip()
        justification = match.group(3).strip()
        # Remove LaTeX for safety
        justification = re.sub(r'\\[\[\(].*?\\[\]\)]', '', justification)
        graded_items.append({
            "item": item,
            "status": status,
            "justification": justification
        })
    
    logging.debug(f"Fallback parsed items: {graded_items}")
    return graded_items

def normalize_item_name(item_name: str) -> str:
    """Remove 'Rubric Item X:' prefix and normalize for matching."""
    item_name = re.sub(r"Rubric Item \d+:\s*", "", item_name, flags=re.IGNORECASE).strip()
    return item_name.lower()

def make_summary_prompt(solution):
    return json.dumps({
        "task": "summary",
        "instr": (
            "Read the following full student's solution and provide a concise summary explaining what the student is doing, including key equations and reasoning. Do not omit any important steps."
            "**Important: Do NOT hallucinate.** Only evaluate steps the student actually wrote—do NOT infer, assume, or invent any reasoning or calculations."
        ),
        "student_solution": solution
    })

def make_analysis_prompt(rubric, solution, summary):
    return json.dumps({
        "task": "analysis",
        "instr": (
            "You are a physics grader. Using the official rubric, the student's full solution, and provided summary, produce a bullet-point analysis for each rubric item. "
            "For each item, clearly state whether the student's solution includes the required information. If an item is completely missing required content, state 'Missing' for that item. Otherwise, provide a concise 2-3 sentence justification."
            "Return a JSON array of {\"item\": string, \"status\": string, \"justification\": string}, where status is "
            "\"Excellent\", \"Partial\", \"Unsatisfactory\", or \"Missing\", and justification explains the evaluation."
            "Use exact rubric item names from the provided rubric."
        ),
        "rubric": rubric,
        "student_solution": solution,
        "summary": summary
    })

@app.route("/")
@login_required
def welcome():
    is_mobile, accessibility = get_device_flags()
    welcome_file = os.path.join(os.path.dirname(__file__), "welcome.txt")
    try:
        with open(welcome_file, "r") as f:
            welcome_message = f.read()
    except FileNotFoundError:
        welcome_message = "Welcome! Please proceed with your submission."
    return render_template("welcome.html", welcome_message=welcome_message, email=current_user.id,
                           is_mobile=is_mobile, accessibility=accessibility)

@app.route("/problem/<int:problem_number>", methods=["GET", "POST"])
@login_required
def problem(problem_number):
    is_mobile, accessibility = get_device_flags()
    total_problems = get_total_problems()
    if problem_number < 1 or problem_number > total_problems:
        return "Invalid problem number.", 400

    if request.method == "GET":
        if count_attempts(current_user.id, problem_number) > 0:
            update_grades_cache()
            records = get_grades_cache()
            record = get_latest_record(current_user.id, problem_number)
            if record:
                try:
                    earned_total, max_total = map(int, record[2].split('/'))
                except Exception:
                    earned_total, max_total = 0, 0
                final_score_eval = float(record[9])
                percentage = final_score_eval
                if percentage <= 40:
                    performance_message = "unsatisfactory performance"
                elif percentage <= 80:
                    performance_message = "satisfactory performance"
                else:
                    performance_message = "excellent work"
                # Retrieve approved_text from grades.csv (column 8)
                approved_text = record[8] if len(record) > 8 and record[8].strip() else "No approved conversion available"
                logging.debug(f"Retrieved approved_text for problem {problem_number}: {approved_text[:100]}...")
                return render_template("result.html",
                                       performance_message=performance_message,
                                       detailed_evaluation=record[4],
                                       student_evaluation=record[4],
                                       final_score_skeleton=record[7] if len(record) > 7 else "N/A",
                                       handwritten_explanation=approved_text,
                                       handwritten_images=record[3].split("||"),
                                       problem_number=problem_number,
                                       earned_total=earned_total,
                                       max_total=max_total,
                                       total_problems=total_problems,
                                       final_score_eval=final_score_eval,
                                       submission_id=record[0],
                                       timestamp=record[5],
                                       is_mobile=is_mobile, accessibility=accessibility)
        else:
            rubric_text = get_rubric(problem_number)
            problem_name = f"Problem {problem_number}"
            if rubric_text:
                for line in rubric_text.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not all(ch == '-' for ch in stripped_line):
                        problem_name = stripped_line
                        break
            return render_template("index.html", problem_number=problem_number, problem_name=problem_name,
                                   attempts_exceeded=False, total_problems=total_problems,
                                   is_mobile=is_mobile, accessibility=accessibility)

    if request.method == "POST":
        if count_attempts(current_user.id, problem_number) > 0:
            flash("You have already submitted an answer for this problem.")
            return redirect(url_for("problem", problem_number=problem_number))
        files = request.files.getlist("handwritten_image")
        if not files or len(files) == 0:
            flash("No file part")
            return redirect(url_for("problem", problem_number=problem_number))

        base64_images = []
        for file in files:
            if file.filename == "":
                continue
            data = file.read()
            base64_images.append(base64.b64encode(data).decode("utf-8"))
        if not base64_images:
            flash("No valid images uploaded.")
            return redirect(url_for("problem", problem_number=problem_number))

        submission_id = str(uuid.uuid4())
        official_explanation = get_rubric(problem_number)
        if official_explanation is None:
            flash(f"Rubric file for problem {problem_number} not found.")
            return redirect(url_for("problem", problem_number=problem_number))

        conversion_prompt = (
            "Your task is to extract the mathematical content and texts from the attached handwritten solution images. "
            "Convert all equations, expressions, and symbols into LaTeX format, and wrap each one individually using MathJax display math delimiters: \\[ ... \\]. "
            "Return all content, including explanatory text, diagrams (as text description if needed), and derivations. "
            "Ensure all mathematical content and texts are captured in full detail, including steps, derivations, and final results. "
            "Match and format the handwritten solution's math and texts accordingly, without omitting any parts."
        )

        parts = []
        for img in base64_images:
            content = [
                types.Content(role="user", parts=[
                    types.Part(text=conversion_prompt),
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img))
                ])
            ]
            part = call_gemini_pixtral(content, temperature=0.0, max_tokens=8192)
            parts.append(part)
        handwritten_explanation = "\n\n".join(parts)

        session['submission_id'] = submission_id
        session['base64_images'] = base64_images
        session['handwritten_explanation'] = handwritten_explanation
        session['problem_number'] = problem_number

        return render_template("conversion_approval.html",
                               conversion_text=handwritten_explanation,
                               base64_images=base64_images,
                               problem_number=problem_number,
                               total_problems=total_problems,
                               is_mobile=is_mobile,
                               accessibility=accessibility)

@app.route("/finalize_submission", methods=["POST"])
@login_required
def finalize_submission():
    is_mobile, accessibility = get_device_flags()
    submission_id = session.get('submission_id')
    base64_images = session.get('base64_images', [])
    handwritten_explanation = session.get('handwritten_explanation', '')
    problem_number = session.get('problem_number')
    total_problems = get_total_problems()

    if not submission_id or problem_number is None:
        flash("Session expired or invalid submission.")
        return redirect(url_for("welcome"))

    approved_text = request.form.get("approved_conversion_text", handwritten_explanation)
    approved_text = approved_text.strip()
    if not approved_text:
        flash("No solution text to evaluate.")
        return redirect(url_for("problem", problem_number=problem_number))

    official_explanation = get_rubric(problem_number)
    if official_explanation is None:
        flash(f"Rubric file for problem {problem_number} not found.")
        return redirect(url_for("problem", problem_number=problem_number))

    summary_prompt = make_summary_prompt(approved_text)
    summary = get_gemini_evaluation(summary_prompt)

    analysis_prompt = make_analysis_prompt(official_explanation, approved_text, summary)
    raw_analysis = get_gemini_evaluation(analysis_prompt)
    logging.debug(f"Raw analysis: {raw_analysis}")
    try:
        cleaned_json = clean_json(raw_analysis)
        graded_items = json.loads(cleaned_json)
    except Exception as e:
        logging.error(f"JSON parsing failed: {e}\nRaw analysis:\n{raw_analysis}")
        # Fallback to parsing raw text
        graded_items = parse_raw_analysis_fallback(raw_analysis)
        if not graded_items:
            logging.warning("Fallback parsing failed, using empty graded items.")
            graded_items = []

    rubric_items = parse_rubric_items(official_explanation)
    logging.debug(f"Rubric items: {[item['item'] for item in rubric_items]}")
    logging.debug(f"Graded items: {[item['item'] for item in graded_items]}")

    total_score = 0
    max_total = 0
    final_score_skeleton = []
    detailed_evaluation_lines = []

    # Normalize and match graded items to rubric items
    graded_items_normalized = [
        {
            "item": normalize_item_name(item["item"]),
            "status": item.get("status", "Missing"),
            "justification": item.get("justification", "No justification provided.")
        }
        for item in graded_items
    ]
    rubric_items_normalized = [(normalize_item_name(item["item"]), item) for item in rubric_items]

    # Match graded items to rubric items using fuzzy matching
    matched_items = []
    for rubric_norm, rubric_item in rubric_items_normalized:
        best_match = None
        best_score = 0
        for graded_item in graded_items_normalized:
            similarity = fuzz.ratio(rubric_norm, graded_item["item"])
            if similarity > best_score and similarity > 80:  # Threshold for matching
                best_score = similarity
                best_match = graded_item
        if best_match:
            matched_items.append((rubric_item, best_match))
            graded_items_normalized.remove(best_match)
        else:
            matched_items.append((rubric_item, None))

    for rubric_item, graded_item in matched_items:
        item_name = rubric_item["item"]
        max_pts = rubric_item["max"]
        partial_pts = rubric_item["partial"]
        if graded_item:
            status = graded_item["status"]
            justification = graded_item["justification"]
            try:
                score = score_from_status(status, max_pts, partial_pts)
            except ValueError:
                logging.warning(f"Invalid status '{status}' for item '{item_name}'")
                score = 0
                status = "Unsatisfactory"
                justification = "Invalid status received from evaluation."
        else:
            status = "Unsatisfactory"  # Use Unsatisfactory instead of Missing to avoid 0/100 unless explicitly indicated
            score = 0
            justification = "Rubric item not addressed or evaluation failed."
        total_score += score
        max_total += max_pts
        final_score_skeleton.append(f"{item_name}: {score}/{max_pts}")
        detailed_evaluation_lines.append(
            f"- {item_name} ({status}): {score}/{max_pts}\n  Justification: {justification}"
        )

    detailed_evaluation = "\n".join(detailed_evaluation_lines)
    earned_total = total_score
    final_score_eval = (total_score / max_total * 100) if max_total > 0 else 0

    if final_score_eval <= 40:
        performance_message = "unsatisfactory performance"
    elif final_score_eval <= 80:
        performance_message = "satisfactory performance"
    else:
        performance_message = "excellent work"

    timestamp = datetime.datetime.now().isoformat()
    image_urls = "||".join([f"data:image/jpeg;base64,{img}" for img in base64_images])
    final_score_skeleton_str = "\n".join(final_score_skeleton)

    with open("grades.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            submission_id, current_user.id, f"{total_score}/{max_total}",
            image_urls, detailed_evaluation, timestamp, problem_number,
            final_score_skeleton_str, approved_text, f"{final_score_eval:.2f}"
        ])
    update_grades_cache()

    session.pop('submission_id', None)
    session.pop('base64_images', None)
    session.pop('handwritten_explanation', None)
    session.pop('problem_number', None)

    return render_template("result.html",
                           performance_message=performance_message,
                           detailed_evaluation=detailed_evaluation,
                           student_evaluation=detailed_evaluation,
                           final_score_skeleton=final_score_skeleton_str,
                           handwritten_explanation=approved_text,
                           handwritten_images=[f"data:image/jpeg;base64,{img}" for img in base64_images],
                           problem_number=problem_number,
                           earned_total=earned_total,
                           max_total=max_total,
                           total_problems=total_problems,
                           final_score_eval=final_score_eval,
                           submission_id=submission_id,
                           timestamp=timestamp,
                           is_mobile=is_mobile, accessibility=accessibility)

@app.route("/reset_attempts/<int:problem_number>", methods=["POST"])
@login_required
def reset_attempts(problem_number):
    is_mobile, accessibility = get_device_flags()
    admin_password = request.form.get("admin_password", "")
    if admin_password != "tiwari123":
        flash("Unauthorized: Incorrect admin password.")
        return redirect(url_for("problem", problem_number=problem_number))
    student_email = request.form.get("student_email", "").strip().lower()
    if not student_email:
        flash("Student email is required.")
        return redirect(url_for("admin_controls"))
    try:
        if os.path.exists("grades.csv"):
            with open("grades.csv", "r", newline="") as f:
                reader = csv.reader(f)
                records = list(reader)
            filtered_records = [
                record for record in records
                if not (record[1].strip().lower() == student_email and int(record[6]) == problem_number)
            ]
            with open("grades.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(filtered_records)
        flash(f"Submission attempts for student {student_email} on Problem {problem_number} have been reset.")
    except Exception as e:
        logging.error(f"Error resetting attempts: {e}")
        flash("An error occurred while resetting the attempts.")
    update_grades_cache()
    return redirect(url_for("admin_controls"))

@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin_controls():
    is_mobile, accessibility = get_device_flags()
    if request.method == "POST":
        admin_password = request.form.get("admin_password", "")
        if admin_password != "tiwari123":
            flash("Incorrect admin password.")
            return redirect(url_for("admin_controls"))
        return render_template("admin_controls.html", is_mobile=is_mobile, accessibility=accessibility)
    return render_template("admin_login.html", is_mobile=is_mobile, accessibility=accessibility)

@app.route("/upload_rubrics", methods=["POST"])
@login_required
def upload_rubrics():
    is_mobile, accessibility = get_device_flags()
    admin_password = request.form.get("admin_password", "")
    if admin_password != "tiwari123":
        flash("Unauthorized: Incorrect admin password.")
        return redirect(url_for("admin_controls"))
    files = request.files.getlist("rubric_files")
    if not files:
        flash("No files uploaded.")
        return redirect(url_for("admin_controls"))
    for file in files:
        filename = file.filename
        if not filename.endswith(".txt"):
            continue
        save_path = os.path.join(os.path.dirname(__file__), filename)
        file.save(save_path)
        problem_number = re.findall(r'\d+', filename)
        if problem_number:
            get_rubric.cache_clear()
    flash("Rubrics uploaded successfully.")
    return redirect(url_for("admin_controls"))

@app.route("/thankyou")
def thankyou():
    is_mobile, accessibility = get_device_flags()
    message_file = os.path.join(os.path.dirname(__file__), "message.txt")
    try:
        with open(message_file, "r") as f:
            thank_you_message = f.read()
    except FileNotFoundError:
        thank_you_message = "Thank you for your submission!"
    return render_template("thankyou.html", thank_you_message=thank_you_message,
                           is_mobile=is_mobile, accessibility=accessibility)

@app.route("/save_comment", methods=["POST"])
@login_required
def save_comment():
    is_mobile, accessibility = get_device_flags()
    submission_id = request.form.get("submission_id")
    comment_text = request.form.get("comment")
    if not submission_id or not comment_text:
        flash("Submission ID or comment is missing.")
        return redirect(request.referrer)
    comment_file = "comments.csv"
    timestamp = datetime.datetime.now().isoformat()
    with open(comment_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([submission_id, current_user.id, comment_text, timestamp])
    flash("Comment saved successfully.")
    return redirect(request.referrer)

@app.route("/download_evaluation/<submission_id>")
@login_required
def download_evaluation(submission_id):
    evaluation_record = None
    if os.path.exists("grades.csv"):
        with open("grades.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == submission_id and row[1].strip().lower() == current_user.id.lower():
                    evaluation_record = row
                    break
    if not evaluation_record:
        flash("Evaluation record not found.")
        return redirect(url_for("welcome"))
    details = f"Submission ID: {evaluation_record[0]}\n"
    details += f"Student Email: {evaluation_record[1]}\n"
    details += f"Score: {evaluation_record[2]}\n"
    details += f"Timestamp: {evaluation_record[5]}\n"
    details += f"Problem Number: {evaluation_record[6]}\n\n"
    details += "Detailed Evaluation:\n"
    details += f"{evaluation_record[4]}\n\n"
    details += "Final Scores:\n"
    details += f"AI Evaluation Score: {evaluation_record[9]}\n"
    comment = ""
    if os.path.exists("comments.csv"):
        with open("comments.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == submission_id and row[1].strip().lower() == current_user.id.lower():
                    comment = row[2]
                    break
    if comment:
        details += f"\nComment: {comment}\n"
    return Response(details, mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment;filename=evaluation_{submission_id}.txt"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
