# Gemini Grading App

This repository contains a Flask‑based web application that grades student handwritten solutions to math problems using Google’s **Gemini** generative AI model (via the `google‑generativeai` client library).  The app authenticates users via MIT email and a unique access code, converts uploaded images to text, compares the student’s solution against an official rubric using the Gemini model, calculates scores, and stores results.

## Features

* **User authentication** – students sign in with their MIT email and a unique access code stored in `userdatabase.csv`.
* **Handwritten solution upload** – users can upload one or more images of their handwritten solution for each problem.
* **Image to LaTeX conversion** – the uploaded images are sent to a conversion prompt that extracts text and equations; this part requires an external model or API capable of OCR and LaTeX generation (not included in this repo).
* **Gemini‑based grading** – the student’s solution is compared against the official rubric using Google’s Gemini generative model.  The code leverages the `google‑generativeai` library to generate evaluations and applies heuristics based on cosine similarity and approximate caching to produce consistent grading.
* **Result reporting** – the app displays detailed scoring feedback and stores grades in `grades.csv` for record keeping.

## Installation

1. **Clone the repository** (or download the source code) and navigate into it:

   ```bash
   git clone https://github.com/amit12950-cloud/gemini-grading-app.git
   cd gemini-grading-app
   ```

2. **Create and activate a Python virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies** using the provided requirements file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Google Cloud authentication**.  The application uses Google’s Generative AI service via Vertex AI.  Before running the app you must:
   * Authenticate to Google Cloud via `gcloud auth login` or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a JSON service account key.
   * Set `GOOGLE_PROJECT_ID` to your Google Cloud project ID.
   * Optionally set `GOOGLE_REGION` (defaults to `us‑central1`).

5. **Run the application**:

   ```bash
   python geminiapp.py
   ```

   By default the Flask development server will run on port 5000.  Open `http://localhost:5000` in your web browser and log in with one of the email/code pairs from `userdatabase.csv`.

## Repository structure

* `geminiapp.py` – main Flask server for the Gemini grading app.  This version has been sanitised to avoid hard‑coded project IDs and reads the Google project ID and region from environment variables.
* `requirements.txt` – lists the Python dependencies required to run the project.
* `grades.csv` – CSV file used to store grading records; initially empty.
* `solution_with_rubric_1.txt` – official rubric used to evaluate the first problem (additional rubric files can be added as `solution_with_rubric_2.txt`, etc.).
* `userdatabase.csv` – contains user email addresses and corresponding access codes.
* `templates/` – HTML templates for the user interface.

## Notes

* The OCR/LaTeX conversion step for handwritten images is **not** implemented in this repository.  You will need to integrate an external service or model (such as a hosted GPT‑4 Vision API) to convert images into LaTeX and plain text.
* For production deployment, ensure you set a secure `app.secret_key` in `geminiapp.py` and configure HTTPS.  This project is provided for educational purposes and should be adapted and secured before real use.