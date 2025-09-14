# Gemini Grading App

This repository contains a Flask‑based web application that grades handwritten student solutions to math problems using Google’s **Generative AI** (Gemini) model via the `google‑generativeai` client library.  Students authenticate with their MIT email address and a unique access code, upload images of their work, and the application automatically evaluates the submission against a rubric, calculates a score and presents detailed feedback.

## How it works

### Authentication and uploading

* **User login** – Only registered students can access grading functionality.  The app uses Flask‑Login and checks that the email ends in `@mit.edu` and that the supplied access code matches the entry in `userdatabase.csv`.
* **Image upload** – Students may upload one or more images of their handwritten solution to a problem.  The server temporarily stores the images and passes them through an external OCR/LaTeX conversion service.  A prompt (`conversion_prompt`) instructs the model to extract text and LaTeX from the images (integration with GPT‑4 Vision or another OCR service is required and is *not* included).

### Generative grading pipeline

* **Summary and analysis** – The application builds prompts instructing Gemini to generate (1) a concise summary of the student’s solution and (2) a detailed analysis rating each rubric item as `excellent`, `partial`, `unsatisfactory` or `missing`.  The prompts discourage hallucination and ask the model to base its rating strictly on the uploaded solution.  The official rubric is read from `solution_with_rubric_1.txt`.
* **Fuzzy matching cache** – To ensure consistent grading and reduce API calls, the app stores previous evaluations in a cache.  Before sending a new request to Gemini, the student’s solution summary is normalised and compared to cached summaries using RapidFuzz.  If the similarity exceeds a threshold, the cached grade is reused.
* **Score computation** – The ratings returned by Gemini are mapped to numerical scores and aggregated.  The app computes an overall percentage and categorises performance as *Unsatisfactory*, *Satisfactory* or *Excellent*.  All grades are recorded in `grades.csv`.

### User interface and feedback

* **Result page** – After grading, the app displays a result page showing the overall score, per‑rubric ratings and a performance message.  Students can see the generated summary and analysis and may leave anonymous comments.  Evaluations can also be downloaded for further review.
* **Admin controls** – A simple admin dashboard (protected by the `ADMIN_PASSWORD` environment variable) allows administrators to view submissions, reset attempts for a given problem, and upload new rubric files.

## Installation

1. **Clone the repository** and change into its directory:

   ```bash
   git clone https://github.com/amit12950-cloud/gemini-grading-app.git
   cd gemini-grading-app
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies** using the provided requirements file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Google Cloud authentication**.  The app uses Google’s Generative AI service via Vertex AI.

   * Either run `gcloud auth login` and ensure your default account has access to the Vertex AI API, **or** set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a JSON service account key.
   * Set `GOOGLE_PROJECT_ID` to your Google Cloud project ID.  Optionally set `GOOGLE_REGION` (defaults to `us‑central1`).
   * Provide `APP_SECRET_KEY` for Flask session security and `ADMIN_PASSWORD` for admin routes.

   For example:

   ```bash
   export GOOGLE_PROJECT_ID=<your‑gcp‑project>
   export GOOGLE_REGION=us‑central1
   export APP_SECRET_KEY="<generate‑a‑random‑secret>"
   export ADMIN_PASSWORD="<admin‑password>"
   # If using a service account key:
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

5. **Run the application**:

   ```bash
   python geminiapp.py
   ```

   The development server runs on port 5000.  Navigate to `http://localhost:5000` in your browser and log in using an email/code pair from `userdatabase.csv`.

## Repository structure

- `geminiapp.py` – main Flask server implementing authentication, file uploads, prompts for summary and rubric analysis, fuzzy matching cache and result rendering.  The code reads project ID and region from environment variables and avoids hard‑coded secrets.
- `requirements.txt` – Python packages required to run the Gemini grading app.
- `grades.csv` – persistent storage for grading records; initially empty.
- `solution_with_rubric_1.txt` – official rubric and sample solution for the first problem.  Additional rubric files can be added as `solution_with_rubric_2.txt`, etc.
- `userdatabase.csv` – list of user email addresses and corresponding access codes.
- `templates/` – HTML templates for login, upload, result view, admin dashboard and other pages.

## Notes

- For production use, set a strong `APP_SECRET_KEY`, configure HTTPS and deploy via a production WSGI server such as Gunicorn.

