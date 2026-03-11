from flask import Flask, request, render_template
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def analyze_resume(resume, jd):

    text = [resume, jd]

    cv = CountVectorizer().fit_transform(text)
    similarity = cosine_similarity(cv)

    match_score = round(similarity[0][1] * 100, 2)

    resume_words = set(resume.lower().split())
    jd_words = set(jd.lower().split())

    missing_skills = jd_words - resume_words

    result = f"Match Score: {match_score}%\n\n"
    result += "Some Missing Keywords:\n"
    result += ", ".join(list(missing_skills)[:20])

    return result


@app.route("/", methods=["GET", "POST"])
def index():

    result = ""

    if request.method == "POST":

        file = request.files["resume"]
        jd = request.form["jd"]

        resume_text = extract_text(file)

        result = analyze_resume(resume_text, jd)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)