from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import re

app = Flask(__name__)
app.secret_key = "secret123"

DATA_FILE = "counseling_data.json"

# 💡 감정 분석 모델 로드
model_name = "alsgyu/sentiment-analysis-fine-tuned-model"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+|\n+', text)

def analyze_and_colorize(text):
    sentences = split_sentences(text)
    colored_sentences = []

    for sentence in sentences:
        clean = sentence.strip()
        if clean:
            emotion = classify_emotion(clean)  # ✅ 문장 단위 감정 분석
            color = {
                "긍정": "green",
                "중립": "gray",
                "부정": "red"
            }.get(emotion, "black")
            colored_sentences.append(f'<span style="color:{color}">{clean}</span>')

    return " ".join(colored_sentences)



# ✅ 감정 분석 함수
def classify_emotion(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prediction = model(**tokens)
    prediction = F.softmax(prediction.logits, dim=1)
    output = prediction.argmax(dim=1).item()
    labels = ["부정", "중립", "긍정"]
    return labels[output]  # 감정 라벨만 반환

# ✅ JSON 파일에서 데이터 불러오기
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ✅ JSON 파일에 데이터 저장하기
def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    role = request.form["role"]
    name = request.form.get("name", "")
    password = request.form.get("password", "")

    if role == "student":
        session["role"] = "student"
        session["name"] = name
        return redirect(url_for("student_page"))

    elif role == "teacher":
        if password == "qwer1234":
            session["role"] = "teacher"
            return redirect(url_for("teacher_page"))
        else:
            return "<script>alert('교사 비밀번호가 틀렸습니다.'); window.location.href='/';</script>"

    return "<script>alert('로그인 실패. 다시 시도하세요.'); window.location.href='/';</script>"

@app.route("/submit_counsel", methods=["POST"])
def submit_counsel():
    if "role" not in session or session["role"] != "student":
        return jsonify({"error": "Unauthorized"}), 403

    name = session["name"]
    data = load_data()

    request_data = request.get_json()
    content = request_data.get("content", "")

    if not content:
        return jsonify({"error": "내용이 없습니다."}), 400

    # 👉 문장별로 감정 분석하고 색상 입힘
    emotion = classify_emotion(content)
    highlighted = analyze_and_colorize(content)

    new_entry = {
        "name": name,
        "content": content,
        "highlighted": highlighted,
        "emotion": emotion  # ✅ 이 부분을 추가하여 emotion 값을 저장합니다.
    }

    data.append(new_entry)
    save_data(data)

    return jsonify(new_entry)

@app.route("/student", methods=["GET", "POST"])
def student_page():
    if "role" not in session or session["role"] != "student":
        return redirect(url_for("home"))

    name = session["name"]
    message = None

    data = load_data()
    past_entries = [
        {
            "content": entry["content"],
            "highlighted": entry.get("highlighted", entry["content"]),  # fallback
            "emotion": entry.get("emotion", "중립")  # 없을 경우 대비해서 기본값 설정
        }
        for entry in data if entry["name"] == name
    ]


    past_entries.reverse()

    if request.method == "POST":
        content = request.form["content"]
        highlighted = analyze_and_colorize(content)
        emotion = classify_emotion(content) # ✅ 이 부분에서 감정 분석을 수행합니다.
        new_entry = {"name": name, "content": content, "highlighted": highlighted, "emotion": emotion} # ✅ emotion 값을 포함합니다.
        data.append(new_entry)
        save_data(data)
        message = "상담 내용이 저장되었습니다!"
        past_entries.insert(0, new_entry)


    return render_template("student.html", name=name, message=message, past_entries=past_entries)

@app.route("/teacher")
def teacher_page():
    if "role" not in session or session["role"] != "teacher":
        return redirect(url_for("home"))

    data = load_data()
    student_names = list(set([item["name"] for item in data]))  # 중복 제거된 학생 이름 목록
    return render_template("teacher.html", student_names=student_names)

@app.route("/get_student_data/<name>")
def get_student_data(name):
    if "role" not in session or session["role"] != "teacher":
        return jsonify({"error": "Unauthorized"}), 403

    data = load_data()
    student_data = [item for item in data if item["name"] == name]
    return jsonify(student_data)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/delete_entry/<name>/<int:index>", methods=["DELETE"])
def delete_entry(name, index):
    if "role" not in session or session["role"] != "teacher":
        return jsonify({"error": "Unauthorized"}), 403

    data = load_data()
    student_data = [item for item in data if item["name"] == name]

    if index < len(student_data):
        data.remove(student_data[index])
        save_data(data)
        return jsonify({"message": "상담 내용이 삭제되었습니다."})

    return jsonify({"error": "삭제할 상담 내용을 찾을 수 없습니다."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)