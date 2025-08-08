from flask import Flask, render_template, request
import os
import subprocess
import webbrowser
from threading import Timer
from speech_recognition import Recognizer, AudioFile
from werkzeug.utils import secure_filename
import re

# Transformers (özetleme için)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Flask ayarları
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"mp3", "m4a"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Özetleme modeli (Türkçe)
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")


# Metin önişleme (temizleme)
def preprocess_text(text):
    text = re.sub(r"([.!?])", r"\1\n", text)  # Cümleleri ayır
    text = re.sub(r"\s+", " ", text)          # Fazla boşlukları temizle
    return text.strip()[:1500]                # 1500 karaktere sınırla

# Özetleme
def summarize_text(text):
    input_text = "tr: " + text
    inputs = tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=80,
        min_length=20,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
    subprocess.run(['ffmpeg', '-i', mp3_path, wav_path, '-y'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def transcribe_audio(wav_path):
    recognizer = Recognizer()
    with AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="tr-TR")
    except Exception as e:
        return f"Ses tanıma hatası: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "Dosya bulunamadı."
        file = request.files["file"]
        if file.filename == "":
            return "Dosya seçilmedi."
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            wav_path = convert_mp3_to_wav(file_path)
            metin = transcribe_audio(wav_path)

            temiz_metin = preprocess_text(metin)
            ozet = summarize_text(temiz_metin)

            return render_template("index.html", metin=metin, ozet=ozet)
    return render_template("index.html")

if __name__ == "__main__":
    import os
    from threading import Timer
    import webbrowser

    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"

    if port == 5000:
        Timer(1, lambda: webbrowser.open(url)).start()

    app.run(debug=True, host="0.0.0.0", port=port)
