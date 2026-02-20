import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import PyPDF2
from summarizer import generate_summary
from models import db, SummarySession, QuizQuestion
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.secret_key = os.urandom(24)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gistq.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

with app.app_context():
    db.create_all()
    print("Database initialized")

def extract_text_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
        text = soup.get_text(separator=" ")
        return " ".join(text.split())
    except Exception as e:
        print(f"URL extraction error: {e}")
        raise Exception(f"Failed to extract text from URL: {str(e)}")

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        print(f"TXT extraction error: {e}")
        raise Exception(f"Failed to read text file: {str(e)}")

def generate_title_from_text(text):
    try:
        snippet = text[:500]
        prompt = f"""Generate a short, descriptive title (3-6 words) for this text. Return ONLY the title, nothing else.

Text: {snippet}"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=50
        )
        title = response.choices[0].message.content.strip()
        title = title.strip('"').strip("'")
        return title[:100]
    except:
        words = text.split()[:5]
        return ' '.join(words) + '...'

def generate_questions_groq(text, num_questions):
    try:
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        prompt = f"""You are a quiz generator. Based on the text below, create exactly {num_questions} educational questions with detailed answers.

CRITICAL: You must return ONLY a valid JSON array. No explanations, no markdown, no additional text.

Format (return this exact structure):
[
{{"question": "First question?", "answer": "Detailed answer to first question."}},
{{"question": "Second question?", "answer": "Detailed answer to second question."}}
]

Text to analyze:
{text}

Return ONLY the JSON array now:"""
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates quiz questions. You ONLY respond with valid JSON arrays, never with explanations or markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=2000
        )
        raw = chat.choices[0].message.content.strip()
        print(f"\n{'='*50}")
        print(f"RAW GROQ RESPONSE:")
        print(f"{'='*50}")
        print(raw)
        print(f"{'='*50}\n")
        questions = None
        try:
            questions = json.loads(raw)
            print("Successfully parsed raw response as JSON")
        except:
            pass
        if questions is None:
            try:
                cleaned = raw.replace('```json', '').replace('```', '').strip()
                questions = json.loads(cleaned)
                print("Successfully parsed after removing markdown")
            except:
                pass
        if questions is None:
            try:
                import re
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    questions = json.loads(json_str)
                    print("Successfully extracted JSON with regex")
            except:
                pass
        if questions is None:
            try:
                lines = raw.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('['):
                        remaining = '\n'.join(lines[i:])
                        start = remaining.find('[')
                        end = remaining.rfind(']') + 1
                        if start != -1 and end > start:
                            json_str = remaining[start:end]
                            questions = json.loads(json_str)
                            print("Successfully parsed with line-by-line search")
                            break
            except:
                pass
        if questions is None:
            print("All JSON parsing attempts failed")
            print(f"Response type: {type(raw)}")
            print(f"Response length: {len(raw)}")
            raise Exception("Could not extract valid JSON from Groq response")
        if not isinstance(questions, list):
            print(f"Response is not a list, it's a {type(questions)}")
            raise ValueError("Response is not a list")
        print(f"Got {len(questions)} questions from Groq")
        valid_questions = []
        for idx, q in enumerate(questions):
            if isinstance(q, dict) and 'question' in q and 'answer' in q:
                valid_questions.append({
                    'question': str(q['question']).strip(),
                    'answer': str(q['answer']).strip()
                })
                print(f"Question {idx+1}: {q['question'][:50]}...")
            else:
                print(f"Skipping invalid question {idx+1}: {q}")
        if len(valid_questions) == 0:
            raise ValueError("No valid questions found in response")
        result = valid_questions[:num_questions]
        print(f"Returning {len(result)} valid questions\n")
        return result
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic content: {raw[:500]}...")
        raise Exception(f"Invalid JSON in Groq response: {str(e)}")
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to generate questions: {str(e)}")

def verify_answer_groq(question, user_answer, correct_answer):
    try:
        prompt = f"""Compare the user's answer to the correct answer for this question.

Question: {question}

Correct Answer: {correct_answer}

User's Answer: {user_answer}

Determine if the user's answer is essentially correct (captures the main idea) even if worded differently.

Return ONLY this JSON format (no other text):
{{"is_correct": true, "feedback": "Brief explanation"}}

or

{{"is_correct": false, "feedback": "What was missing or incorrect"}}"""
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fair quiz grader. You ONLY respond with valid JSON objects."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        raw = chat.choices[0].message.content.strip()
        print(f"Verify answer response: {raw}")
        raw = raw.replace('```json', '').replace('```', '').strip()
        result = None
        try:
            result = json.loads(raw)
        except:
            pass
        if result is None:
            try:
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
            except:
                pass
        if result is None:
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    result = json.loads(raw[start:end])
            except:
                pass
        if result is None:
            print("Warning: Could not parse verification response, returning False")
            return {
                "is_correct": False,
                "feedback": "Unable to verify answer automatically. Please check with instructor."
            }
        if "is_correct" not in result:
            result["is_correct"] = False
        if "feedback" not in result:
            result["feedback"] = "No feedback available."
        if isinstance(result["is_correct"], str):
            result["is_correct"] = result["is_correct"].lower() in ['true', 'yes', '1']
        return result
    except Exception as e:
        print(f"Error in verify_answer: {e}")
        return {
            "is_correct": False,
            "feedback": "Error verifying answer. Please try again."
        }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    return render_template("quiz.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        input_type = request.form.get("input_type", "text")
        level = request.form.get("level", "summary")
        num_questions = int(request.form.get("num_questions", 5))
        if level not in ["abstract", "summary", "article"]:
            level = "summary"
        if num_questions not in [3, 5, 10]:
            num_questions = 5
        text = ""
        source_info = ""
        if input_type == "text":
            text = request.form.get("text", "").strip()
            if not text or len(text) < 50:
                return jsonify({"error": "Text must be at least 50 characters"}), 400
            source_info = "Direct text input"
        elif input_type == "url":
            url = request.form.get("url", "").strip()
            if not url:
                return jsonify({"error": "URL is required"}), 400
            text = extract_text_from_url(url)
            source_info = url
            if len(text) < 50:
                return jsonify({"error": "Extracted text is too short"}), 400
        elif input_type == "file":
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            filename = file.filename.lower()
            source_info = file.filename
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif filename.endswith('.txt'):
                text = extract_text_from_txt(file)
            else:
                return jsonify({"error": "Unsupported file type. Use PDF or TXT"}), 400
            if len(text) < 50:
                return jsonify({"error": "Extracted text is too short"}), 400
        else:
            return jsonify({"error": "Invalid input type"}), 400
        print(f"\n{'='*50}")
        print(f"PROCESSING REQUEST")
        print(f"{'='*50}")
        print(f"Input type: {input_type}")
        print(f"Text length: {len(text)} characters")
        print(f"Level: {level}")
        print(f"Questions: {num_questions}")
        print(f"{'='*50}\n")
        print("Generating title...")
        title = generate_title_from_text(text)
        print(f"Title: {title}\n")
        print("Generating summary with BART...")
        summary = generate_summary(text, level)
        print(f"Summary generated: {len(summary.split())} words\n")
        print(f"Generating {num_questions} questions with Groq...")
        questions = generate_questions_groq(text, num_questions)
        print(f"Generated {len(questions)} questions\n")
        if not summary:
            return jsonify({"error": "Failed to generate summary"}), 500
        if len(questions) == 0:
            return jsonify({"error": "Failed to generate questions. Please try again."}), 500
        try:
            session = SummarySession(
                title=title,
                original_text=text[:5000],
                summary_text=summary,
                level=level,
                word_count=len(summary.split()),
                input_type=input_type,
                source_info=source_info
            )
            db.session.add(session)
            db.session.flush()
            for idx, q in enumerate(questions):
                question = QuizQuestion(
                    session_id=session.id,
                    question_text=q['question'],
                    correct_answer=q['answer'],
                    question_order=idx + 1
                )
                db.session.add(question)
            db.session.commit()
            print(f"Saved to database with ID: {session.id}\n")
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {e}")
        return jsonify({
            "summary": summary,
            "level": level,
            "word_count": len(summary.split()),
            "questions": questions,
            "title": title,
            "session_id": session.id,
            "status": "success"
        })
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR IN /summarize")
        print(f"{'='*50}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}\n")
        return jsonify({"error": str(e)}), 500

@app.route("/verify_answer", methods=["POST"])
def verify_answer():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        question = data.get('question')
        user_answer = data.get('user_answer')
        correct_answer = data.get('correct_answer')
        question_id = data.get('question_id')
        if not all([question, user_answer, correct_answer]):
            return jsonify({"error": "Missing required fields"}), 400
        result = verify_answer_groq(question, user_answer, correct_answer)
        if question_id:
            try:
                quiz_question = QuizQuestion.query.get(question_id)
                if quiz_question:
                    quiz_question.user_answer = user_answer
                    quiz_question.is_correct = result['is_correct']
                    quiz_question.feedback = result.get('feedback', '')
                    db.session.commit()
            except Exception as e:
                print(f"Error saving answer: {e}")
                db.session.rollback()
        return jsonify(result)
    except Exception as e:
        print(f"Error in /verify_answer: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "is_correct": False,
            "feedback": "Error verifying answer. Please try again."
        }), 500

@app.route("/api/history")
def get_history():
    try:
        sessions = SummarySession.query.order_by(SummarySession.created_at.desc()).all()
        return jsonify({
            "sessions": [
                {
                    'id': s.id,
                    'title': s.title,
                    'level': s.level,
                    'created_at': s.created_at.strftime('%Y-%m-%d %H:%M'),
                    'word_count': s.word_count
                }
                for s in sessions
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/session/<int:session_id>")
def get_session(session_id):
    try:
        session = SummarySession.query.get_or_404(session_id)
        return jsonify(session.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/session/<int:session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        session = SummarySession.query.get_or_404(session_id)
        db.session.delete(session)
        db.session.commit()
        return jsonify({"success": True, "message": "Session deleted"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "summarizer": "BART (facebook/bart-large-cnn)",
        "qa_model": MODEL,
        "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
        "database": "SQLite"
    })

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set!")
        print("Set it with: export GROQ_API_KEY='your-key-here'")
    os.makedirs("templates", exist_ok=True)
    print("\n" + "="*50)
    print("GistQ - Smart Summarization & Quiz")
    print("="*50)
    print("Summarization: BART (facebook/bart-large-cnn)")
    print(f"Q&A Generation: Groq ({MODEL})")
    print("Database: SQLite (gistq.db)")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
