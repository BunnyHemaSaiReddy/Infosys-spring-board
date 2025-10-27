from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify, Response
import os
import pyrebase
from firebase_config import firebase_config
import requests
import cv2
import numpy as np
import base64
import tempfile
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-this-in-env')

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('home.html')


# -------- Signup --------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        
        # Server-side validation
        if password != confirm_password:
            flash('Passwords do not match. Please try again.')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.')
            return redirect(url_for('signup'))
        
        try:
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            flash("Verification email sent! Please check your inbox.")
            return redirect(url_for('login'))
        except requests.exceptions.HTTPError as e:
            error_message = 'UNKNOWN_ERROR'
            try:
                if e.response is not None:
                    error_json = e.response.json()
                    error_message = error_json.get('error', {}).get('message', 'UNKNOWN_ERROR')
            except Exception:
                pass
            mapped = {
                'EMAIL_EXISTS': 'Account already exists. Please go to login.',
                'OPERATION_NOT_ALLOWED': 'Password sign-in is disabled for this project.',
                'TOO_MANY_ATTEMPTS_TRY_LATER': 'Too many attempts. Try again later.'
            }.get(error_message, 'Signup failed. Please try again.')
            flash(mapped)
            return redirect(url_for('signup'))
        except Exception:
            flash('Signup failed. Please try again.')
            return redirect(url_for('signup'))
    return render_template('signup.html')

# -------- Login --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user_info = auth.get_account_info(user['idToken'])
            verified = user_info['users'][0].get('emailVerified', False)
            if not verified:
                flash("Please verify your email before logging in.")
                return redirect(url_for('login'))
            session['user'] = email
            return redirect(url_for('index'))
        except requests.exceptions.HTTPError as e:
            error_message = 'UNKNOWN_ERROR'
            try:
                if e.response is not None:
                    error_json = e.response.json()
                    error_message = error_json.get('error', {}).get('message', 'UNKNOWN_ERROR')
            except Exception:
                pass
            mapped = {
                'EMAIL_NOT_FOUND': 'Email not found. Please sign up.',
                'INVALID_PASSWORD': 'Invalid password.',
                'USER_DISABLED': 'This account has been disabled.'
            }.get(error_message, 'Login failed. Please try again.')
            if error_message == 'EMAIL_NOT_FOUND':
                flash(mapped)
                return redirect(url_for('signup'))
            flash(mapped)
            return redirect(url_for('login'))
        except Exception:
            flash('Login failed. Please try again.')
            return redirect(url_for('login'))
    return render_template('login.html')

# -------- Forgot Password --------
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form['email']
    try:
        auth.send_password_reset_email(email)
        return 'Password reset link sent to your email.', 200
    except Exception as e:
        return 'Error sending reset email. Try again.', 400

# -------- Google Login (using Firebase Web SDK, token from frontend) --------
@app.route('/google-login', methods=['POST'])
def google_login():
    id_token = request.json.get('idToken')
    try:
        req_url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={firebase_config['apiKey']}"
        headers = {'Content-Type': 'application/json'}
        res = requests.post(req_url, json={'idToken': id_token}, headers=headers)
        res.raise_for_status()
        user_info = res.json()
        email = user_info['users'][0]['email']
        session['user'] = email
        return redirect(url_for('index'))
    except requests.exceptions.HTTPError:
        return {"error": "Google sign-in failed."}, 400
    except Exception:
        return {"error": "Unexpected error during Google sign-in."}, 400

# -------- Index (Dashboard) --------
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session.get('user'))

# -------- Logout --------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# -------- Apple Detection Helpers --------

def _detect_apples_in_bgr_image(bgr_image: np.ndarray):
    """Return (count, annotated_bgr) for apples detected using HSV red mask."""
    if bgr_image is None or bgr_image.size == 0:
        return 0, None
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apple_count = 0
    annotated = bgr_image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            apple_count += 1
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(annotated, center, radius, (0, 255, 0), 3)
            cv2.putText(annotated, f"{apple_count}", (int(x) - 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(annotated, f"Total Apples: {apple_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return apple_count, annotated

def _bgr_image_to_base64_png(bgr_image: np.ndarray) -> str:
    success, buf = cv2.imencode('.png', bgr_image)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode('utf-8')

# -------- Apple Detection Endpoints --------

@app.route('/apple/detect-image', methods=['POST'])
def apple_detect_image():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({"error": "No image uploaded"}), 400
    filename = secure_filename(file.filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        file.save(path)
        bgr = cv2.imread(path)
    count, annotated = _detect_apples_in_bgr_image(bgr)
    img64 = _bgr_image_to_base64_png(annotated) if annotated is not None else ""
    return jsonify({"count": count, "image": img64})

@app.route('/apple/detect-webcam', methods=['POST'])
def apple_detect_webcam():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data_url = request.json.get('image') if request.is_json else None
    if not data_url:
        return jsonify({"error": "No image data"}), 400
    try:
        header, b64data = data_url.split(',', 1)
        img_bytes = base64.b64decode(b64data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400
    count, annotated = _detect_apples_in_bgr_image(bgr)
    img64 = _bgr_image_to_base64_png(annotated) if annotated is not None else ""
    return jsonify({"count": count, "image": img64})

@app.route('/apple/detect-video', methods=['POST'])
def apple_detect_video():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    file = request.files.get('video')
    if file is None or file.filename == '':
        return jsonify({"error": "No video uploaded"}), 400
    filename = secure_filename(file.filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        file.save(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 400
        ret, frame = cap.read()
        cap.release()
    if not ret:
        return jsonify({"error": "Failed to read video"}), 400
    count, annotated = _detect_apples_in_bgr_image(frame)
    img64 = _bgr_image_to_base64_png(annotated) if annotated is not None else ""
    return jsonify({"count": count, "image": img64})

@app.route('/apple/process-video', methods=['POST'])
def apple_process_video():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    file = request.files.get('video')
    if file is None or file.filename == '':
        return jsonify({"error": "No video uploaded"}), 400
    filename = secure_filename(file.filename)

    os.makedirs(os.path.join('static', 'outputs'), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, filename)
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_name = os.path.splitext(filename)[0] + '_processed.mp4'
        out_path = os.path.join('static', 'outputs', out_name)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, annotated = _detect_apples_in_bgr_image(frame)
            writer.write(annotated)

        cap.release()
        writer.release()

    return jsonify({"video_url": url_for('static', filename=f'outputs/{out_name}', _external=False)})

# -------- Apple Webcam Live Stream (start/stop) --------

_apple_cam = None
_apple_cam_lock = threading.Lock()
_apple_video_path_by_session = {}
_last_webcam_frame = None  # type: np.ndarray | None
_last_file_frame = None    # type: np.ndarray | None
_last_webcam_count = 0
_last_file_count = 0

@app.route('/apple/stream-webcam/start', methods=['POST'])
def apple_stream_webcam_start():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global _apple_cam
    with _apple_cam_lock:
        # Release any existing handle before opening a new one
        if _apple_cam is not None:
            try:
                _apple_cam.release()
            except Exception:
                pass
            _apple_cam = None
        _apple_cam = cv2.VideoCapture(0)
        if not _apple_cam.isOpened():
            _apple_cam = None
            return jsonify({"error": "Cannot open webcam"}), 400
    return jsonify({"status": "started"})

def _apple_webcam_generator():
    global _apple_cam
    if _apple_cam is None or not _apple_cam.isOpened():
        return
    try:
        while True:
            with _apple_cam_lock:
                cam = _apple_cam
            if cam is None or not cam.isOpened():
                break
            ok, frame = cam.read()
            if not ok:
                break
            try:
                frame = cv2.resize(frame, (300, 200))
            except Exception:
                pass
            c, annotated = _detect_apples_in_bgr_image(frame)
            # store last frame for pause snapshot
            try:
                globals()['_last_webcam_frame'] = annotated.copy()
            except Exception:
                globals()['_last_webcam_frame'] = annotated
            globals()['_last_webcam_count'] = c
            success, buf = cv2.imencode('.jpg', annotated)
            if not success:
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    except GeneratorExit:
        # Client disconnected; fall through to cleanup
        pass

@app.route('/apple/stream-webcam/feed')
def apple_stream_webcam_feed():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return Response(_apple_webcam_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/apple/stream-webcam/stop', methods=['POST'])
def apple_stream_webcam_stop():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global _apple_cam
    with _apple_cam_lock:
        if _apple_cam is not None:
            try:
                _apple_cam.release()
            except Exception:
                pass
            _apple_cam = None
    return jsonify({"status": "stopped"})

# Aliases matching the sample-style endpoints
@app.route('/apple/webcam/start', methods=['POST'])
def apple_webcam_start_alias():
    return apple_stream_webcam_start()

@app.route('/apple/video_feed')
def apple_video_feed_alias():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return Response(_apple_webcam_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/apple/webcam/stop', methods=['POST'])
def apple_webcam_stop_alias():
    return apple_stream_webcam_stop()

# Force reset route in case of stuck camera
@app.route('/apple/webcam/reset', methods=['POST'])
def apple_webcam_reset():
    return apple_stream_webcam_stop()

@app.route('/apple/webcam/snapshot')
def apple_webcam_snapshot():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global _last_webcam_frame
    frame = _last_webcam_frame
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return jsonify({"error": "No frame available"}), 400
    b64 = _bgr_image_to_base64_png(frame)
    return jsonify({"image": b64, "count": globals().get('_last_webcam_count', 0)})

# -------- Video file streaming with cv2 (start/stop + feed) --------

@app.route('/apple/video/start', methods=['POST'])
def apple_video_start():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    file = request.files.get('video')
    if file is None or file.filename == '':
        return jsonify({"error": "No video uploaded"}), 400
    filename = secure_filename(file.filename)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, filename)
    file.save(path)
    # Store per-session path
    _apple_video_path_by_session[session.get('user')] = path
    return jsonify({"status": "ready"})

def _apple_video_generator(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame = cv2.resize(frame, (640, 360))
            except Exception:
                pass
            c, annotated = _detect_apples_in_bgr_image(frame)
            try:
                globals()['_last_file_frame'] = annotated.copy()
            except Exception:
                globals()['_last_file_frame'] = annotated
            globals()['_last_file_count'] = c
            success, buf = cv2.imencode('.jpg', annotated)
            if not success:
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    except GeneratorExit:
        pass
    finally:
        cap.release()

@app.route('/apple/video_feed_file')
def apple_video_feed_file():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    path = _apple_video_path_by_session.get(session.get('user'))
    if not path or not os.path.exists(path):
        return jsonify({"error": "No video prepared"}), 400
    return Response(_apple_video_generator(path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/apple/video/stop', methods=['POST'])
def apple_video_stop():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_key = session.get('user')
    path = _apple_video_path_by_session.pop(user_key, None)
    if path:
        try:
            os.remove(path)
        except Exception:
            pass
        try:
            os.rmdir(os.path.dirname(path))
        except Exception:
            pass
    return jsonify({"status": "stopped"})

@app.route('/apple/video/snapshot')
def apple_video_snapshot():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global _last_file_frame
    frame = _last_file_frame
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return jsonify({"error": "No frame available"}), 400
    b64 = _bgr_image_to_base64_png(frame)
    return jsonify({"image": b64, "count": globals().get('_last_file_count', 0)})

# ========== RUN ==========

if __name__ == '__main__':
    app.run(debug=True)
