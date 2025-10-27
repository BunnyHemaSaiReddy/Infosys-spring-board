from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify, Response
import os
import pyrebase
from firebase_config import firebase_config
import requests


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

# ========== RUN ==========

if __name__ == '__main__':
    app.run(debug=True)

