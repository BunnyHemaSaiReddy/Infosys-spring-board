import os

# Web SDK config (do not include service account private keys here)
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyA_dvNAf19mcXXdLZjoHsgC8SmmLKcbKKg"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "signup-login-b28f5.firebaseapp.com"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", ""),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "signup-login-b28f5"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "signup-login-b28f5.appspot.com"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "946139293369"),
    "appId": os.getenv("FIREBASE_APP_ID", "1:946139293369:web:25c7ab86146ec144a35d68"),
}
