import os

# Database URL
DATABASE_URL = "postgresql://oralytics:mysecretpassword@db/oralytics_db"

# JWT Settings
SECRET_KEY = "a_very_secret_key_for_jwt" # In production, use a more complex key from env variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30