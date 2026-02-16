from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv(dotenv_path=".env")

print("DATABASE_URL:", os.getenv("DATABASE_URL"))
print("SECRET_KEY:", os.getenv("SECRET_KEY"))