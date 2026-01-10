import os
from dotenv import load_dotenv

print("CWD:", os.getcwd())
env_path = os.path.join(os.getcwd(), ".env")
print("Expect .env at:", env_path)
print("Exists?", os.path.exists(env_path))

load_dotenv(override=True)
key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"Key Found: '{key[:5]}...{key[-5:]}'")
    print(f"Key Length: {len(key)}")
    # Check for quotes
    if key.startswith('"') or key.startswith("'"):
        print("WARNING: Key starts with quote!")
    if " " in key:
        print("WARNING: Key contains spaces!")
else:
    print("KEY NOT FOUND")
