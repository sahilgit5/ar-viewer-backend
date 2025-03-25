from app import app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
else:
    # This ensures the app is properly initialized when running with gunicorn
    port = int(os.environ.get("PORT", 5000)) 