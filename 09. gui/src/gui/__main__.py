"""Entry point: python -m gui"""
from gui.app import app

if __name__ == "__main__":
    app.run(debug=True, port=5000)
