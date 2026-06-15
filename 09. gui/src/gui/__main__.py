"""Entry point: python -m gui"""
from gui.app import app


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
