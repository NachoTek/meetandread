# meetandread

Windows desktop audio transcription widget with dual-mode enhancement.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python -m meetandread.main
```

## Development

The project uses a widget-style interface built with QGraphicsView for smooth animations
and complex visual components.

## Project Structure

```
src/meetandread/
├── __init__.py
├── main.py                 # Application entry point
└── widgets/
    ├── __init__.py
    └── main_widget.py      # Main QGraphicsView widget
```