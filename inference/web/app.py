"""Flask web interface: MJPEG stream + index page.

The frame source (a generator) is injected from main.py so this module stays
decoupled from the pipeline. Endpoints and response format are unchanged from
the original program.
"""
import cv2
from flask import Flask, Response, render_template_string


def build_app(pipeline):
    """Build the Flask app wired to the given pipeline instance."""
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Video Streaming</h1>
        <img src="{{ url_for('video_feed') }}">
    </body>
    </html>
    ''')

    @app.route('/video_feed')
    def video_feed():
        return Response(pipeline.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app
