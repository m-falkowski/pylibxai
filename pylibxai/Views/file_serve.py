import http.server
import socketserver
import threading
import os

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        return super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def run_file_server(directory, port=9000):
    """Start a file server in a background thread."""
    os.chdir(directory)

    handler = CORSHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    print(f"Serving files from {directory} at http://localhost:{port}/")

    # Run in background thread
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    return httpd  # To stop later with httpd.shutdown()
