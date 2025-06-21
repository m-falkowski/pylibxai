import subprocess
import sys
import os
from .file_serve import run_file_server
from pylibxai.utils import get_install_path
from pylibxai.Interfaces.view import ViewInterface
from pylibxai.pylibxai_context import PylibxaiContext

class WebView(ViewInterface):
    def __init__(self, context, port=9000):
        super().__init__(context, port)
        self.vite_dir = get_install_path() / "pylibxai" / "pylibxai-ui"
        self.server = None
        self.vite_process = None

    def start(self):
        # Start the file server
        self.server = run_file_server(self.context.workdir, self.port)
        # Start the Vite UI
        env = os.environ.copy()
        env['VITE_PYLIBXAI_STATIC_PORT'] = str(self.port)
        print(f"Vite UI directory: {self.vite_dir}")
        self.vite_process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=self.vite_dir,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True
        )
        print(f"Vite UI launched at http://localhost:{self.port}/ (UI dev server running)")

    def stop(self):
        if self.server:
            self.server.shutdown()
            print("File server stopped.")
        if self.vite_process:
            self.vite_process.terminate()
            self.vite_process.wait()
            print("Vite UI process terminated.")
