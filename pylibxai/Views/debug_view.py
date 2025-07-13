from pylibxai.Interfaces.view import ViewInterface
import os
import json

class DebugView(ViewInterface):
    def __init__(self, context):
        super().__init__(context)
        self.context = context

    def start(self):
        print("=== DEBUG VIEW: PylibxaiContext Content ===")
        print(f"Working directory: {self.context.workdir}")
        print()
        
        # Display directory structure
        print("Directory structure:")
        self._print_directory_tree(self.context.workdir)
        print()
        
        # Display content of each subdirectory
        subdirs = ["shap", "lrp", "lime"]
        for subdir in subdirs:
            subdir_path = os.path.join(self.context.workdir, subdir)
            if os.path.exists(subdir_path):
                print(f"=== {subdir.upper()} Directory Content ===")
                self._display_directory_content(subdir_path)
                print()
        
        # Display any JSON files in the root directory
        print("=== Root Directory Files ===")
        self._display_directory_content(self.context.workdir, root_only=True)
        print()
        
        print("=== DEBUG VIEW: Content Display Complete ===")

    def stop(self):
        pass

    def _print_directory_tree(self, directory, prefix="", max_depth=3, current_depth=0):
        """Print directory tree structure"""
        if current_depth >= max_depth:
            return
            
        try:
            items = sorted(os.listdir(directory))
            for i, item in enumerate(items):
                item_path = os.path.join(directory, item)
                is_last = i == len(items) - 1
                
                if os.path.isdir(item_path):
                    print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                    extension = "    " if is_last else "│   "
                    self._print_directory_tree(item_path, prefix + extension, max_depth, current_depth + 1)
                else:
                    print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
        except PermissionError:
            print(f"{prefix}[Permission Denied]")

    def _display_directory_content(self, directory, root_only=False):
        """Display content of files in directory"""
        try:
            files = os.listdir(directory)
            if not files:
                print(f"  Directory is empty: {directory}")
                return
                
            for file in sorted(files):
                file_path = os.path.join(directory, file)
                
                if os.path.isfile(file_path):
                    print(f"  File: {file}")
                    
                    # Display JSON file content
                    if file.endswith('.json'):
                        self._display_json_content(file_path)
                    
                    # Display file size for other files
                    else:
                        file_size = os.path.getsize(file_path)
                        print(f"    Size: {file_size} bytes")
                        
                        # Show file type info
                        if file.endswith(('.wav', '.mp3', '.mp4')):
                            print(f"    Type: Audio file")
                        elif file.endswith(('.png', '.jpg', '.jpeg')):
                            print(f"    Type: Image file")
                        elif file.endswith('.txt'):
                            print(f"    Type: Text file")
                    
                    print()
                elif os.path.isdir(file_path) and not root_only:
                    print(f"  Directory: {file}/")
                    
        except PermissionError:
            print(f"  [Permission Denied] Cannot access: {directory}")

    def _display_json_content(self, json_file_path):
        """Display JSON file content"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                print(f"    JSON Content:")
                
                # Pretty print JSON with limited depth
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 5:
                            print(f"      {key}: [Array with {len(value)} items]")
                        elif isinstance(value, dict):
                            print(f"      {key}: {dict}")
                        else:
                            print(f"      {key}: {value}")
                else:
                    print(f"      {str(data)[:100]}{'...' if len(str(data)) > 100 else ''}")
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"    Error reading JSON file: {e}")