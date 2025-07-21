import pytest
import sys
from unittest.mock import Mock, patch
from pathlib import Path

from pylibxai.Views.web_view import WebView
from pylibxai.pylibxai_context.pylibxai_context import PylibxaiContext


class TestWebView:
    """Test suite for WebView class."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        context = Mock(spec=PylibxaiContext)
        context.workdir = "/test/workdir"
        return context
    
    @pytest.fixture
    def web_view(self, mock_context):
        """Create a WebView instance for testing."""
        return WebView(mock_context, port=8000)
    
    def test_init_default_port(self, mock_context):
        """Test WebView initialization with default port."""
        web_view = WebView(mock_context)
        
        assert web_view.context == mock_context
        assert web_view.port == 9000
        assert web_view.server is None
        assert web_view.vite_process is None
    
    def test_init_custom_port(self, mock_context):
        """Test WebView initialization with custom port."""
        web_view = WebView(mock_context, port=8080)
        
        assert web_view.context == mock_context
        assert web_view.port == 8080
        assert web_view.server is None
        assert web_view.vite_process is None
    
    @patch('pylibxai.Views.web_view.get_install_path')
    def test_init_vite_dir_path(self, mock_get_install_path, mock_context):
        """Test that vite_dir is set correctly."""
        mock_install_path = Path("/mock/install/path")
        mock_get_install_path.return_value = mock_install_path
        
        web_view = WebView(mock_context)
        
        expected_vite_dir = mock_install_path / "pylibxai" / "pylibxai-ui"
        assert web_view.vite_dir == expected_vite_dir
    
    @patch('pylibxai.Views.web_view.subprocess.Popen')
    @patch('pylibxai.Views.web_view.run_file_server')
    @patch('pylibxai.Views.web_view.os.environ')
    def test_start_success(self, mock_environ, mock_run_file_server, mock_popen, web_view):
        """Test successful start of WebView."""
        mock_server = Mock()
        mock_run_file_server.return_value = mock_server
        
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        mock_env = {'EXISTING_VAR': 'value'}
        mock_environ.copy.return_value = mock_env
        
        web_view.start()
        
        mock_run_file_server.assert_called_once_with("/test/workdir", 8000)
        assert web_view.server == mock_server
        
        assert mock_env['VITE_PYLIBXAI_STATIC_PORT'] == '8000'
        
        mock_popen.assert_called_once_with(
            ['npm', 'run', 'dev'],
            cwd=web_view.vite_dir,
            env=mock_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True
        )
        
        assert web_view.vite_process == mock_process
    
    @patch('pylibxai.Views.web_view.subprocess.Popen')
    @patch('pylibxai.Views.web_view.run_file_server')
    @patch('builtins.print')
    def test_start_prints_messages(self, mock_print, mock_run_file_server, mock_popen, web_view):
        """Test that start() prints appropriate messages."""
        mock_run_file_server.return_value = Mock()
        mock_popen.return_value = Mock()
        
        web_view.start()
        
        print_calls = mock_print.call_args_list
        assert any('Vite UI directory:' in str(call) for call in print_calls)
        assert any('Vite UI launched at http://localhost:8000/' in str(call) for call in print_calls)
    
    @patch('pylibxai.Views.web_view.subprocess.Popen')
    @patch('pylibxai.Views.web_view.run_file_server')
    def test_start_subprocess_exception(self, mock_run_file_server, mock_popen, web_view):
        """Test start() handles subprocess exceptions gracefully."""
        mock_run_file_server.return_value = Mock()
        mock_popen.side_effect = OSError("Command not found")
        
        with pytest.raises(OSError):
            web_view.start()
    
    @patch('builtins.print')
    def test_stop_with_vite_process(self, mock_print, web_view):
        """Test stop() when vite_process exists."""
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        web_view.vite_process = mock_process
        
        mock_server = Mock()
        mock_server.shutdown = Mock()
        web_view.server = mock_server
        
        web_view.stop()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        
        mock_server.shutdown.assert_called_once()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Vite UI process terminated.' in call for call in print_calls)
        assert any('File server stopped.' in call for call in print_calls)
    
    @patch('builtins.print')
    def test_stop_without_vite_process(self, mock_print, web_view):
        """Test stop() when vite_process is None."""
        mock_server = Mock()
        mock_server.shutdown = Mock()
        web_view.server = mock_server
        web_view.vite_process = None
        
        web_view.stop()
        
        mock_server.shutdown.assert_called_once()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('File server stopped.' in call for call in print_calls)
        assert not any('Vite UI process terminated.' in call for call in print_calls)
    
    @patch('builtins.print')
    def test_stop_without_server(self, mock_print, web_view):
        """Test stop() when server is None."""
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        web_view.vite_process = mock_process
        web_view.server = None
        
        web_view.stop()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Vite UI process terminated.' in call for call in print_calls)
        assert not any('File server stopped.' in call for call in print_calls)
    
    def test_stop_with_nothing_to_stop(self, web_view):
        """Test stop() when both vite_process and server are None."""
        web_view.vite_process = None
        web_view.server = None
        
        web_view.stop()
    
    def test_stop_process_exception(self, web_view):
        """Test stop() handles process termination exceptions."""
        mock_process = Mock()
        mock_process.terminate.side_effect = OSError("Process not found")
        web_view.vite_process = mock_process
        
        with pytest.raises(OSError):
            web_view.stop()
    
    def test_stop_server_exception(self, web_view):
        """Test stop() handles server shutdown exceptions."""
        mock_server = Mock()
        mock_server.shutdown.side_effect = Exception("Server shutdown failed")
        web_view.server = mock_server
        
        with pytest.raises(Exception):
            web_view.stop()
    
    def test_inheritance(self, web_view):
        """Test that WebView properly inherits from ViewInterface."""
        from pylibxai.Interfaces.view import ViewInterface
        assert isinstance(web_view, ViewInterface)
    
    def test_context_workdir_passed_to_file_server(self, web_view):
        """Test that context.workdir is passed to the file server."""
        with patch('pylibxai.Views.web_view.run_file_server') as mock_run_file_server:
            with patch('pylibxai.Views.web_view.subprocess.Popen'):
                mock_run_file_server.return_value = Mock()
                
                web_view.start()
                
                mock_run_file_server.assert_called_once_with(
                    web_view.context.workdir, 
                    web_view.port
                )


class TestWebViewIntegration:
    """Integration tests for WebView (may require actual dependencies)."""
    
    @pytest.fixture
    def real_context(self, tmp_path):
        """Create a real context with temporary directory."""
        return PylibxaiContext(str(tmp_path))
    
    def test_real_context_integration(self, real_context):
        """Test WebView with a real context (marked as integration test)."""
        web_view = WebView(real_context, port=0)  # Use port 0 to get any available port
        
        assert web_view.context == real_context
        assert web_view.port == 0
        assert str(real_context.workdir) in web_view.context.workdir


class TestWebViewParametrized:
    """Parametrized tests for WebView."""
    
    @pytest.mark.parametrize("port", [8000, 9000, 3000, 8080])
    def test_different_ports(self, port):
        """Test WebView with different port values."""
        mock_context = Mock(spec=PylibxaiContext)
        mock_context.workdir = "/test"
        
        web_view = WebView(mock_context, port=port)
        assert web_view.port == port
    