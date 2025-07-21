import pytest
import tempfile
import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from pylibxai.pylibxai_context import PylibxaiContext


class TestPylibxaiContext:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def context(self, temp_dir):
        """Create PylibxaiContext instance"""
        return PylibxaiContext(temp_dir)
    
    # Constructor Tests
    def test_constructor_creates_workdir_if_not_exists(self, temp_dir):
        """Test workdir creation when it doesn't exist"""
        # Remove the temp_dir to test creation
        shutil.rmtree(temp_dir)
        assert not os.path.exists(temp_dir)
        
        context = PylibxaiContext(temp_dir)
        
        assert os.path.exists(temp_dir)
        assert context.workdir == temp_dir
    
    def test_constructor_creates_subdirectories(self, context, temp_dir):
        """Test that required subdirectories are created"""
        assert os.path.exists(os.path.join(temp_dir, "shap"))
        assert os.path.exists(os.path.join(temp_dir, "lrp"))
        assert os.path.exists(os.path.join(temp_dir, "lime"))
    
    def test_constructor_with_existing_workdir(self, temp_dir):
        """Test behavior when workdir already exists"""
        # Create some files in the directory first
        test_file = os.path.join(temp_dir, "existing_file.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        context = PylibxaiContext(temp_dir)
        
        assert os.path.exists(test_file)  # Existing files preserved
        assert context.workdir == temp_dir
    
    ## write_plt_image Tests
    #def test_write_plt_image_saves_figure(self, context, temp_dir):
    #    """Test saving matplotlib figure"""
    #    fig, ax = plt.subplots()
    #    ax.plot([1, 2, 3], [1, 4, 2])
    #    
    #    suffix = "test_plot.png"
    #    context.write_plt_image(fig, suffix)
    #    
    #    expected_path = os.path.join(temp_dir, suffix)
    #    assert os.path.exists(expected_path)
    #    
    #    plt.close(fig)
    
    # def test_write_plt_image_with_different_formats(self, context, temp_dir):
    #     """Test saving figures with different formats"""
    #     fig, ax = plt.subplots()
    #     ax.plot([1, 2, 3])
    #     
    #     formats = ["test.png", "test.jpg", "test.pdf"]
    #     
    #     for fmt in formats:
    #         context.write_plt_image(fig, fmt)
    #         assert os.path.exists(os.path.join(temp_dir, fmt))
    #     
    #     plt.close(fig)
    
    def test_write_plt_image_bbox_inches_parameter(self, context):
        """Test that bbox_inches='tight' is applied"""
        fig = MagicMock()
        
        context.write_plt_image(fig, "test.png")
        
        fig.savefig.assert_called_once()
        args, kwargs = fig.savefig.call_args
        assert kwargs.get('bbox_inches') == 'tight'
    
    # write_attribution Tests
    def test_write_attribution_creates_json(self, context, temp_dir):
        """Test attribution JSON creation"""
        test_data = np.array([0.1, 0.2, 0.3, 0.4])
        suffix = "test_attribution.json"
        
        context.write_attribution(test_data, suffix)
        
        file_path = os.path.join(temp_dir, suffix)
        assert os.path.exists(file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert "attributions" in data
        assert data["attributions"] == [0.1, 0.2, 0.3, 0.4]
    
    def test_write_attribution_with_different_shapes(self, context, temp_dir):
        """Test attribution writing with different numpy array shapes"""
        # 1D array
        data_1d = np.array([1.0, 2.0, 3.0])
        context.write_attribution(data_1d, "1d_attr.json")
        
        # 2D array
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        context.write_attribution(data_2d, "2d_attr.json")
        
        # Verify both files exist and contain correct data
        with open(os.path.join(temp_dir, "1d_attr.json"), 'r') as f:
            data = json.load(f)
            assert data["attributions"] == [1.0, 2.0, 3.0]
        
        with open(os.path.join(temp_dir, "2d_attr.json"), 'r') as f:
            data = json.load(f)
            assert data["attributions"] == [[1.0, 2.0], [3.0, 4.0]]
    
    def test_write_attribution_json_formatting(self, context, temp_dir):
        """Test JSON is properly formatted with indentation"""
        test_data = np.array([0.1, 0.2])
        suffix = "formatted_attr.json"
        
        context.write_attribution(test_data, suffix)
        
        with open(os.path.join(temp_dir, suffix), 'r') as f:
            content = f.read()
        
        # Check that it's properly indented (contains newlines and spaces)
        assert '\n' in content
        assert '    ' in content  # 4-space indentation
    
    # write_label_mapping Tests
    def test_write_label_mapping_creates_json(self, context, temp_dir):
        """Test label mapping JSON creation"""
        labels = {"0": "rock", "1": "pop", "2": "jazz"}
        suffix = "labels.json"
        
        context.write_label_mapping(labels, suffix)
        
        file_path = os.path.join(temp_dir, suffix)
        assert os.path.exists(file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert data == labels
    
    def test_write_label_mapping_with_integer_keys(self, context, temp_dir):
        """Test label mapping with integer keys"""
        labels = {0: "rock", 1: "pop", 2: "jazz"}
        suffix = "int_labels.json"
        
        context.write_label_mapping(labels, suffix)
        
        with open(os.path.join(temp_dir, suffix), 'r') as f:
            data = json.load(f)
        
        # JSON converts integer keys to strings
        expected = {"0": "rock", "1": "pop", "2": "jazz"}
        assert data == expected
    
    def test_write_label_mapping_with_nested_data(self, context, temp_dir):
        """Test label mapping with complex nested data"""
        labels = {
            "genres": {
                "rock": {"id": 0, "description": "Rock music"},
                "pop": {"id": 1, "description": "Pop music"}
            }
        }
        
        context.write_label_mapping(labels, "nested_labels.json")
        
        with open(os.path.join(temp_dir, "nested_labels.json"), 'r') as f:
            data = json.load(f)
        
        assert data == labels
    
    # write_audio Tests
    def test_write_audio_copy_file(self, context, temp_dir):
        """Test copying audio file from string path"""
        # Create a dummy audio file
        source_file = os.path.join(temp_dir, "source.wav")
        with open(source_file, 'w') as f:
            f.write("dummy audio content")
        
        suffix = "copied_audio.wav"
        context.write_audio(source_file, suffix)
        
        target_file = os.path.join(temp_dir, suffix)
        assert os.path.exists(target_file)
        
        with open(target_file, 'r') as f:
            content = f.read()
        assert content == "dummy audio content"
    
    @patch('soundfile.write')
    def test_write_audio_numpy_array(self, mock_sf_write, context):
        """Test writing numpy array as audio"""
        audio_data = np.array([0.1, 0.2, 0.3, 0.4])
        suffix = "array_audio.wav"
        sample_rate = 44100
        
        context.write_audio(audio_data, suffix, sample_rate)
        
        expected_path = os.path.join(context.workdir, suffix)
        mock_sf_write.assert_called_once_with(expected_path, audio_data, sample_rate)
    
    @patch('soundfile.write')
    def test_write_audio_with_kwargs(self, mock_sf_write, context):
        """Test write_audio passes through kwargs"""
        audio_data = np.array([0.1, 0.2])
        suffix = "test.wav"
        
        context.write_audio(audio_data, suffix, samplerate=48000, subtype='PCM_24')
        
        expected_path = os.path.join(context.workdir, suffix)
        mock_sf_write.assert_called_once_with(
            expected_path, audio_data, samplerate=48000, subtype='PCM_24'
        )
    
    # Integration Tests
    def test_complete_workflow(self, context, temp_dir):
        """Test complete workflow with multiple file operations"""
        # Write attribution
        attribution_data = np.array([0.5, 0.7, 0.2])
        context.write_attribution(attribution_data, "shap/attribution.json")
        
        # Write label mapping
        labels = {"0": "class_a", "1": "class_b"}
        context.write_label_mapping(labels, "labels.json")
        
        # Write plot
        fig, ax = plt.subplots()
        ax.bar([0, 1], [0.5, 0.7])
        context.write_plt_image(fig, "lrp/plot.png")
        
        # Verify all files exist
        assert os.path.exists(os.path.join(temp_dir, "shap", "attribution.json"))
        assert os.path.exists(os.path.join(temp_dir, "labels.json"))
        assert os.path.exists(os.path.join(temp_dir, "lrp", "plot.png"))
        
        plt.close(fig)
    
    def test_file_permissions_readable(self, context, temp_dir):
        """Test that created files are readable"""
        test_data = np.array([1.0, 2.0])
        context.write_attribution(test_data, "readable_test.json")
        
        file_path = os.path.join(temp_dir, "readable_test.json")
        assert os.access(file_path, os.R_OK)
    
    def test_subdirectory_structure_maintained(self, context, temp_dir):
        """Test that subdirectory structure is maintained across operations"""
        # Perform operations in different subdirectories
        context.write_attribution(np.array([1]), "shap/test1.json")
        context.write_attribution(np.array([2]), "lrp/test2.json")
        context.write_attribution(np.array([3]), "lime/test3.json")
        
        # Verify directory structure is maintained
        assert os.path.exists(os.path.join(temp_dir, "shap", "test1.json"))
        assert os.path.exists(os.path.join(temp_dir, "lrp", "test2.json"))
        assert os.path.exists(os.path.join(temp_dir, "lime", "test3.json"))