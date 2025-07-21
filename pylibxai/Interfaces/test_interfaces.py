import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Callable
from pylibxai.Interfaces import (
    LimeAdapter, 
    ShapAdapter, 
    LrpAdapter, 
    ModelLabelProvider, 
    ViewInterface,
    ViewType
)

class TestAbstractMethodEnforcement:
    """Test that abstract methods are properly enforced"""

    # LimeAdapter Tests
    def test_lime_adapter_incomplete_implementation_raises_error(self):
        """Test that LimeAdapter raises TypeError when abstract methods are not implemented"""
        
        class IncompleteLimeAdapter(LimeAdapter):
            # Missing get_lime_predict_fn implementation
            pass
        
        with pytest.raises(TypeError) as excinfo:
            IncompleteLimeAdapter()
        
        assert "abstract method" in str(excinfo.value).lower()
        assert "get_lime_predict_fn" in str(excinfo.value)

    def test_lime_adapter_complete_implementation_works(self):
        """Test that LimeAdapter works when all abstract methods are implemented"""
        
        class CompleteLimeAdapter(LimeAdapter):
            def get_lime_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
                def predict(x):
                    return np.array([0.5, 0.3, 0.2])
                return predict
        
        # Should not raise an error
        adapter = CompleteLimeAdapter()
        predict_fn = adapter.get_lime_predict_fn()
        result = predict_fn(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    # ShapAdapter Tests
    def test_shap_adapter_incomplete_implementation_raises_error(self):
        """Test that ShapAdapter raises TypeError when abstract methods are not implemented"""
        
        class IncompleteShapAdapter(ShapAdapter):
            # Missing both abstract methods
            pass
        
        with pytest.raises(TypeError) as excinfo:
            IncompleteShapAdapter()
        
        assert "abstract method" in str(excinfo.value).lower()

    def test_shap_adapter_partial_implementation_raises_error(self):
        """Test that ShapAdapter raises TypeError when only one abstract method is implemented"""
        
        class PartialShapAdapter(ShapAdapter):
            def get_shap_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
                def predict(x):
                    return torch.tensor([0.1, 0.9])
                return predict
            # Missing shap_prepare_inference_input
        
        with pytest.raises(TypeError) as excinfo:
            PartialShapAdapter()
        
        assert "abstract method" in str(excinfo.value).lower()
        assert "shap_prepare_inference_input" in str(excinfo.value)

    def test_shap_adapter_complete_implementation_works(self):
        """Test that ShapAdapter works when all abstract methods are implemented"""
        
        class CompleteShapAdapter(ShapAdapter):
            def get_shap_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
                def predict(x):
                    return torch.tensor([0.1, 0.9])
                return predict
            
            def shap_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(0)
        
        # Should not raise an error
        adapter = CompleteShapAdapter()
        predict_fn = adapter.get_shap_predict_fn()
        result = predict_fn(torch.tensor([1.0, 2.0]))
        assert isinstance(result, torch.Tensor)
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        prepared = adapter.shap_prepare_inference_input(input_tensor)
        assert prepared.shape[0] == 1  # Should have batch dimension

    # LrpAdapter Tests
    def test_lrp_adapter_incomplete_implementation_raises_error(self):
        """Test that LrpAdapter raises TypeError when abstract methods are not implemented"""
        
        class IncompleteLrpAdapter(LrpAdapter):
            # Missing get_lrp_predict_fn implementation
            pass
        
        with pytest.raises(TypeError) as excinfo:
            IncompleteLrpAdapter()
        
        assert "abstract method" in str(excinfo.value).lower()
        assert "get_lrp_predict_fn" in str(excinfo.value)

    def test_lrp_adapter_complete_implementation_works(self):
        """Test that LrpAdapter works when all abstract methods are implemented"""
        
        class CompleteLrpAdapter(LrpAdapter):
            def get_lrp_predict_fn(self) -> nn.Module:
                class SimpleModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = nn.Linear(10, 2)
                    
                    def forward(self, x):
                        return self.linear(x)
                
                return SimpleModel()
        
        # Should not raise an error
        adapter = CompleteLrpAdapter()
        model = adapter.get_lrp_predict_fn()
        assert isinstance(model, nn.Module)

    # ModelLabelProvider Tests
    def test_model_label_provider_incomplete_implementation_raises_error(self):
        """Test that ModelLabelProvider raises TypeError when abstract methods are not implemented"""
        
        class IncompleteModelLabelProvider(ModelLabelProvider):
            # Missing both abstract methods
            pass
        
        with pytest.raises(TypeError) as excinfo:
            IncompleteModelLabelProvider()
        
        assert "abstract method" in str(excinfo.value).lower()

    def test_model_label_provider_partial_implementation_raises_error(self):
        """Test that ModelLabelProvider raises TypeError when only one abstract method is implemented"""
        
        class PartialModelLabelProvider(ModelLabelProvider):
            def get_label_mapping(self) -> Dict[int, str]:
                return {0: "rock", 1: "pop", 2: "jazz"}
            # Missing map_target_to_id
        
        with pytest.raises(TypeError) as excinfo:
            PartialModelLabelProvider()
        
        assert "abstract method" in str(excinfo.value).lower()
        assert "map_target_to_id" in str(excinfo.value)

    def test_model_label_provider_complete_implementation_works(self):
        """Test that ModelLabelProvider works when all abstract methods are implemented"""
        
        class CompleteModelLabelProvider(ModelLabelProvider):
            def get_label_mapping(self) -> Dict[int, str]:
                return {0: "rock", 1: "pop", 2: "jazz"}
            
            def map_target_to_id(self, target: str) -> int:
                mapping = {"rock": 0, "pop": 1, "jazz": 2}
                return mapping.get(target, -1)
        
        # Should not raise an error
        provider = CompleteModelLabelProvider()
        labels = provider.get_label_mapping()
        assert labels[0] == "rock"
        
        rock_id = provider.map_target_to_id("rock")
        assert rock_id == 0

    # ViewInterface Tests
    def test_view_interface_incomplete_implementation_raises_error(self):
        """Test that ViewInterface raises TypeError when abstract methods are not implemented"""
        
        class IncompleteViewInterface(ViewInterface):
            # Missing all abstract methods including __init__
            pass
        
        with pytest.raises(TypeError) as excinfo:
            IncompleteViewInterface()
        
        assert "abstract method" in str(excinfo.value).lower()

    def test_view_interface_partial_implementation_raises_error(self):
        """Test that ViewInterface raises TypeError when only some abstract methods are implemented"""
        
        class PartialViewInterface(ViewInterface):
            def __init__(self, context):
                self.context = context
            
            def start(self) -> None:
                pass
            # Missing stop method
        
        with pytest.raises(TypeError) as excinfo:
            PartialViewInterface(context="dummy_context")
        
        assert "abstract method" in str(excinfo.value).lower()
        assert "stop" in str(excinfo.value)

    def test_view_interface_complete_implementation_works(self):
        """Test that ViewInterface works when all abstract methods are implemented"""
        
        class CompleteViewInterface(ViewInterface):
            def __init__(self, context):
                self.context = context
                self.started = False
            
            def start(self) -> None:
                self.started = True
            
            def stop(self) -> None:
                self.started = False
        
        # Should not raise an error
        view = CompleteViewInterface(context="test_context")
        assert view.context == "test_context"
        assert not view.started
        
        view.start()
        assert view.started
        
        view.stop()
        assert not view.started

    # Multiple Interface Implementation Tests
    def test_multiple_interfaces_incomplete_raises_error(self):
        """Test class implementing multiple interfaces with missing methods"""
        
        class MultipleInterfaceAdapter(LimeAdapter, ShapAdapter):
            def get_lime_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
                def predict(x):
                    return np.array([0.1, 0.9])
                return predict
            # Missing ShapAdapter methods
        
        with pytest.raises(TypeError) as excinfo:
            MultipleInterfaceAdapter()
        
        assert "abstract method" in str(excinfo.value).lower()

    def test_multiple_interfaces_complete_works(self):
        """Test class implementing multiple interfaces with all methods"""
        
        class CompleteMultipleInterfaceAdapter(LimeAdapter, ShapAdapter, ModelLabelProvider):
            def get_lime_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
                def predict(x):
                    return np.array([0.1, 0.9])
                return predict
            
            def get_shap_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
                def predict(x):
                    return torch.tensor([0.1, 0.9])
                return predict
            
            def shap_prepare_inference_input(self, x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(0)
            
            def get_label_mapping(self) -> Dict[int, str]:
                return {0: "class_a", 1: "class_b"}
            
            def map_target_to_id(self, target: str) -> int:
                mapping = {"class_a": 0, "class_b": 1}
                return mapping.get(target, -1)
        
        # Should not raise an error
        adapter = CompleteMultipleInterfaceAdapter()
        
        # Test LimeAdapter functionality
        lime_predict = adapter.get_lime_predict_fn()
        lime_result = lime_predict(np.array([1, 2, 3]))
        assert isinstance(lime_result, np.ndarray)
        
        # Test ShapAdapter functionality
        shap_predict = adapter.get_shap_predict_fn()
        shap_result = shap_predict(torch.tensor([1.0, 2.0]))
        assert isinstance(shap_result, torch.Tensor)
        
        # Test ModelLabelProvider functionality
        labels = adapter.get_label_mapping()
        assert labels[0] == "class_a"
        
        class_id = adapter.map_target_to_id("class_a")
        assert class_id == 0

    # Edge Cases
    def test_abstract_method_inheritance_chain(self):
        """Test abstract method enforcement through inheritance chain"""
        
        class BaseAdapter(LimeAdapter):
            def some_helper_method(self):
                return "helper"
            # Still missing get_lime_predict_fn
        
        class DerivedAdapter(BaseAdapter):
            pass
        
        with pytest.raises(TypeError) as excinfo:
            DerivedAdapter()
        
        assert "abstract method" in str(excinfo.value).lower()

    def test_concrete_method_overriding_abstract(self):
        """Test that concrete implementations properly override abstract methods"""
        
        class ConcreteAdapter(LimeAdapter):
            def get_lime_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
                # This should properly override the abstract method
                def predict(x):
                    return x * 2
                return predict
        
        adapter = ConcreteAdapter()
        predict_fn = adapter.get_lime_predict_fn()
        result = predict_fn(np.array([1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))
