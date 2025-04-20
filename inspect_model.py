import pickle
import sys
import os
import numpy as np

def inspect_object(obj, name="object", depth=0, max_depth=3):
    """
    Recursively inspect an object and print its structure
    
    Args:
        obj: Object to inspect
        name: Name of the object
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    """
    indent = "  " * depth
    
    # Print the object name and type
    print(f"{indent}{name} ({type(obj).__name__}):")
    
    # If maximum depth reached, stop recursion
    if depth >= max_depth:
        print(f"{indent}  [Maximum recursion depth reached]")
        return
        
    # Handle different types of objects
    if isinstance(obj, dict):
        if len(obj) == 0:
            print(f"{indent}  [Empty dictionary]")
        else:
            for key, value in obj.items():
                inspect_object(value, f"Key: {key}", depth + 1, max_depth)
    
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) == 0:
            print(f"{indent}  [Empty {type(obj).__name__}]")
        else:
            print(f"{indent}  [Contains {len(obj)} items]")
            # Print first few items
            for i, item in enumerate(list(obj)[:3]):
                inspect_object(item, f"Item {i}", depth + 1, max_depth)
            if len(obj) > 3:
                print(f"{indent}  [... and {len(obj) - 3} more items]")
    
    elif isinstance(obj, np.ndarray):
        print(f"{indent}  [NumPy array with shape {obj.shape} and dtype {obj.dtype}]")
    
    else:
        # For other objects, print attributes
        attributes = dir(obj)
        interesting_attrs = [attr for attr in attributes 
                           if not attr.startswith('__') and not callable(getattr(obj, attr))]
        
        if interesting_attrs:
            print(f"{indent}  [Has attributes: {', '.join(interesting_attrs[:10])}]")
            if len(interesting_attrs) > 10:
                print(f"{indent}  [... and {len(interesting_attrs) - 10} more attributes]")
            
            # Print values of some common interesting attributes
            common_attrs = ['classes_', 'n_classes_', 'n_neighbors', 'n_features_in_', 
                           'feature_names_in_', 'n_samples_seen_']
            for attr in common_attrs:
                if attr in interesting_attrs:
                    value = getattr(obj, attr)
                    inspect_object(value, f"Attribute: {attr}", depth + 1, max_depth)
        else:
            print(f"{indent}  [No interesting attributes]")

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py <model_path>")
        return
        
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found")
        return
        
    print(f"Inspecting model file: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        inspect_object(model, "Model")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()