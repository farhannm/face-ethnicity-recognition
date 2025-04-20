import pickle
import os
import sys
import numpy as np

def convert_model(input_path, output_path, class_names=None):
    """
    Convert a model to a format compatible with the ethnic classification app
    
    Args:
        input_path (str): Path to the input model file
        output_path (str): Path to save the converted model
        class_names (list, optional): List of class names
    """
    print(f"Loading model from: {input_path}")
    
    try:
        # Load the model
        with open(input_path, 'rb') as f:
            model = pickle.load(f)
            
        # Determine the structure of the model
        if isinstance(model, dict):
            print("Model is already a dictionary.")
            
            # Check if it has the required keys
            if 'model' in model and 'classes' in model:
                print("Model already has the required structure.")
                converted_model = model
            else:
                print("Converting model dictionary to required format...")
                # Create a new dictionary with the required structure
                converted_model = {
                    'model': model,
                    'classes': class_names or get_class_names(model)
                }
        else:
            print("Converting model to dictionary format...")
            # Create a dictionary with the model and class names
            converted_model = {
                'model': model,
                'classes': class_names or get_class_names(model)
            }
            
        # Print information about the converted model
        print("\nConverted Model Information:")
        print(f"- Type: {type(converted_model)}")
        print(f"- Keys: {list(converted_model.keys())}")
        print(f"- Classes: {converted_model.get('classes')}")
        
        # Save the converted model
        with open(output_path, 'wb') as f:
            pickle.dump(converted_model, f)
            
        print(f"\nConverted model saved to: {output_path}")
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return False
        
    return True

def get_class_names(model):
    """
    Try to extract class names from a model
    
    Args:
        model: The model object
        
    Returns:
        list: List of class names
    """
    # Try to get class names from the model
    if hasattr(model, 'classes_'):
        return list(model.classes_)
    
    # If the model is a dictionary, try to find classes in it
    if isinstance(model, dict):
        if 'classes' in model:
            return model['classes']
        if 'class_names' in model:
            return model['class_names']
            
        # Try to find a key that might contain a classifier
        for key, value in model.items():
            if hasattr(value, 'classes_'):
                return list(value.classes_)
    
    # If no class names found, create generic ones
    if hasattr(model, 'n_classes_'):
        n_classes = model.n_classes_
    elif hasattr(model, '_n_classes'):
        n_classes = model._n_classes
    else:
        # Guess a reasonable number of classes
        n_classes = 5
        
    print(f"Warning: Could not find class names, creating generic ones for {n_classes} classes.")
    return [f"Class_{i}" for i in range(n_classes)]

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python convert_model.py <input_model_path> <output_model_path> [class1 class2 ...]")
        return
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return
        
    # Get class names from command line if provided
    class_names = None
    if len(sys.argv) > 3:
        class_names = sys.argv[3:]
        print(f"Using provided class names: {class_names}")
    
    # Convert the model
    if convert_model(input_path, output_path, class_names):
        print("\nModel conversion completed successfully!")
        print("\nYou can now use this model with the ethnic classification app.")
    else:
        print("\nModel conversion failed!")

if __name__ == "__main__":
    main()