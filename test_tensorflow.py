"""
Simple script to test TensorFlow installation.
Run with: python test_tensorflow.py
"""

def test_tensorflow():
    print("Testing TensorFlow installation...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow successfully imported. Version: {tf.__version__}")
        
        # Test if TensorFlow can access the GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {gpus}")
        else:
            print("⚠️ No GPU detected. Using CPU for computation.")
        
        # Test basic TensorFlow operations
        print("\nTesting basic TensorFlow operations:")
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result:\n{c.numpy()}")
        
        # Test Keras
        try:
            from tensorflow import keras
            print(f"\n✅ Keras successfully imported. Version: {keras.__version__}")
            
            # Create a simple model
            model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(784,)),
                keras.layers.Dense(10, activation='softmax')
            ])
            
            # Compile the model
            model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
            
            print("✅ Keras model successfully created and compiled.")
            
        except ImportError as e:
            print(f"❌ Error importing Keras: {e}")
        
        return True
    
    except ImportError as e:
        print(f"❌ Error importing TensorFlow: {e}")
        print("\nPossible solutions:")
        print("1. For macOS with Apple Silicon (M1/M2/M3):")
        print("   pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.0")
        print("2. For other systems:")
        print("   pip install tensorflow==2.13.0")
        return False
    except Exception as e:
        print(f"❌ Unexpected error testing TensorFlow: {e}")
        return False

if __name__ == "__main__":
    success = test_tensorflow()
    
    if success:
        print("\n✅ TensorFlow test completed successfully! You're ready to run the application.")
    else:
        print("\n❌ TensorFlow test failed. Please fix the installation issues before running the application.")