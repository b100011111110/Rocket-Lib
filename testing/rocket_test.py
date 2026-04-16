import sys
import os

# Add the CMake build directory to Python's sys.path so we can import the compiled module
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
if build_dir not in sys.path:
    sys.path.append(build_dir)

try:
    import rocket
except ImportError as e:
    print(f"Failed to import rocket module: {e}")
    print(f"Please ensure you have built the project with CMake in the '{build_dir}' directory.")
    sys.exit(1)

def test_tensor():
    print("====================================")
    print("Testing Rocket Tensor functionalities")
    print("====================================")

    print("\n1. Creating t1 (2x3)...")
    t1 = rocket.Tensor(2, 3)
    t1.print()

    print("\n2. Creating t2 (3x2)...")
    t2 = rocket.Tensor(3, 2)
    t2.print()

    print("\n3. Matrix Multiplication (t3 = t1 * t2)...")
    t3 = t1 * t2
    t3.print()
    
    print("\n4. Matrix Addition (t4 = t3 + t3)...")
    t4 = t3 + t3
    t4.print()

    print("\n5. Unary Negation (t5 = -t4)...")
    t5 = -t4
    t5.print()
    
    print("\nAll tests completed seamlessly!")

if __name__ == "__main__":
    test_tensor()
