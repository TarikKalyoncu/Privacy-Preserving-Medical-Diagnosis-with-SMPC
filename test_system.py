"""
SMPC System Test and Analysis Code (Detailed Explanation)
This file runs the main system (smpc_system.py), measures its speed and accuracy.
"""

# Importing necessary libraries
import numpy as np          # For numerical operations and calculating averages
import matplotlib.pyplot as plt # For plotting graphs (Barchart, Histogram, etc.)
import time                 # To measure processing times (in seconds)
# We import our main system class here.
# NOTE: The 'smpc_system.py' file must be in the same folder as this file!
from smpc_system import PrivacyPreservingDiagnosisSystem

def test_accuracy(num_tests=50):
    """
    FUNCTION 1: Accuracy Test
    Purpose: To measure if the system makes correct predictions on 50 different random patients.
    The most important feature of SMPC systems is that the result is correct despite the encryption.
    """
    print("\n" + "="*60)
    print("ACCURACY TEST")
    print("="*60)
    
    # 1. Initialize the System
    system = PrivacyPreservingDiagnosisSystem()
    system.setup_encryption() # Generate encryption keys
    X_test, y_test = system.train_model() # Train the model and get test data
    
    correct = 0 # Counter for correct predictions
    times = []  # List to store how long each operation takes
    
    # 2. Select 50 random patients (Pick random indices from the dataset)
    # replace=False: Do not select the same patient twice.
    test_indices = np.random.choice(len(X_test), num_tests, replace=False)
    
    # 3. Loop for each patient
    for i, idx in enumerate(test_indices):
        print(f"\nTest {i+1}/{num_tests}...", end="")
        
        patient_data = X_test[idx] # The patient's data
        true_label = y_test[idx]   # The actual condition (Cancer or not?)
        
        start = time.time() # Start the stopwatch
        
        # --- SMPC PROCESS STARTS ---
        # A) Encrypt
        encrypted_patient = system.encrypt_patient_data(patient_data)
        # B) Perform encrypted prediction at the Hospital
        encrypted_prediction = system.encrypted_inference(encrypted_patient)
        # C) Decrypt the result
        prediction = system.decrypt_result(encrypted_prediction)
        # ----------------------------
        
        elapsed = time.time() - start # Calculate elapsed time
        times.append(elapsed) # Add time to the list
        
        # 4. Accuracy Check
        # If prediction > 0.5 it is considered "Positive" (1), otherwise "Negative" (0).
        # We check if it matches the true_label.
        if (prediction > 0.5) == true_label:
            correct += 1
            print(" ✓") # Correct guess
        else:
            print(" ✗") # Wrong guess
    
    # 5. Calculate Statistics
    accuracy = correct / num_tests * 100 # What percentage did we get right?
    avg_time = np.mean(times) # What is the average processing time?
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total Tests: {num_tests}")
    print(f"Correct Predictions: {correct}")
    print(f"ACCURACY: {accuracy:.2f}%") # This rate should be above 90%
    print(f"Average Processing Time: {avg_time:.4f} seconds")
    
    return accuracy, avg_time, times


def compare_encrypted_vs_plain():
    """
    FUNCTION 2: Performance Comparison
    Purpose: To show the speed difference between encrypted processing (SMPC) and normal processing (Plaintext).
    This answers the professor's question: "How much speed do we sacrifice for security?"
    """
    print("\n" + "="*60)
    print("ENCRYPTED vs NORMAL PROCESSING COMPARISON")
    print("="*60)
    
    system = PrivacyPreservingDiagnosisSystem()
    system.setup_encryption()
    X_test, y_test = system.train_model()
    
    # Select a single random patient
    patient_data = X_test[0]
    
    # --- SCENARIO A: NORMAL PREDICTION (Plaintext) ---
    print("\n[A] NORMAL PREDICTION (Unencrypted):")
    start = time.time()
    
    # Scale the data (Scaler)
    patient_scaled = system.scaler.transform(patient_data.reshape(1, -1))[0]
    
    # Simple Math: (Data x Weights) + Bias
    # numpy.dot() function takes less than a millisecond.
    plain_result = np.dot(system.weights, patient_scaled) + system.bias
    plain_prediction = 1 / (1 + np.exp(-plain_result)) # Sigmoid
    
    plain_time = time.time() - start
    print(f"  Duration: {plain_time:.6f} seconds")
    print(f"  Result: {plain_prediction*100:.2f}%")
    
    # --- SCENARIO B: ENCRYPTED PREDICTION (SMPC) ---
    print("\n[B] ENCRYPTED PREDICTION (SMPC):")
    start = time.time()
    
    # 1. Encrypt -> 2. Compute -> 3. Decrypt
    encrypted_patient = system.encrypt_patient_data(patient_data)
    encrypted_prediction = system.encrypted_inference(encrypted_patient)
    encrypted_result = system.decrypt_result(encrypted_prediction)
    
    encrypted_time = time.time() - start
    print(f"  Duration: {encrypted_time:.6f} seconds")
    print(f"  Result: {encrypted_result*100:.2f}%")
    
    # --- COMPARISON ---
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    # Calculate how many times slower it is
    print(f"Time Difference: {encrypted_time/plain_time:.2f}x slower")
    
    # Mathematical difference between results (Error margin)
    # This difference should be very small, like 0.0001.
    print(f"Result Difference: {abs(plain_prediction - encrypted_result)*100:.4f}%")
    print("\n💡 NOTE: Encrypted processing is slow but guarantees PRIVACY!")
    
    return plain_time, encrypted_time


def plot_performance_analysis():
    """
    FUNCTION 3: Plotting Graphs
    Purpose: To create the beautiful graphs you will put in your report.
    """
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE ANALYSIS GRAPHS")
    print("="*60)
    
    # Run previous test functions to collect data
    accuracy, avg_time, times = test_accuracy(30)
    plain_time, encrypted_time = compare_encrypted_vs_plain()
    
    # Create graphing area (size 12x5)
    plt.figure(figsize=(12, 5))
    
    # --- GRAPH 1: HISTOGRAM (Left Graph) ---
    # Shows the distribution of processing times.
    plt.subplot(1, 2, 1) # 1st plot in a 1 row, 2 column grid
    plt.hist(times, bins=15, color='skyblue', edgecolor='black')
    # Show average time with a red dashed line
    plt.axvline(avg_time, color='red', linestyle='--', 
                label=f'Average: {avg_time:.4f}s')
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Encrypted Processing Time Distribution')
    plt.legend()
    plt.grid(alpha=0.3) # Grid lines
    
    # --- GRAPH 2: BAR CHART (Right Graph) ---
    # Compares Encrypted and Plaintext processing times side by side.
    plt.subplot(1, 2, 2) # 2nd plot in a 1 row, 2 column grid
    methods = ['Normal\n(Plaintext)', 'SMPC\n(Encrypted)']
    times_comparison = [plain_time, encrypted_time]
    colors = ['lightgreen', 'salmon']
    
    bars = plt.bar(methods, times_comparison, color=colors, edgecolor='black')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Encrypted vs Normal Processing Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    # Write second values on top of the bars
    for bar, time_val in zip(bars, times_comparison):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight') # Save to file
    print("\n✓ Graph saved: performance_analysis.png")
    plt.show() # Show on screen


def security_analysis():
    """
    FUNCTION 4: Security Analysis
    Purpose: To prove how much the data grows and becomes complex when encrypted.
    """
    print("\n" + "="*60)
    print("SECURITY ANALYSIS")
    print("="*60)
    
    system = PrivacyPreservingDiagnosisSystem()
    system.setup_encryption()
    system.train_model()
    
    # Generate random data with 10 numbers
    patient_data = np.random.randn(10)
    
    # Encrypt
    encrypted = system.encrypt_patient_data(patient_data)
    
    print("\nORIGINAL DATA (First 5 features):")
    print(patient_data[:5]) # Numbers are readable like 0.5, -1.2.
    
    print("\nENCRYPTED DATA (Binary format, first 100 bytes):")
    # serialize() converts data to a byte array.
    encrypted_bytes = encrypted.serialize()
    print(encrypted_bytes[:100]) # Output: Complex, unreadable characters (\x04\xe2...)
    
    print("\nSECURITY FEATURES:")
    print(f"  ✓ Encryption Scheme: CKKS (Homomorphic)")
    print(f"  ✓ Polynomial Modulus Degree: {system.poly_mod_degree} bit")
    print(f"  ✓ Encrypted data size: {len(encrypted_bytes)} bytes")
    print(f"  ✓ Original data size: {patient_data.nbytes} bytes")
    # Calculate how much the size increased (Expansion Ratio)
    print(f"  ✓ Size Expansion: {len(encrypted_bytes)/patient_data.nbytes:.2f}x")
    
    print("\n🔒 SECURITY GUARANTEE:")
    print("  - No one can read the encrypted data")
    print("  - Only the patient has the decryption key")
    print("  - The hospital cannot access the data, only process it")


def main():
    """Main program: Runs all tests sequentially"""
    
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "SMPC SYSTEM TEST SUITE" + " "*24 + "║")
    print("╚" + "="*58 + "╝")
    
    # Call functions sequentially
    test_accuracy(50)           # 1. Accuracy
    compare_encrypted_vs_plain() # 2. Performance
    security_analysis()         # 3. Security
    plot_performance_analysis() # 4. Visualization
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED ✓")
    print("="*60)


if __name__ == "__main__":
    main()