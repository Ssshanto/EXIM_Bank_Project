#!/usr/bin/env python3
"""
Script to run the EXIM Bank Project Streamlit application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    print("Local URL: http://localhost:8501")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "interface.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main() 