import os
import sys
import argparse

# Add src to the Python path so it works seamlessly from the IDE
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

def main():
    parser = argparse.ArgumentParser(description="Run ForensTatt Core CBIR")
    parser.add_argument("--tui", action="store_true", help="Start the interactive Text User Interface (TUI)")
    
    args = parser.parse_args()
    
    if args.tui:
        from tui.app import run_tui
        run_tui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
