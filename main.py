import os
import sys
import argparse

# Add src to the Python path so it works seamlessly from the IDE
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

def main():
    parser = argparse.ArgumentParser(description="Run ForensTatt Core CBIR")
    parser.add_argument("--tui", action="store_true", help="Start the interactive Text User Interface (TUI)")
    
    subparsers = parser.add_subparsers(dest="command")
    
    clear_parser = subparsers.add_parser("clear", help="Clear results and data folders")
    clear_parser.add_argument("--targets", nargs="+", default=["all"], help="Specific targets to clear (e.g. data charts)")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.command == "clear":
        from core.cleaner import WorkspaceCleaner
        if not args.force:
            targets_str = ", ".join(args.targets)
            confirm = input(f"WARNING: This will permanently delete files in: {targets_str}. Are you sure? [y/N]: ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return
        WorkspaceCleaner.clean(args.targets)
    elif args.tui:
        from tui.app import run_tui
        run_tui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
