import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Football Match Prediction Application',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='medium',
        choices=['simple', 'medium', 'full'],
        help='''Choose application mode:
simple: Select from existing matches in test set
medium: Select teams and date (players from similar matches)
full: Select teams, players, and date manually
'''
    )
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        print("Starting Simple Mode: Select from existing matches")
        os.system(f"{sys.executable} app_simple.py")
    elif args.mode == 'medium':
        print("Starting Medium Mode: Select teams and date")
        os.system(f"{sys.executable} app_medium.py")
    elif args.mode == 'full':
        print("Starting Full Mode: Select teams, players, and date")
        os.system(f"{sys.executable} app.py")
    else:
        print("Invalid mode selected. Please choose from: simple, medium, full")

if __name__ == "__main__":
    main()
