#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def main():
    # Ensure project root is on sys.path so we can import `deepfake_detector`
    root_dir = Path(__file__).resolve().parents[2]
    sys.path.append(str(root_dir))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_web.settings')
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
