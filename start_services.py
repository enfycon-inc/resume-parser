import subprocess
import socket
import os
import sys
import time

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_in_new_window(title, command):
    """Starts a command in a new command prompt window (Windows specific)."""
    # Using 'start' command to open a new window with a specific title
    full_command = f'start "{title}" cmd /k "{command}"'
    subprocess.Popen(full_command, shell=True)

def main():
    print("--- ATS Service Manager (Auto-Restart Mode) ---")
    
    # 0. Kill existing ATS windows to avoid port conflicts
    print("[!] Closing old ATS windows...")
    try:
        # We target the specific window titles we created earlier
        subprocess.run('taskkill /F /FI "WINDOWTITLE eq ATS-API*" /T', shell=True, stderr=subprocess.DEVNULL)
        subprocess.run('taskkill /F /FI "WINDOWTITLE eq ATS-Worker*" /T', shell=True, stderr=subprocess.DEVNULL)
        time.sleep(1) # Give OS time to release ports
    except:
        pass

    # 1. Check Redis
    if is_port_in_use(6379):
        print("[OK] Redis is already running on port 6379.")
    else:
        print("[!] Redis is NOT running.")
        print("    Attempting to start redis-server in a new window...")
        run_in_new_window("ATS-Redis", "redis-server")
        time.sleep(2) # Give it a moment to start

    # 2. Start FastAPI (Force fresh start)
    print("[+] Starting FastAPI...")
    run_in_new_window("ATS-API", f'"{sys.executable}" main.py')

    # 3. Start Celery Worker (Force fresh start)
    print("[+] Starting Celery Worker...")
    run_in_new_window("ATS-Worker", f'"{sys.executable}" -m celery -A tasks worker --loglevel=info -P solo')

    print("\n[SUCCESS] All services restarted!")
    print("---------------------------")

if __name__ == "__main__":
    # Check if we are on Windows
    if os.name != 'nt':
        print("This script is designed for Windows (uses 'start' command).")
        sys.exit(1)
    main()
