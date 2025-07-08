import time
import glob
import os
from datetime import datetime

def cleanup():
    pattern = "/home/zyang2/XViper/logdir/*/XAIL/*/replay/*"
    files = glob.glob(pattern, recursive=True)

    deleted_files = 0
    deleted_dirs = 0

    for path in files:
        try:
            if os.path.isfile(path):
                os.remove(path)
                deleted_files += 1
            elif os.path.isdir(path):
                os.rmdir(path)  # only removes if empty
                deleted_dirs += 1
        except Exception as e:
            print(f"[ERROR] Could not delete: {path}\nReason: {e}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Cleanup complete: {deleted_files} files and {deleted_dirs} empty folders removed.\n")

def countdown(minutes):
    total_secs = minutes * 60
    while total_secs > 0:
        mins, secs = divmod(total_secs, 60)
        timer_str = f"Next cleanup in {mins:02d}:{secs:02d}"
        print(timer_str, end='\r', flush=True)
        time.sleep(1)
        total_secs -= 1

if __name__ == "__main__":
    print("🚀 Replay cleaner started. Running every 30 minutes.")
    while True:
        cleanup()
        countdown(30)

