"""Logging helper: console + file dual output with timestamp."""
from datetime import datetime


def log_info(msg, log_file=None, print_to_console=True):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_msg = f"[{timestamp}] {msg}"
    if print_to_console:
        print(log_msg)
    if log_file:
        log_file.write(log_msg + "\n")
