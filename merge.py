import sys
import signal

n_timeout = 0

class Timeout(Exception):
    pass

def _handle_timeout(signum, frame):
    raise Timeout("Function timed out")
signal.signal(signal.SIGALRM, _handle_timeout)
for i in range(10):
    signal.alarm(2)  # Set a timeout of 60 seconds for each iteration
    try:
        import time
        time.sleep(5)
    except Exception as e:
        n_timeout += 1
        print(f"Timeout {n_timeout}: {e}")
        if n_timeout > 3:
            print("Too many timeouts, exiting.")
            sys.exit(1)
        continue