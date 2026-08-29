import time


def log_timestamp(last_timestamp, msg):
    """log the elapsed time since last timestamp"""
    current_timestamp = time.time()

    # Calculate the elapsed time
    elapsed_time = current_timestamp - last_timestamp
    print(f"{msg} took {elapsed_time:.4f} seconds")

    return current_timestamp
