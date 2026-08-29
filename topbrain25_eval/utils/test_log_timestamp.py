from io import StringIO
from unittest.mock import patch

from log_timestamp import log_timestamp


def test_log_timestamp():
    # Mock time.time() to return a fixed value
    with patch("time.time", return_value=2024):
        last_timestamp = 2020  # Simulate last timestamp

        # Capture the print output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            current_timestamp = log_timestamp(last_timestamp, "Test operation 1")

            # Get the output printed by the function
            output = mock_stdout.getvalue()

            # Ensure the correct elapsed time is in the output
            assert "Test operation 1 took 4.0000 seconds" in output

            # Ensure the current_timestamp was updated correctly
            assert current_timestamp == 2024

    # Mock time.time() to return a fixed value
    with patch("time.time", return_value=2025):
        last_timestamp = 1001  # Simulate last timestamp

        # Capture the print output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            current_timestamp = log_timestamp(last_timestamp, "Test operation 2")

            # Get the output printed by the function
            output = mock_stdout.getvalue()

            # Ensure the correct elapsed time is in the output
            assert "Test operation 2 took 1024.0000 seconds" in output

            # Ensure the current_timestamp was updated correctly
            assert current_timestamp == 2025
