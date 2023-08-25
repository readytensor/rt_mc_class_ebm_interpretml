import logging
from typing import Any

from src.logger import close_handlers, get_logger, log_error


def test_get_logger(caplog: Any) -> None:
    """
    Tests the `get_logger` function.

    This function tests the creation of a logger object by the `get_logger` function.
    It checks that the logger has the correct level, name, and handlers.
    It also checks that a log message is correctly captured.

    Args:
        caplog (Any): A pytest fixture that captures log output.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    task_name = "Test task"

    # When
    logger = get_logger(task_name)

    # Then
    assert logger.level == logging.INFO
    assert logger.name == task_name
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

    # Log a message to test handlers
    logger.info("Test log message")

    assert "Test log message" in caplog.text

    # Close handlers
    close_handlers(logger)


def test_log_error(tmpdir: Any) -> None:
    """
    Tests the `log_error` function.

    This function tests the writing of an error message and traceback to an
    error file by the `log_error` function.
    It checks that the error message, exception, and traceback are correctly
    written to the error file.

    Args:
        tmpdir (Any): A pytest fixture that provides a temporary directory
        unique to the test invocation.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    message = "Test error message"
    error_file_path = tmpdir.join("error.log")

    # When
    try:
        raise Exception("Test Exception")
    except Exception as error:
        log_error(message, error, str(error_file_path))

    # Then
    with open(error_file_path, "r", encoding="utf-8") as file:
        error_msg = file.read()

    assert message in error_msg
    assert "Test Exception" in error_msg
    assert "Traceback" in error_msg
