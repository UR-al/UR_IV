"""utils.app_logger — 서드파티 소음 억제, 로그 경로 격리, 날짜 있는 파일 줄."""
from __future__ import annotations

import io
import logging
import os
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from unittest import mock

from utils import app_logger


class FileHandlerFallbackTests(unittest.TestCase):
    """로그 파일을 열지 못하면 파일 핸들러 없이 콘솔로만 — 기록마다 'Logging error' 가 쏟아지지 않게.

    예전 핸들러는 delay=True 라 생성자가 파일을 열지 않았고, except OSError 는 makedirs 실패만 잡았다.
    열기 실패는 첫 기록 때 터져 logging.handleError 가 기록마다 PermissionError 트레이스백을 찍었다."""

    def _fresh_logger(self) -> logging.Logger:
        logger = logging.Logger(f"tests.app_logger.fresh.{id(self)}")   # 등록되지 않은 독립 로거
        self.addCleanup(self._close_handlers, logger)
        return logger

    @staticmethod
    def _close_handlers(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def test_directory_path_adds_no_file_handler_and_prints_one_notice(self):
        with tempfile.TemporaryDirectory() as temp:
            target = os.path.join(temp, "app.log")
            os.mkdir(target)   # 로그 파일 자리에 디렉터리가 있다 — 열기가 실패한다
            logger = self._fresh_logger()
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
                app_logger.configure_root(logger, target)
                logger.info("console only")
                logger.debug("debug line")
            self.assertFalse(any(isinstance(h, logging.FileHandler) for h in logger.handlers))
            self.assertEqual(1, out.getvalue().count("[Log] 로그 파일을 열지 못해"), out.getvalue())
            self.assertIn("console only", err.getvalue())
            self.assertNotIn("Logging error", err.getvalue())
            self._close_handlers(logger)

    def test_build_file_handler_opens_the_file_eagerly(self):
        with tempfile.TemporaryDirectory() as temp:
            target = os.path.join(temp, "logs", "app.log")
            handler = app_logger.build_file_handler(target)
            try:
                self.assertIsInstance(handler, RotatingFileHandler)
                self.assertIsNotNone(handler.stream, "지연 열기면 실패가 첫 기록 때로 밀린다")
                self.assertTrue(os.path.isfile(target))
            finally:
                handler.close()

    def test_build_file_handler_returns_none_for_a_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                self.assertIsNone(app_logger.build_file_handler(temp))
            self.assertIn(temp, out.getvalue())


class ResolveLogFileTests(unittest.TestCase):
    def test_default_is_repository_app_log(self):
        self.assertEqual(app_logger.resolve_log_file({}), app_logger._LOG_FILE)
        self.assertEqual(app_logger.resolve_log_file({app_logger.LOG_FILE_ENV: "   "}), app_logger._LOG_FILE)

    def test_environment_override_is_made_absolute(self):
        resolved = app_logger.resolve_log_file({app_logger.LOG_FILE_ENV: os.path.join("logs", "x.log")})
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.endswith(os.path.join("logs", "x.log")))


class RootLoggerTests(unittest.TestCase):
    def setUp(self):
        app_logger.get_logger("tests.app_logger")   # 한 번만 초기화된다(이미 됐으면 그대로)
        self.file_handlers = [
            h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)
        ]

    def test_third_party_noise_is_raised_but_root_stays_debug(self):
        self.assertEqual(logging.getLogger().level, logging.DEBUG)
        self.assertEqual(logging.getLogger("PIL").level, logging.INFO)
        self.assertEqual(logging.getLogger("urllib3").level, logging.WARNING)
        # PIL 하위 로거(PngImagePlugin)의 DEBUG 는 걸러지고, 앱 로거의 DEBUG 는 통과한다.
        self.assertFalse(logging.getLogger("PIL.PngImagePlugin").isEnabledFor(logging.DEBUG))
        self.assertFalse(logging.getLogger("urllib3.connectionpool").isEnabledFor(logging.INFO))
        self.assertTrue(logging.getLogger("core.some_module").isEnabledFor(logging.DEBUG))

    def test_tests_do_not_write_to_the_production_app_log(self):
        self.assertTrue(self.file_handlers, "루트에 파일 핸들러가 없다")
        production = os.path.normcase(os.path.abspath(app_logger._LOG_FILE))
        for handler in self.file_handlers:
            self.assertNotEqual(os.path.normcase(handler.baseFilename), production,
                                "테스트가 운영 app.log 에 기록한다 — tests/__init__.py 의 AISTUDIO_LOG_FILE 확인")
            self.assertEqual(os.path.normcase(handler.baseFilename),
                             os.path.normcase(app_logger.resolve_log_file()))

    def test_file_lines_carry_the_date(self):
        handler = self.file_handlers[0]
        self.assertEqual(handler.formatter.datefmt, app_logger.FILE_DATEFMT)
        self.assertIn("%Y-%m-%d", handler.formatter.datefmt)


if __name__ == "__main__":
    unittest.main()
