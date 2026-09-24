"""safe_print — 진단 로그가 콘솔 인코딩 때문에 예외를 내지 않는다.

회귀 방지 대상
  · 한국어 Windows에서 stdout이 파이프·파일이면 cp949라 print('—')가 UnicodeEncodeError를
    내고, except 블록 안의 로그가 정리 코드(알림·VRAM 반납)를 건너뛰던 문제 (sam_refiner)
"""
import contextlib
import io
import unittest

from core.safe_print import safe_print


def _strict_console(encoding='cp949'):
    """한국어 Windows에서 리다이렉트된 stdout과 같은 strict 인코딩 스트림."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors='strict', newline='\n')


def _written(stream) -> bytes:
    stream.flush()
    return stream.buffer.getvalue()


class SafePrintTests(unittest.TestCase):
    def test_unencodable_characters_are_escaped_not_raised(self):
        console = _strict_console()
        with contextlib.redirect_stdout(console):
            with self.assertRaises(UnicodeEncodeError):
                print('[SAM] Error: x \u2014 fallback')       # 원래 print는 여기서 터진다
            safe_print('[SAM] Error: x \u2014 SAM 정밀화 없이 \u2713')
        text = _written(console).decode('cp949')
        self.assertEqual(text, '[SAM] Error: x \\u2014 SAM 정밀화 없이 \\u2713\n',
                         "한글은 그대로, cp949에 없는 문자만 \\uXXXX 로")

    def test_matches_print_formatting(self):
        buffer = io.StringIO()
        safe_print('a', 1, None, sep='|', end='!\n', file=buffer)
        safe_print(file=buffer)
        safe_print('x', 'y', sep=None, end=None, file=buffer)
        self.assertEqual(buffer.getvalue(), 'a|1|None!\n\nx y\n')

    def test_writes_each_line_once_even_after_an_encoding_error(self):
        console = _strict_console()
        safe_print('ok', '\u2014', 'tail', file=console)
        self.assertEqual(_written(console).decode('cp949'), 'ok \\u2014 tail\n',
                         "앞 인자가 먼저 써진 뒤 다시 쓰여 중복되면 안 된다")

    def test_lone_surrogates_on_utf8_stream(self):
        console = _strict_console('utf-8')
        safe_print('name-\udcff.png', file=console)
        self.assertEqual(_written(console).decode('utf-8'), 'name-\\udcff.png\n')

    def test_missing_or_broken_stream_is_ignored(self):
        with contextlib.redirect_stdout(None):
            safe_print('no console (pythonw)')
        closed = io.StringIO()
        closed.close()
        safe_print('closed', file=closed)

        class Broken:
            encoding = 'no-such-codec'

            def write(self, _text):
                raise UnicodeEncodeError('cp949', 'x', 0, 1, 'boom')

        safe_print('\u2014', file=Broken())               # 재시도도 실패해도 조용히 끝난다

    def test_unprintable_argument_is_ignored(self):
        class Bad:
            def __str__(self):
                raise RuntimeError('no str')

        buffer = io.StringIO()
        safe_print('before', Bad(), file=buffer)
        self.assertEqual(buffer.getvalue(), '')

    def test_flush(self):
        class Recorder(io.StringIO):
            flushed = 0

            def flush(self):
                self.flushed += 1
                super().flush()

        stream = Recorder()
        safe_print('x', file=stream, flush=True)
        self.assertEqual(stream.flushed, 1)


if __name__ == '__main__':
    unittest.main()
