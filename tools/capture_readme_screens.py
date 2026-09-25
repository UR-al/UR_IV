"""README 스크린샷 — 헤드리스 Chrome + CDP 로 개발용 목업(frontend/dev/theme-audit.html?readme)을 캡처한다.

실제 앱·백엔드·GPU·개인 파일을 쓰지 않는다. 목업의 README 모드가 합성 그림과 샘플 데이터를 넣는다.

1. 개발 서버를 띄운다:  cd frontend && npm run dev        (http://localhost:5173)
2. 저장소 루트에서:     venv\\Scripts\\python.exe tools\\capture_readme_screens.py [출력 폴더] [장면 ...]
   출력 폴더 기본값은 docs/images/readme. 장면을 주면 그것만 찍는다
   (t2i, t2i-params, search, gallery, pnginfo, editor, creator, chat, settings, wildcard-modal, t2i-light).
Chrome 경로는 환경 변수 CHROME_PATH 로 바꿀 수 있다.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

CHROME = os.environ.get("CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PORT = 9444
URL = "http://localhost:5173/dev/theme-audit.html?readme=1"
WIDTH, HEIGHT = 1600, 1000

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=120, suppress_origin=True)
        self.next_id = 0

    def call(self, method, **params):
        self.next_id += 1
        mid = self.next_id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr, await_promise=True):
        res = self.call("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JS error: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")


HIDE_HARNESS = """
(() => {
  const s = document.createElement('style');
  s.id = 'readme-capture-style';
  s.textContent = `#preview-status,#preview-tools,#preview-log,#preview-help{display:none!important}
    .toast-container{display:none!important}
    *{caret-color:transparent!important}`;
  document.head.appendChild(s);
  return true;
})()
"""

TOOL = """(async (label) => {
  const b = [...document.querySelectorAll('#preview-tools button')].find(x => x.textContent.trim() === label);
  if (!b) throw new Error('no harness button: ' + label);
  b.click();
  await new Promise(r => setTimeout(r, 900));
  return true;
})(%s)"""

APP_CLICK = """(async (text) => {
  const all = [...document.querySelectorAll('#app button, #app [role=tab], #app a')];
  const b = all.find(x => x.textContent.trim() === text) || all.find(x => x.textContent.trim().startsWith(text));
  if (!b) throw new Error('no app control: ' + text);
  b.click();
  await new Promise(r => setTimeout(r, 900));
  return true;
})(%s)"""


def tool(c, label):
    c.js(TOOL % json.dumps(label))


def app_click(c, text):
    c.js(APP_CLICK % json.dumps(text))


def esc(c):
    c.js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true})); new Promise(r=>setTimeout(()=>r(true),500))")


def shot(c, out_dir, name):
    time.sleep(0.6)
    data = c.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)["data"]
    path = os.path.join(out_dir, f"{name}.png")
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(data))
    print("saved", path)


def scenes(c, out_dir, wanted):
    def want(n):
        return not wanted or n in wanted

    tool(c, "default")
    tool(c, "T2I")
    tool(c, "샘플 이미지")   # README 모드: 히스토리를 합성 그림 6장으로 채운다
    if want("t2i"):
        shot(c, out_dir, "t2i")
    if want("t2i-params"):
        app_click(c, "파라미터")
        shot(c, out_dir, "t2i-params")
        esc(c)
    if want("search"):
        tool(c, "Search 샘플")
        shot(c, out_dir, "search")
    if want("gallery"):
        tool(c, "Gallery 샘플")
        shot(c, out_dir, "gallery")
    if want("pnginfo"):
        tool(c, "PNG Info 샘플")
        shot(c, out_dir, "pnginfo")
    if want("editor"):
        tool(c, "Editor")
        tool(c, "샘플 이미지")
        # 목업은 그림을 data URL 로 넘겨 파일명 자리에 그 문자열이 찍힌다 — 캡처에서만 샘플 이름으로 바꾼다
        c.js("(() => { const el = document.querySelector('.bar-filename'); if (el) { el.childNodes.forEach(n => { if (n.nodeType === 3 && n.textContent.trim()) n.textContent = 'sample_meadow.png' }); if (!el.textContent.includes('sample_meadow')) el.firstChild && (el.firstChild.textContent = 'sample_meadow.png') } return true })()", await_promise=False)
        shot(c, out_dir, "editor")
    if want("creator"):
        tool(c, "Creator")
        shot(c, out_dir, "creator")
    if want("chat"):
        tool(c, "대화")
        shot(c, out_dir, "chat")
    if want("settings"):
        tool(c, "Settings")
        shot(c, out_dir, "settings")
    if want("wildcard-modal"):
        tool(c, "와일드카드 모달")
        c.js("""(async () => {
          const item = [...document.querySelectorAll('#app *')].find(e => e.children.length === 0 && e.textContent.trim() === 'hairstyle');
          if (item) item.click();
          await new Promise(r => setTimeout(r, 700));
          return !!item;
        })()""")
        shot(c, out_dir, "wildcard-modal")
        esc(c)
    if want("t2i-light"):
        tool(c, "light")
        tool(c, "T2I")
        shot(c, out_dir, "t2i-light")
        tool(c, "default")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = sys.argv[1:]
    # 첫 인자가 경로처럼 생겼으면(구분자 포함 또는 이미 있는 폴더) 출력 폴더, 아니면 장면 이름이다
    if args and (os.sep in args[0] or "/" in args[0] or os.path.isdir(args[0])):
        out_dir = os.path.abspath(args.pop(0))
    else:
        out_dir = os.path.join(root, "docs", "images", "readme")
    wanted = set(args)
    os.makedirs(out_dir, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="readme_chrome_")
    proc = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
        f"--window-size={WIDTH},{HEIGHT}", "--hide-scrollbars", "--force-device-scale-factor=1",
        "--no-first-run", "--no-default-browser-check", "--lang=ko-KR", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2) as r:
                    pages = [p for p in json.loads(r.read().decode("utf-8")) if p.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.5)
        if not ws_url:
            raise SystemExit("Chrome CDP 에 연결하지 못했습니다")
        c = CDP(ws_url)
        c.call("Page.enable")
        c.call("Runtime.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)
        c.call("Page.navigate", url=URL)
        state = None
        for _ in range(120):
            state = c.js("document.querySelector('#preview-status') && document.querySelector('#preview-status').dataset.state", await_promise=False)
            if state in ("ready", "error"):
                break
            time.sleep(0.5)
        print("harness state:", state)
        if state != "ready":
            print("status:", c.js("document.querySelector('#preview-status').textContent", await_promise=False))
            raise SystemExit(1)
        c.js(HIDE_HARNESS, await_promise=False)
        scenes(c, out_dir, wanted)
        errors = c.js("document.querySelector('#preview-status').dataset.state", await_promise=False)
        print("final harness state:", errors)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
