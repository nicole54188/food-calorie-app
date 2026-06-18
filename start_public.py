"""
start_public.py — 一鍵產生 HTTPS 公開網址（任何地方都能連）
================================================================
使用 Cloudflare Quick Tunnel（完全免費、不需要帳號）
自動下載 cloudflared.exe → 啟動伺服器 → 產生 HTTPS 網址

執行方式：
    python start_public.py

第一次執行會自動下載 cloudflared.exe（約 30MB），之後不需重複下載。
"""

import subprocess
import sys
import os
import time
import re
import threading
import urllib.request

# ── 路徑設定 ──
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT   = os.path.join(BASE_DIR, 'server.py')
CLOUDFLARED_EXE = os.path.join(BASE_DIR, 'cloudflared.exe')

# Cloudflare 官方下載網址（Windows 64 位元）
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-windows-amd64.exe"
)

def download_cloudflared():
    """自動下載 cloudflared.exe（如果還沒有的話）"""
    if os.path.exists(CLOUDFLARED_EXE):
        return  # 已存在，不重複下載

    print("📦 第一次使用，自動下載 cloudflared.exe（約 30MB）...")
    print("   下載來源：Cloudflare 官方 GitHub Releases")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
            print(f"\r   [{bar}] {pct:.0f}%", end='', flush=True)

    try:
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_EXE, show_progress)
        print("\n✅ cloudflared.exe 下載完成！")
    except Exception as e:
        print(f"\n❌ 下載失敗：{e}")
        print("   請手動下載：")
        print(f"   {CLOUDFLARED_URL}")
        print(f"   存到：{CLOUDFLARED_EXE}")
        sys.exit(1)

def start_flask_server():
    """在背景啟動 Flask 伺服器（server.py）"""
    # CREATE_NEW_CONSOLE：開一個新的 CMD 視窗跑伺服器，方便查看 log
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    proc = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT],
        creationflags=flags
    )
    return proc

def start_tunnel():
    """
    啟動 Cloudflare Quick Tunnel，持續讀取輸出找到 HTTPS 網址

    cloudflared 會把網址印在 stderr，格式像：
      INF | https://random-name.trycloudflare.com |
    或
      INF Connection established connIndex=0 url=https://xxx.trycloudflare.com
    """
    proc = subprocess.Popen(
        [CLOUDFLARED_EXE, 'tunnel', '--url', 'http://localhost:8080'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    url_found = threading.Event()
    public_url = [None]

    def read_output(stream):
        """讀取 cloudflared 輸出，抓出 HTTPS 網址"""
        for raw_line in stream:
            try:
                line = raw_line.decode('utf-8', errors='ignore')
            except Exception:
                continue

            # 用正則表達式找 trycloudflare.com 的網址
            match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
            if match and not url_found.is_set():
                public_url[0] = match.group(0)
                url_found.set()

    # 同時讀 stdout 和 stderr（cloudflared 版本不同，輸出位置不同）
    t1 = threading.Thread(target=read_output, args=(proc.stdout,), daemon=True)
    t2 = threading.Thread(target=read_output, args=(proc.stderr,), daemon=True)
    t1.start()
    t2.start()

    # 等待最多 30 秒讓 URL 出現
    if url_found.wait(timeout=30):
        return proc, public_url[0]
    else:
        proc.terminate()
        return None, None

def main():
    print("=" * 60)
    print("  🍽️  食物熱量估算系統 — HTTPS 公開分享模式")
    print("=" * 60)

    # 步驟 1：下載 cloudflared（第一次才需要）
    download_cloudflared()

    # 步驟 2：啟動 Flask 後端
    print("\n⚙️  啟動 Flask 後端伺服器（新視窗）...")
    flask_proc = start_flask_server()
    print("   等待模型載入（約 3 秒）...")
    time.sleep(4)

    # 步驟 3：建立 Cloudflare HTTPS 隧道
    print("🌐 建立 Cloudflare HTTPS 隧道中...")
    cf_proc, public_url = start_tunnel()

    if public_url:
        print("\n" + "=" * 60)
        print("  ✅ 公開 HTTPS 網址已建立！")
        print()
        print(f"  👉  {public_url}")
        print()
        print("  把上面的網址傳給同學或老師")
        print("  任何地方、手機或電腦瀏覽器都能直接開啟")
        print("  網址是 HTTPS，安全加密連線 🔒")
        print("=" * 60)
        print("\n  📌 關閉這個視窗或按 Ctrl+C 即可停止分享\n")

        try:
            while True:
                time.sleep(1)
                # 如果 cloudflared 意外結束，提示使用者
                if cf_proc.poll() is not None:
                    print("⚠️  隧道已斷線，請重新執行 start_public.py")
                    break
        except KeyboardInterrupt:
            print("\n\n🛑 停止分享中...")
            if cf_proc:
                cf_proc.terminate()
            print("✅ 已停止。網址已失效。")
    else:
        print("\n❌ 無法取得公開網址，請確認：")
        print("   1. 電腦有連上網路")
        print("   2. server.py 正在執行（應有新視窗開啟）")
        print("   3. 防火牆沒有封鎖 cloudflared.exe")
        print("\n   也可以改用手動指令：")
        print(f"   {CLOUDFLARED_EXE} tunnel --url http://localhost:8080")

if __name__ == '__main__':
    main()
