"""Add a discoverable link to the live monitor in the generated Pages build."""
from pathlib import Path

page = Path("site/index.html")
if not page.exists():
    raise SystemExit("site/index.html is missing")
html = page.read_text(encoding="utf-8")
marker = '<!-- LIVE_MONITOR_LINK -->'
if marker not in html:
    link = (
        '\n    <!-- LIVE_MONITOR_LINK -->'
        '<p><a href="./live.html" style="font-weight:700">📡 開啟 0050 盤中即時監控（每 5 分鐘更新）</a></p>\n'
    )
    if "</header>" not in html:
        raise SystemExit("Could not find the dashboard header insertion point")
    html = html.replace("</header>", link + "</header>", 1)
    page.write_text(html, encoding="utf-8")
