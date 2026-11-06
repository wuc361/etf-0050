"""Remove the retired standalone live-monitor link from the generated dashboard."""
from pathlib import Path

page = Path("site/index.html")
if not page.exists():
    raise SystemExit("site/index.html is missing")

html = page.read_text(encoding="utf-8")
html = html.replace(
    '<!-- LIVE_MONITOR_LINK --><p><a href="./live.html" style="font-weight:700">📡 開啟 0050 盤中即時監控（每 5 分鐘更新）</a></p>',
    "",
)
html = html.replace(
    '<p><a href="./live.html" style="font-weight:700">📡 開啟 0050 盤中即時監控（每 5 分鐘更新）</a></p>',
    "",
)
page.write_text(html, encoding="utf-8")
