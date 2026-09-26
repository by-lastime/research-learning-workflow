#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html as html_module
import json
import re
import time
from pathlib import Path
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from urllib.parse import urlparse

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "01_论文原文"
WEB_DIR = ROOT / "02_网页原始内容"
TEXT_DIR = ROOT / "03_可检索正文"
INDEX_DIR = ROOT / "00_索引"
for directory in (PDF_DIR, WEB_DIR, TEXT_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ("01_attention", "Attention Is All You Need", "https://arxiv.org/abs/1706.03762", "paper", "1706.03762"),
    ("02_attention_wechat", "注意力机制中文辅助材料（微信）", "https://mp.weixin.qq.com/s/RLxWevVWHXgX-UcoxDS70w", "web", None),
    ("03_transformer_zhihu", "Transformer 中文辅助材料（知乎）", "https://zhuanlan.zhihu.com/p/53682800", "article", "53682800"),
    ("04_ddpm", "Denoising Diffusion Probabilistic Models", "https://arxiv.org/pdf/2006.11239", "paper", "2006.11239"),
    ("05_diffusion_tutorial_en", "What are Diffusion Models?", "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/", "web", None),
    ("06_diffusion_tutorial_zh", "扩散模型中文教程（知乎）", "https://zhuanlan.zhihu.com/p/525106459", "article", "525106459"),
    ("07_rectified_flow", "Flow Straight and Fast", "https://arxiv.org/pdf/2209.03003", "paper", "2209.03003"),
    ("08_latent_diffusion", "High-Resolution Image Synthesis with Latent Diffusion Models", "https://arxiv.org/pdf/2112.10752", "paper", "2112.10752"),
    ("09_sd_zhihu", "Stable Diffusion 原理与应用（知乎回答）", "https://www.zhihu.com/question/577079491/answer/2954363993", "answer", "2954363993"),
    ("10_dit", "Scalable Diffusion Models with Transformers", "https://arxiv.org/abs/2212.09748", "paper", "2212.09748"),
    ("11_mmdit", "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "https://arxiv.org/abs/2403.03206", "paper", "2403.03206"),
    ("12_sdedit", "SDEdit", "https://arxiv.org/pdf/2108.01073.pdf", "paper", "2108.01073"),
    ("13_blended_latent_diffusion", "Blended Latent Diffusion", "https://omriavrahami.com/blended-latent-diffusion-page/static/paper/Blended_Latent_Diffusion_Paper.pdf", "paper_url", None),
    ("14_prompt_to_prompt", "Prompt-to-Prompt Image Editing with Cross-Attention Control", "https://arxiv.org/pdf/2208.01626.pdf", "paper", "2208.01626"),
    ("15_flowedit", "FlowEdit", "https://arxiv.org/abs/2412.08629", "paper", "2412.08629"),
    ("16_stable_flow", "Stable Flow", "https://arxiv.org/abs/2411.14430", "paper", "2411.14430"),
    ("17_controlnet", "Adding Conditional Control to Text-to-Image Diffusion Models", "https://arxiv.org/pdf/2302.05543", "paper", "2302.05543"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
OPENER = build_opener()


class Response:
    def __init__(self, content: bytes, url: str, headers) -> None:
        self.content = content
        self.url = url
        self.headers = headers

    @property
    def text(self) -> str:
        charset = self.headers.get_content_charset() or "utf-8"
        return self.content.decode(charset, errors="replace")

    def json(self):
        return json.loads(self.text)


def get(url: str, timeout: int = 60) -> Response:
    last = None
    for attempt in range(3):
        try:
            with OPENER.open(Request(url, headers=HEADERS), timeout=timeout) as response:
                return Response(response.read(), response.geturl(), response.headers)
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"下载失败: {url}: {last}")


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks = []
    for number, page in enumerate(reader.pages, 1):
        chunks.append(f"\n\n===== PAGE {number} =====\n\n{page.extract_text() or ''}")
    return "".join(chunks).strip() + "\n"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    text = html_module.unescape("".join(parser.parts))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()) + "\n"


def save_web(prefix: str, title: str, source_url: str, html: str, raw_suffix: str = ".html") -> tuple[Path, Path]:
    raw = WEB_DIR / f"{prefix}{raw_suffix}"
    raw.write_text(html, encoding="utf-8")
    text_path = TEXT_DIR / f"{prefix}.md"
    text_path.write_text(f"# {title}\n\n原始网址：{source_url}\n\n{html_text(html)}", encoding="utf-8")
    return raw, text_path


def fetch_item(prefix: str, title: str, url: str, kind: str, identifier: str | None) -> dict:
    result = {"id": prefix, "title": title, "source_url": url, "kind": kind, "status": "failed"}
    try:
        if kind == "paper":
            pdf_url = f"https://export.arxiv.org/pdf/{identifier}"
            response = get(pdf_url)
            if not response.content.startswith(b"%PDF"):
                raise RuntimeError(f"返回内容不是 PDF ({response.headers.get('content-type')})")
            path = PDF_DIR / f"{prefix}_{identifier.replace('.', '_')}.pdf"
            path.write_bytes(response.content)
            text_path = TEXT_DIR / f"{prefix}.txt"
            text_path.write_text(pdf_text(path), encoding="utf-8")
            result.update(status="saved", saved_url=response.url, raw_file=str(path.relative_to(ROOT)), text_file=str(text_path.relative_to(ROOT)), bytes=path.stat().st_size)
        elif kind == "paper_url":
            response = get(url)
            if not response.content.startswith(b"%PDF"):
                raise RuntimeError(f"返回内容不是 PDF ({response.headers.get('content-type')})")
            path = PDF_DIR / f"{prefix}.pdf"
            path.write_bytes(response.content)
            text_path = TEXT_DIR / f"{prefix}.txt"
            text_path.write_text(pdf_text(path), encoding="utf-8")
            result.update(status="saved", saved_url=response.url, raw_file=str(path.relative_to(ROOT)), text_file=str(text_path.relative_to(ROOT)), bytes=path.stat().st_size)
        elif kind in {"article", "answer"}:
            endpoint = f"https://www.zhihu.com/api/v4/{'articles' if kind == 'article' else 'answers'}/{identifier}"
            response = get(endpoint)
            data = response.json()
            json_path = WEB_DIR / f"{prefix}.json"
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            content = data.get("content") or data.get("excerpt") or ""
            text_path = TEXT_DIR / f"{prefix}.md"
            text_path.write_text(f"# {title}\n\n原始网址：{url}\n\n{html_text(content)}", encoding="utf-8")
            result.update(status="saved", saved_url=endpoint, raw_file=str(json_path.relative_to(ROOT)), text_file=str(text_path.relative_to(ROOT)), bytes=json_path.stat().st_size)
        else:
            response = get(url)
            raw, text_path = save_web(prefix, title, url, response.text)
            result.update(status="saved", saved_url=response.url, raw_file=str(raw.relative_to(ROOT)), text_file=str(text_path.relative_to(ROOT)), bytes=raw.stat().st_size)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    results = []
    for item in ITEMS:
        print(f"[{item[0]}] {item[1]}", flush=True)
        results.append(fetch_item(*item))
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "items": results,
    }
    manifest_path = INDEX_DIR / "抓取清单.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": sum(r["status"] == "saved" for r in results), "failed": sum(r["status"] != "saved" for r in results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
