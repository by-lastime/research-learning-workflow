#!/usr/bin/env python3
"""Clean the user-supplied Zhihu capture for source URL 09.

The source document contains Zhihu navigation before the selected answer and
other answers/promotions after it.  Paragraph numbers were established by
manual inspection with officecli.  This script keeps the question title and
the selected answer body (including its figures), then emits a searchable
Markdown companion.
"""

from pathlib import Path
from shutil import copy2

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "02_网页原始内容/09_sd_用户抓取/09_sd_用户抓取原稿.docx"
CLEAN = ROOT / "02_网页原始内容/09_sd_用户抓取/09_sd_知乎回答_清洗版.docx"
MARKDOWN = ROOT / "03_可检索正文/09_sd_知乎回答_用户抓取清洗正文.md"
SOURCE_URL = "https://www.zhihu.com/question/577079491/answer/2954363993"

# One-based paragraph numbers in the archived source document.
TITLE_PARAGRAPH = 43
BODY_START = 96
BODY_END = 688  # End of the answer's reference list; site action bar begins at 689.


def remove_paragraph(paragraph: Paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)


def has_numbering(paragraph: Paragraph) -> bool:
    ppr = paragraph._p.pPr
    return ppr is not None and ppr.numPr is not None


def image_count(paragraph: Paragraph) -> int:
    return len(paragraph._p.xpath(".//w:drawing")) + len(
        paragraph._p.xpath(".//w:pict")
    )


def insert_source_note_after(title: Paragraph) -> None:
    note = OxmlElement("w:p")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = f"来源：{SOURCE_URL}（用户抓取稿清洗；访问页面元素已移除）"
    run.append(text)
    note.append(run)
    title._p.addnext(note)


def remove_broken_images(document: Document) -> int:
    """Remove image drawings whose relationship is already absent in source."""
    removed = 0
    valid_relationships = set(document.part.rels)
    for blip in list(document.element.body.xpath(".//a:blip")):
        relationship_id = blip.get(qn("r:embed"))
        if not relationship_id or relationship_id in valid_relationships:
            continue
        drawing = blip
        while drawing is not None and drawing.tag not in {
            qn("w:drawing"),
            qn("w:pict"),
        }:
            drawing = drawing.getparent()
        if drawing is not None and drawing.getparent() is not None:
            drawing.getparent().remove(drawing)
            removed += 1
    return removed


def remove_inherited_style_schema_noise(document: Document) -> int:
    """Drop invalidly ordered web-export uiPriority nodes from inherited styles."""
    removed = 0
    for node in list(document.styles.element.xpath(".//w:uiPriority")):
        node.getparent().remove(node)
        removed += 1
    return removed


def make_clean_docx() -> None:
    copy2(SOURCE, CLEAN)
    document = Document(CLEAN)
    original = list(document.paragraphs)
    keep = {TITLE_PARAGRAPH, *range(BODY_START, BODY_END + 1)}
    for index, paragraph in enumerate(original, 1):
        if index not in keep:
            remove_paragraph(paragraph)
    insert_source_note_after(document.paragraphs[0])
    removed = remove_broken_images(document)
    style_nodes = remove_inherited_style_schema_noise(document)
    document.save(CLEAN)
    print(f"removed_broken_images={removed}")
    print(f"removed_invalid_style_nodes={style_nodes}")


def make_markdown() -> tuple[int, int]:
    document = Document(CLEAN)
    lines = [
        "<!-- 由用户提供的网页抓取稿清洗生成；正文文字未作知识性改写。 -->",
        "",
    ]
    figure_number = 0
    in_code = False

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        style = paragraph.style.name if paragraph.style else ""
        is_code = style == "HTML Preformatted"

        if is_code and not in_code:
            lines.extend(["```python"])
            in_code = True
        elif not is_code and in_code:
            lines.extend(["```", ""])
            in_code = False

        if is_code:
            lines.append(paragraph.text.rstrip())
        elif text:
            if style.lower().startswith("heading"):
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 2
                lines.extend([f"{'#' * level} {text}", ""])
            elif has_numbering(paragraph):
                lines.extend([f"- {text}", ""])
            else:
                lines.extend([text, ""])

        for _ in range(image_count(paragraph)):
            figure_number += 1
            lines.extend([f"[插图 {figure_number}：保存在清洗版 Word 文档中]", ""])

    if in_code:
        lines.extend(["```", ""])

    MARKDOWN.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    chars = sum(len(p.text) for p in document.paragraphs)
    return figure_number, chars


if __name__ == "__main__":
    make_clean_docx()
    figures, characters = make_markdown()
    print(f"clean_docx={CLEAN}")
    print(f"markdown={MARKDOWN}")
    print(f"figures={figures} characters={characters}")
