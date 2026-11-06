import io
import re
from xml.sax.saxutils import escape
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def _clean(line):
    return re.sub(r"\*\*(.*?)\*\*", r"\1", line).replace("*", "").strip()


def make_docx(text, title="Data Analysis Report"):
    doc = Document()
    doc.add_heading(title, 0)
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            doc.add_heading(_clean(s.lstrip("# ")), level=1)
        elif s.startswith(("- ", "* ")):
            doc.add_paragraph(_clean(s[2:]), style="List Bullet")
        else:
            doc.add_paragraph(_clean(s))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_pdf(text, title="Data Analysis Report"):
    buf = io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            story.append(Paragraph(escape(_clean(s.lstrip("# "))), styles["Heading2"]))
        elif s.startswith(("- ", "* ")):
            story.append(Paragraph("• " + escape(_clean(s[2:])), styles["BodyText"]))
        else:
            story.append(Paragraph(escape(_clean(s)), styles["BodyText"]))
        story.append(Spacer(1, 6))
    pdf.build(story)
    return buf.getvalue()