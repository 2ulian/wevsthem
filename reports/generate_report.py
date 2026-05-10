"""
Mid-internship progress PDF — detailed but scannable, ~2 pages.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = "reports/midterm_report.pdf"

DARK   = colors.HexColor("#1A237E")
BLUE   = colors.HexColor("#3949AB")
LBLUE  = colors.HexColor("#E8EAF6")
ACCENT = colors.HexColor("#E65100")
GRAY   = colors.HexColor("#616161")
LGRAY  = colors.HexColor("#F5F5F5")
GREEN  = colors.HexColor("#1B5E20")
LGREEN = colors.HexColor("#E8F5E9")
WHITE  = colors.white

doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm)
W = doc.width

base = getSampleStyleSheet()
def st(name, **kw): return ParagraphStyle(name, parent=base["Normal"], **kw)

S_TITLE  = st("T",  fontSize=20, fontName="Helvetica-Bold", textColor=DARK,  alignment=TA_CENTER, spaceAfter=2)
S_SUB    = st("Su", fontSize=10, textColor=GRAY, alignment=TA_CENTER, spaceAfter=10)
S_SEC    = st("Se", fontSize=12, fontName="Helvetica-Bold", textColor=DARK,  spaceBefore=12, spaceAfter=5)
S_BODY   = st("B",  fontSize=9.5, leading=14, textColor=colors.HexColor("#212121"), spaceAfter=4)
S_SMALL  = st("Sm", fontSize=8,  textColor=GRAY, alignment=TA_CENTER)
S_FOOT   = st("Ft", fontSize=7.5, textColor=GRAY, alignment=TA_CENTER)

def sp(n=6):  return Spacer(1, n)
def hr(c=LBLUE): return HRFlowable(width="100%", thickness=0.7, color=c, spaceAfter=6)
def p(txt):   return Paragraph(txt, S_BODY)

def sec(title):
    """Section header with colored left bar via table trick."""
    data = [[
        "",
        Paragraph(title, st("sh", fontSize=11, fontName="Helvetica-Bold",
                             textColor=DARK, spaceBefore=0, spaceAfter=0)),
    ]]
    t = Table(data, colWidths=[0.25*cm, W - 0.25*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), BLUE),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (1,0), (1,0), 8),
    ]))
    return [sp(10), t, sp(6)]

def kpi_row(items):
    """Row of big-number KPI boxes."""
    vals   = [Paragraph(v, st("kv", fontSize=17, fontName="Helvetica-Bold",
                               textColor=DARK, alignment=TA_CENTER)) for v,_ in items]
    labels = [Paragraph(l, st("kl", fontSize=7.5, textColor=GRAY,
                               alignment=TA_CENTER)) for _,l in items]
    t = Table([vals, labels], colWidths=[W/len(items)]*len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LBLUE),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("GRID",          (0,0), (-1,-1), 0.3, WHITE),
    ]))
    return t

def two_col(left_items, right_items, header_l="", header_r=""):
    """Two-column layout using a table."""
    col = W / 2 - 0.3*cm
    rows = []
    if header_l:
        rows.append([
            Paragraph(header_l, st("hl", fontSize=9, fontName="Helvetica-Bold", textColor=BLUE)),
            Paragraph(header_r, st("hr", fontSize=9, fontName="Helvetica-Bold", textColor=BLUE)),
        ])
    max_len = max(len(left_items), len(right_items))
    for i in range(max_len):
        l = left_items[i]  if i < len(left_items)  else ""
        r = right_items[i] if i < len(right_items) else ""
        rows.append([
            Paragraph(l, st(f"lc{i}", fontSize=9, leading=13)) if l else Paragraph("", S_SMALL),
            Paragraph(r, st(f"rc{i}", fontSize=9, leading=13)) if r else Paragraph("", S_SMALL),
        ])
    t = Table(rows, colWidths=[col, col], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("LINEAFTER",     (0,0), (0,-1), 0.5, colors.HexColor("#C5CAE9")),
    ]))
    return t

def plain_table(header, rows, col_ws=None, highlight_last=False):
    col_ws = col_ws or [W/len(header)]*len(header)
    data = [[Paragraph(h, st("ph", fontSize=9, fontName="Helvetica-Bold",
                              textColor=WHITE, alignment=TA_CENTER)) for h in header]]
    for i, row in enumerate(rows):
        is_last = highlight_last and i == len(rows)-1
        cells = []
        for j, cell in enumerate(row):
            align = TA_LEFT if j == 0 else TA_CENTER
            fc    = GREEN if is_last else colors.HexColor("#212121")
            fn    = "Helvetica-Bold" if is_last else "Helvetica"
            cells.append(Paragraph(cell, st(f"c{i}{j}", fontSize=9, leading=13,
                                             textColor=fc, fontName=fn, alignment=align)))
        data.append(cells)

    style = [
        ("BACKGROUND",    (0,0), (-1,0), BLUE),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#C5CAE9")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LGRAY]),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("RIGHTPADDING",  (0,0), (-1,-1), 7),
    ]
    if highlight_last:
        style.append(("BACKGROUND", (0, len(rows)), (-1, len(rows)), LGREEN))
    t = Table(data, colWidths=col_ws)
    t.setStyle(TableStyle(style))
    return t

def bar_table(items, max_val, label_w=3.8*cm, bar_w=None):
    """Horizontal bar chart using colored cells."""
    bar_w = bar_w or (W - label_w - 1.8*cm)
    rows = []
    for label, val, count in items:
        pct = val / max_val
        filled = int(pct * 30)
        empty  = 30 - filled
        bar_str = "█" * filled + "░" * empty
        rows.append([
            Paragraph(label, st("bl", fontSize=8.5, textColor=colors.HexColor("#212121"))),
            Paragraph(f"<font color='#3949AB'>{'█'*filled}</font>"
                      f"<font color='#C5CAE9'>{'░'*empty}</font>",
                      st("bb", fontSize=7, fontName="Courier")),
            Paragraph(f"{count:,}", st("bc", fontSize=8.5, textColor=GRAY, alignment=TA_CENTER)),
        ])
    t = Table(rows, colWidths=[label_w, bar_w, 1.8*cm])
    t.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [WHITE, LGRAY]),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
story = []

# ── HEADER ───────────────────────────────────────────────────────────────────
story.append(Paragraph("We vs Them", S_TITLE))
story.append(Paragraph("Detecting Othering Language in Social Media — Mid-Internship Progress · May 2026", S_SUB))
story.append(hr())
story.append(sp(2))

# ── WHAT IS THIS PROJECT ─────────────────────────────────────────────────────
story.append(p(
    '<b>"Othering"</b> is the discursive construction of an in-group ("we") against a '
    'dehumanised out-group ("them"). The goal of this project is to build an end-to-end NLP '
    'pipeline that detects such language at scale in social media posts, combining '
    'rule-based linguistics and machine learning.'
))
story.append(sp(4))

# ── GLOBAL KPIs ──────────────────────────────────────────────────────────────
story.append(kpi_row([
    ("134 881", "posts collected"),
    ("10 000",  "posts ML-scored"),
    ("7 115",   "othering posts flagged"),
    ("5.3 %",   "othering rate"),
    ("Step 4",  "classifier — in progress"),
]))
story.append(sp(4))

# ════════════════════════════════════════════════════════════════════════════
# STEP 0 — DATA
# ════════════════════════════════════════════════════════════════════════════
story += sec("Step 0 — Data Collection")

story.append(two_col(
    [
        "📦 <b>Source 1</b> — ucberkeley-dlab/measuring-hate-speech (HuggingFace)",
        "→ 133 592 annotated social media comments",
        "→ Multi-label hate-speech ratings by trained annotators",
        "",
        "📦 <b>Source 2</b> — Pushshift Reddit (streaming, 50 k rows)",
        "→ 1 289 posts after subreddit filtering",
        "→ Subreddits: <i>politics, immigration, europe, worldnews, conspiracy</i>",
    ],
    [
        "🧹 <b>Cleaning</b>",
        "→ Title + body merged for Reddit posts",
        "→ Posts &lt; 20 chars removed",
        "→ Unified schema: text · source · subreddit",
        "",
        "📊 <b>Final dataset</b>",
        "→ <b>134 881 posts</b> saved to <i>dataset_clean.csv</i>",
    ],
))

# ════════════════════════════════════════════════════════════════════════════
# STEPS 1–2 — ENRICHMENT
# ════════════════════════════════════════════════════════════════════════════
story += sec("Steps 1 & 2 — Text Enrichment")

story.append(two_col(
    [
        "🏷 <b>Pronoun tagging</b> (all 134 k posts)",
        "→ <i>We-markers</i>: we, us, our, ourselves",
        "→ <i>Them-markers</i>: they, them, their, those people…",
        "",
        "  Pronoun type   │  Posts",
        "  ───────────────┼────────",
        "  none           │  85 292  (63.3 %)",
        "  them only      │  24 916  (18.5 %)",
        "  we only        │  15 261  (11.3 %)",
        "  both           │   8 990  ( 6.7 %)",
    ],
    [
        "☠ <b>Toxicity scoring</b> — unitary/toxic-bert (10 k sample)",
        "→ 5 dimensions: toxicity, severe_toxicity,",
        "   identity_attack, insult, threat",
        "→ Average toxicity score: <b>0.593</b>",
        "",
        "😤 <b>Emotion classification</b> — GoEmotions (10 k sample)",
        "→ top-1 label among 27 emotion classes",
        "→ neutral 35.4 % · anger 18.0 % · annoyance 10.4 %",
        "→ curiosity 5.8 % · admiration 5.0 %",
    ],
))

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — OTHERING DETECTOR
# ════════════════════════════════════════════════════════════════════════════
story += sec("Step 3 — Rule-Based Othering Detector")

story.append(p(
    'Each post is scanned against <b>32 compiled regex patterns</b> grouped in 4 linguistic families. '
    'Each matched family adds 1 point to an <b>othering_score</b> (0–4).'
))
story.append(sp(4))

families_data = [
    ["Family", "Example patterns", "Posts matched"],
    ["Dehumanising metaphors", "invasion · flood · swarm · vermin · parasite · animals", "5 463 (4.1 %)"],
    ["Moral exclusion",        "don't belong · go back · send them back · get out",        "  626 (0.5 %)"],
    ["Generalisation",         "all of them · those people · they never · none of them",   "  976 (0.7 %)"],
    ["Threat framing",         "taking over · replacing us · great replacement",             "   50 (0.04%)"],
]
story.append(plain_table(families_data[0], families_data[1:],
    col_ws=[3.5*cm, 8.5*cm, 2.5*cm]))
story.append(sp(6))

story.append(p("<b>Top matched patterns across the full dataset:</b>"))
story.append(bar_table([
    ("animals",        3326, 3326),
    ("invasion",       1104, 1104),
    ("vermin",         1041, 1041),
    ("none_of_them",    832,  832),
    ("parasite",        831,  831),
    ("go_back",         274,  274),
    ("all_of_them",     150,  150),
], max_val=3326, label_w=3.2*cm))
story.append(sp(6))

story.append(two_col(
    [
        "📊 <b>Othering score distribution</b>",
        "→ score 0 (no othering)  : 127 344 posts  (94.7 %)",
        "→ score 1 (1 family)     :   6 263 posts  ( 4.7 %)",
        "→ score 2 (2 families)   :     849 posts  ( 0.6 %)",
        "→ score 3 (3 families)   :       3 posts  (&lt; 0.1 %)",
    ],
    [
        "☠ <b>Toxicity correlation</b>",
        "→ Avg toxicity — othering posts     : <b>0.760</b>",
        "→ Avg toxicity — non-othering posts : 0.583",
        "→ Othering posts are significantly more toxic",
        "",
        "📍 <b>By source</b>: hate_speech 5.3 % · reddit 0.7 %",
    ],
))

# ════════════════════════════════════════════════════════════════════════════
# NEXT STEPS
# ════════════════════════════════════════════════════════════════════════════
story += sec("Next Steps")

next_data = [
    ["Step 4",  "ML classifier — in progress (TF-IDF & sentence embeddings × LR / LinearSVC)"],
    ["Step 5",  "Topic modelling (BERTopic) on the 7 115 othering posts → identify recurring narratives"],
    ["Step 6",  "Temporal & community analysis — othering rate over time, by subreddit"],
    ["Step 7",  "Fine-tuned BERT classifier with a small set of manual annotations"],
]
t_next = Table(
    [[Paragraph(s, st(f"ns{i}", fontSize=9, fontName="Helvetica-Bold",
                       textColor=WHITE, alignment=TA_CENTER)),
      Paragraph(d, st(f"nd{i}", fontSize=9, leading=13))]
     for i,(s,d) in enumerate(next_data)],
    colWidths=[1.5*cm, W - 1.5*cm]
)
t_next.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (0,-1), BLUE),
    ("ROWBACKGROUNDS",(1,0), (-1,-1), [WHITE, LGRAY, WHITE, LGRAY]),
    ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#C5CAE9")),
    ("ALIGN",         (0,0), (0,-1), "CENTER"),
    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",    (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("LEFTPADDING",   (1,0), (1,-1), 8),
    ("RIGHTPADDING",  (0,0), (-1,-1), 6),
]))
story.append(t_next)

# ── FOOTER ───────────────────────────────────────────────────────────────────
story.append(sp(14))
story.append(HRFlowable(width="100%", thickness=0.4, color=GRAY))
story.append(sp(3))
story.append(Paragraph("We vs Them · Mid-Internship Report · May 2026", S_FOOT))

doc.build(story)
print(f"PDF written to {OUTPUT}")
