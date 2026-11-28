"""Render publication-ready attribution plots from redacted Calibration metrics."""
from html import escape
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE/"failure-attribution.json").read_text(encoding="utf-8"))
CAUSES = {
    "F1":("#65758a","F1  Missing in Top-5"),
    "F2":("#ca8655","F2  No complete unit"),
    "F3":("#8375ad","F3  Relevance"),
    "F4":("#9a6487","F4  Redundancy / MMR"),
    "F5":("#2f688d","F5  Byte limits"),
    "F6":("#aa8645","F6  Missing header"),
    "F7":("#9a7070","F7  Scorer mismatch"),
    "delivered":("#397765","Delivered"),
}


def text(x, y, value, *, size=14, color="#334155", anchor="start", bold=False):
    weight = 650 if bold else 400
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(str(value))}</text>')


def rect(x, y, width, height, color):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}"/>'


def svg(name, title, subtitle, parts, height):
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 {height}" role="img" aria-labelledby="title">',
        f'<title id="title">{escape(title)}</title>',
        rect(0,0,860,height,"#f8fafc"),
        '<g font-family="system-ui,sans-serif">',
        text(25,33,title,size=22,bold=True),
        text(25,57,subtitle,size=13),
        *parts, "</g>", "</svg>",
    ]
    (HERE/name).write_text("\n".join(body)+"\n",encoding="utf-8")


def attribution():
    body = []
    for idx, (stage, (color,label)) in enumerate(CAUSES.items()):
        x = 120 + (idx % 4)*176
        y = 77 + (idx//4)*23
        body.extend([rect(x,y,12,12,color),text(x+17,y+11,label,size=12)])
    for idx,(arm,budget) in enumerate((arm,b) for b in (3200,1600) for arm in ("C2","C3","C4")):
        y = 135+idx*34 + (12 if idx>=3 else 0)
        x = 120
        group = DATA["corrected_first_loss_counts"][f"{arm}-{budget}"]
        body.append(text(25,y+16,f"{arm} · {budget}",size=14,bold=True))
        for stage,(color,_) in CAUSES.items():
            n = group.get(stage,0)
            if not n:
                continue
            width = round(708*n/group["rows"],2)
            body.append(rect(x,y,width,23,color))
            if n>=8:
                body.append(text(round(x+width/2,2),y+16,n,anchor="middle",size=12,color="#ffffff"))
            x += width
    body.append(text(120,366,"First-loss categories are mutually exclusive. Segment labels are row counts.",size=13))
    svg("failure-distribution.svg","Where reference rows first go missing",
        "FinanceBench Calibration · 49 questions · 200 corrected reference rows per arm and byte budget",body,385)


def funnel():
    body = []
    names = ("Expected","In S3 Top-5","After unit stage","After relevance / MMR","Delivered")
    x = (170,305,440,575,710)
    for y,value in [(91,200),(141,150),(191,100),(241,50),(291,0)]:
        body.extend([rect(135,y,640,1,"#dce3ec"),text(120,y+5,value,anchor="end",size=12)])
    for label,at in zip(names,x,strict=True):
        body.append(text(at,312,label,anchor="middle",size=12))
    for arm,color,width in (("C2","#2f688d",4),("C4","#9a6440",2.5)):
        g = DATA["corrected_first_loss_counts"][f"{arm}-3200"]
        total = g["rows"]
        sequence = [total,total-g.get("F1",0),total-g.get("F1",0)-g.get("F2",0),
            total-sum(g.get(f"F{i}",0) for i in range(1,5)),g.get("source_match",0)]
        pts = [(at,291-n/200*200) for at,n in zip(x,sequence,strict=True)]
        coords = " ".join(f"{xx},{yy:.2f}" for xx,yy in pts)
        body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        for xx,yy in pts:
            body.append(f'<circle cx="{xx}" cy="{yy:.2f}" r="4" fill="{color}"/>')
        y = 337 if arm=="C2" else 360
        label = "C2 / C3 (same counts)" if arm=="C2" else arm
        body.extend([rect(137,y-8,18,3,color),text(163,y,label,size=13),
                     text(370,y," → ".join(map(str,sequence)),size=13)])
    svg("evidence-stage-loss.svg","Reference rows remaining by stage",
        "FinanceBench Calibration · 3200-byte budget · cumulative survival by first-loss classification",
        body,380)


if __name__ == "__main__":
    attribution()
    funnel()
