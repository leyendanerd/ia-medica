"""
odontogram_renderer.py

Renderiza el odontograma automático como un gráfico SVG embebible en
Streamlit, agrupado por cuadrantes FDI (11-18, 21-28, 31-38, 41-48).

Objetivo del proyecto: "automatizar la generación de odontogramas"
"""

STATUS_COLORS = {
    "sano":         {"fill": "#0f6e56", "stroke": "#2ed573", "text": "#eafff5"},
    "caries":       {"fill": "#712b13", "stroke": "#ff4757", "text": "#ffece9"},
    "perdida_osea": {"fill": "#854f0b", "stroke": "#ffa502", "text": "#fff3e0"},
    "quiste":       {"fill": "#3c3489", "stroke": "#a259ff", "text": "#f2ecff"},
    "ausente":      {"fill": "#1a2235", "stroke": "#3a4560", "text": "#8899bb"},
}

STATUS_LABELS = {
    "sano": "Sano", "caries": "Caries",
    "perdida_osea": "Pérdida ósea", "quiste": "Quiste", "ausente": "Ausente",
}


def render_odontogram_svg(odontogram: list, width: int = 640) -> str:
    """
    Genera un SVG del odontograma completo (32 piezas, notación FDI),
    organizado en 4 cuadrantes tal como se presenta clínicamente.
    """
    by_id = {t["tooth_id"]: t for t in odontogram}

    upper_right = ["18", "17", "16", "15", "14", "13", "12", "11"]
    upper_left  = ["21", "22", "23", "24", "25", "26", "27", "28"]
    lower_left  = ["31", "32", "33", "34", "35", "36", "37", "38"]
    lower_right = ["48", "47", "46", "45", "44", "43", "42", "41"]

    tooth_w, tooth_h, gap = 34, 44, 4
    row_gap = 18
    quad_gap = 10

    total_w = (tooth_w + gap) * 8 * 2 + quad_gap
    svg_w = max(width, total_w + 40)
    svg_h = tooth_h * 2 + row_gap + 60

    def tooth_box(tooth_id, x, y, flip_label_pos="top"):
        t = by_id.get(tooth_id, {"status": "ausente", "confidence": None})
        colors = STATUS_COLORS[t["status"]]
        conf_txt = f'{t["confidence"]:.0%}' if t.get("confidence") else ""

        label_y = y - 6 if flip_label_pos == "top" else y + tooth_h + 14

        return f"""
        <g>
            <rect x="{x}" y="{y}" width="{tooth_w}" height="{tooth_h}" rx="6"
                  fill="{colors['fill']}" stroke="{colors['stroke']}" stroke-width="1.5"/>
            <text x="{x + tooth_w/2}" y="{y + tooth_h/2 - 2}" text-anchor="middle"
                  font-size="13" font-weight="600" fill="{colors['text']}"
                  font-family="Space Grotesk, sans-serif">{tooth_id}</text>
            <text x="{x + tooth_w/2}" y="{y + tooth_h/2 + 13}" text-anchor="middle"
                  font-size="9" fill="{colors['text']}" opacity="0.85"
                  font-family="Space Grotesk, sans-serif">{conf_txt}</text>
        </g>"""

    elements = []
    start_x = 20
    upper_y = 30
    lower_y = upper_y + tooth_h + row_gap

    x = start_x
    for tid in upper_right:
        elements.append(tooth_box(tid, x, upper_y))
        x += tooth_w + gap
    x += quad_gap
    for tid in upper_left:
        elements.append(tooth_box(tid, x, upper_y))
        x += tooth_w + gap

    x = start_x
    for tid in lower_left:
        elements.append(tooth_box(tid, x, lower_y))
        x += tooth_w + gap
    x += quad_gap
    for tid in lower_right:
        elements.append(tooth_box(tid, x, lower_y))
        x += tooth_w + gap

    midline_x = start_x + 8 * (tooth_w + gap) + quad_gap / 2 - gap / 2
    elements.append(
        f'<line x1="{midline_x}" y1="{upper_y - 10}" x2="{midline_x}" y2="{lower_y + tooth_h + 10}" '
        f'stroke="#2a3654" stroke-width="1" stroke-dasharray="3,3"/>'
    )
    elements.append(
        f'<line x1="{start_x}" y1="{upper_y + tooth_h + row_gap/2}" x2="{x - gap}" y2="{upper_y + tooth_h + row_gap/2}" '
        f'stroke="#2a3654" stroke-width="1" stroke-dasharray="3,3"/>'
    )

    legend_y = lower_y + tooth_h + 30
    legend_items = ["sano", "caries", "perdida_osea", "quiste", "ausente"]
    legend_x = start_x
    for key in legend_items:
        colors = STATUS_COLORS[key]
        elements.append(f"""
        <rect x="{legend_x}" y="{legend_y}" width="12" height="12" rx="3" fill="{colors['fill']}" stroke="{colors['stroke']}" stroke-width="1"/>
        <text x="{legend_x + 18}" y="{legend_y + 10.5}" font-size="11" fill="#8899bb" font-family="Space Grotesk, sans-serif">{STATUS_LABELS[key]}</text>
        """)
        legend_x += len(STATUS_LABELS[key]) * 6.5 + 34

    svg = f"""
    <svg viewBox="0 0 {svg_w} {svg_h + 30}" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif;">
        <rect width="{svg_w}" height="{svg_h + 30}" fill="transparent"/>
        {''.join(elements)}
    </svg>
    """
    return svg
