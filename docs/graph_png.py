#!/usr/bin/env python3
"""
Generate PNG visualization of LangGraph using PIL/Pillow
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_graph_png():
    # Image dimensions
    width, height = 1200, 900
    
    # Create image with dark background
    img = Image.new('RGB', (width, height), '#0f172a')
    draw = ImageDraw.Draw(img)
    
    # Try to get a font, fall back to default if not available
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_header = ImageFont.truetype("arial.ttf", 20)
        font_body = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Colors
    colors = {
        'start_end': '#64748b',
        'router': '#fbbf24',
        'hr': '#3b82f6',
        'it': '#22c55e',
        'general': '#9ca3af',
        'text': '#f1f5f9',
        'text_dark': '#1e293b',
        'arrow': '#94a3b8',
        'line': '#475569'
    }
    
    # Title
    draw.text((width//2, 30), "HR-IT Copilot LangGraph Architecture", 
              fill=colors['text'], font=font_title, anchor='mt')
    
    # Node positions
    nodes = {
        'START': (600, 100),
        'router': (600, 180),
        'hr_rag': (200, 320),
        'hr_leave': (500, 320),
        'it_support': (800, 320),
        'general': (1050, 320),
        'END': (600, 500)
    }
    
    # Node sizes
    node_sizes = {
        'START': (80, 40),
        'router': (200, 70),
        'hr_rag': (180, 100),
        'hr_leave': (200, 140),
        'it_support': (180, 100),
        'general': (160, 70),
        'END': (80, 40)
    }
    
    # Draw edges first (behind nodes)
    edges = [
        ('START', 'router', ''),
        ('router', 'hr_rag', 'hr_policy'),
        ('router', 'hr_leave', 'leave_*'),
        ('router', 'it_support', 'it_*'),
        ('router', 'general', 'default'),
        ('hr_rag', 'END', ''),
        ('hr_leave', 'END', ''),
        ('it_support', 'END', ''),
        ('general', 'END', '')
    ]
    
    for start, end, label in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Adjust start/end points to edge of nodes
        w1, h1 = node_sizes[start]
        w2, h2 = node_sizes[end]
        
        # Simple line drawing
        draw.line([(x1, y1 + h1//2), (x2, y2 - h2//2)], 
                  fill=colors['arrow'], width=2)
        
        # Arrow head
        if y2 > y1:
            draw.polygon([(x2-5, y2-h2//2-10), (x2+5, y2-h2//2-10), (x2, y2-h2//2)], 
                        fill=colors['arrow'])
        
        # Label
        if label:
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            draw.text((mid_x, mid_y), label, fill=colors['text'], font=font_small, anchor='mm')
    
    # Draw nodes
    node_configs = [
        ('START', 'START', colors['start_end'], colors['text']),
        ('router', '🔀 router_node\nIntent Router', colors['router'], colors['text_dark']),
        ('hr_rag', '📚 hr_rag\nHR Policy Q&A\n(RAG)', colors['hr'], colors['text']),
        ('hr_leave', '🏖️ hr_leave\nLeave Management\n• Balance\n• Apply\n• Approve\n• Status', colors['hr'], colors['text']),
        ('it_support', '🎧 it_support\nIT Support\n• Tickets\n• Assets', colors['it'], colors['text']),
        ('general', '💬 general\nFallback', colors['general'], colors['text']),
        ('END', 'END', colors['start_end'], colors['text'])
    ]
    
    for node_id, text, bg_color, text_color in node_configs:
        x, y = nodes[node_id]
        w, h = node_sizes[node_id]
        
        # Draw rounded rectangle
        draw.rounded_rectangle([x-w//2, y-h//2, x+w//2, y+h//2], 
                               radius=10, fill=bg_color, outline=colors['line'], width=2)
        
        # Draw text
        lines = text.split('\n')
        line_height = 20
        start_y = y - (len(lines) * line_height) // 2 + 10
        
        for i, line in enumerate(lines):
            fy = start_y + i * line_height
            f = font_header if i == 0 and node_id not in ['START', 'END'] else font_body
            draw.text((x, fy), line, fill=text_color, font=f, anchor='mm')
    
    # Legend
    legend_y = 600
    draw.text((50, legend_y), "Domain Coloring:", fill=colors['text'], font=font_header)
    
    legend_items = [
        (100, legend_y + 35, colors['router'], "Router (Intent Classification)"),
        (100, legend_y + 65, colors['hr'], "HR Domain (Leave & Policy)"),
        (100, legend_y + 95, colors['it'], "IT Domain (Support & Assets)"),
        (100, legend_y + 125, colors['general'], "General (Fallback)"),
    ]
    
    for lx, ly, color, label in legend_items:
        draw.rounded_rectangle([lx, ly-10, lx+30, ly+10], radius=5, fill=color)
        draw.text((lx + 40, ly), label, fill=colors['text'], font=font_body, anchor='lm')
    
    # Intent routing table
    table_x = 650
    table_y = 600
    
    draw.text((table_x, table_y), "Intent Routing:", fill=colors['text'], font=font_header)
    
    routing = [
        ("hr_policy", "→ hr_rag"),
        ("leave_balance, leave_apply", "→ hr_leave"),
        ("leave_approve, leave_status", "→ hr_leave"),
        ("it_ticket, it_asset", "→ it_support"),
        ("general (default)", "→ general"),
    ]
    
    for i, (intent, target) in enumerate(routing):
        y = table_y + 35 + i * 25
        draw.text((table_x, y), intent, fill=colors['text'], font=font_small, anchor='lt')
        draw.text((table_x + 250, y), target, fill='#a5b4fc', font=font_small, anchor='lt')
    
    # Save
    output_path = 'docs/langgraph_structure.png'
    img.save(output_path, 'PNG')
    print(f"✅ PNG saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_graph_png()
