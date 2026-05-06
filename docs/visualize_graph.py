#!/usr/bin/env python3
"""
LangGraph Visualizer for HR-IT Copilot
Generates a visual representation of the agent graph structure
"""

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("graphviz not installed. Install with: pip install graphviz")


def create_graphviz_diagram():
    """Create a visual diagram using graphviz"""
    if not HAS_GRAPHVIZ:
        print("Please install graphviz: pip install graphviz")
        return
    
    dot = graphviz.Digraph(
        name='HR_IT_Copilot_Graph',
        format='png',
        graph_attr={'rankdir': 'TB', 'bgcolor': 'white', 'fontname': 'Arial'},
        node_attr={'fontname': 'Arial', 'shape': 'box', 'style': 'rounded,filled'},
        edge_attr={'fontname': 'Arial'}
    )
    
    # Entry point
    dot.node('START', 'START', shape='circle', fillcolor='#e5e7eb', style='filled')
    
    # Router
    dot.node('router', '🔀 router_node\n(Intent Classification)', 
             fillcolor='#fef3c7', shape='box')
    
    # HR Domain
    dot.node('hr_rag', '📚 hr_rag\nHR Policy Q&A (RAG)', 
             fillcolor='#dbeafe')
    dot.node('hr_leave', '🏖️ hr_leave\nLeave Management\n• Balance queries\n• Apply for leave\n• Approve/Reject\n• Status checks', 
             fillcolor='#dbeafe', height='1.5')
    
    # IT Domain
    dot.node('it_support', '🎧 it_support\nIT Support\n• Raise tickets\n• Request assets', 
             fillcolor='#dcfce7', height='1.2')
    
    # General
    dot.node('general', '💬 general\nFallback Responses', 
             fillcolor='#f3f4f6')
    
    # Exit
    dot.node('END', 'END', shape='circle', fillcolor='#e5e7eb', style='filled')
    
    # Edges
    dot.edge('START', 'router', label='')
    
    # Router conditional edges
    dot.edge('router', 'hr_rag', label='intent: hr_policy', color='#2563eb')
    dot.edge('router', 'hr_leave', label='intent: leave_*', color='#2563eb')
    dot.edge('router', 'it_support', label='intent: it_*', color='#16a34a')
    dot.edge('router', 'general', label='default', color='#6b7280', style='dashed')
    
    # Exit edges
    dot.edge('hr_rag', 'END', color='#6b7280')
    dot.edge('hr_leave', 'END', color='#6b7280')
    dot.edge('it_support', 'END', color='#6b7280')
    dot.edge('general', 'END', color='#6b7280')
    
    # Save
    dot.render('langgraph_structure', directory='.', cleanup=True)
    print("✅ Diagram saved as 'langgraph_structure.png'")


def print_text_diagram():
    """Print ASCII art representation"""
    diagram = """
╔═══════════════════════════════════════════════════════════════╗
║           HR-IT Copilot LangGraph Structure                   ║
╚═══════════════════════════════════════════════════════════════╝

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌───────────────────────────────────────┐
    │     🔀 router_node                    │
    │     (Intent Classification)             │
    └────┬──────────────────────────────────┘
         │
    ┌────┴────┬────────────┬────────────┬────────────┐
    │         │            │            │            │
    ▼         ▼            ▼            ▼            ▼
┌────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
│📚      │ │🏖️         │ │🎧        │ │💬        │
│hr_rag  │ │hr_leave   │ │it_support│ │general   │
│        │ │           │ │          │ │          │
│Policy  │ │• Balance  │ │• Tickets │ │Fallback │
│Q&A     │ │• Apply    │ │• Assets  │ │         │
│(RAG)   │ │• Approve  │ │          │ │          │
│        │ │• Status   │ │          │ │          │
└────┬───┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘
     │           │            │            │            │
     └───────────┴────────────┴────────────┴────────────┘
                          │
                          ▼
                    ┌─────────┐
                    │   END   │
                    └─────────┘

═══════════════════════════════════════════════════════════════

Intent Routing Table:
┌────────────────────────┬──────────────────────────────────────┐
│ Intent                 │ Target Node                          │
├────────────────────────┼──────────────────────────────────────┤
│ hr_policy              │ hr_rag (HR Policy Q&A)              │
├────────────────────────┼──────────────────────────────────────┤
│ leave_balance          │                                      │
│ leave_apply            │ hr_leave (Leave Management)         │
│ leave_status           │                                      │
│ leave_approve          │                                      │
│ leave_cancel           │                                      │
│ pending_approvals      │                                      │
├────────────────────────┼──────────────────────────────────────┤
│ it_ticket              │                                      │
│ it_asset               │ it_support (IT Support)             │
│ it_ticket_update       │                                      │
├────────────────────────┼──────────────────────────────────────┤
│ general (default)      │ general (Fallback)                  │
└────────────────────────┴──────────────────────────────────────┘
"""
    print(diagram)


if __name__ == "__main__":
    print("=" * 60)
    print("HR-IT Copilot LangGraph Visualizer")
    print("=" * 60)
    
    print("\n📊 Generating GraphViz diagram...")
    create_graphviz_diagram()
    
    print("\n📝 Text Diagram:")
    print_text_diagram()
    
    print("\n💡 To view the Mermaid diagram, open 'langgraph_visual.md' in a Markdown viewer")
    print("   that supports Mermaid (GitHub, VS Code with Mermaid extension, etc.)")
