import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, RegularPolygon
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

# --- COLORS ---
C_BG = '#f8f9fa'
C_THOUGHT = '#e74c3c'  # Red for Reasoning
C_ACTION = '#3498db'   # Blue for Tooling
C_OBS = '#2ecc71'      # Green for Observation
C_INPUT = '#95a5a6'    # Grey for Input
C_BAM = '#f1c40f'      # Yellow for BAM!

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# --- Helper to draw boxes ---
def draw_box(ax, x, y, width, height, color, title, subtitle):
    rect = Rectangle((x, y), width, height, facecolor='white', edgecolor=color, lw=4, zorder=2, transform=ax.transData)
    ax.add_patch(rect)
    # Title bar
    title_rect = Rectangle((x, y + height - 0.8), width, 0.8, facecolor=color, edgecolor=color, zorder=2)
    ax.add_patch(title_rect)
    
    ax.text(x + width/2, y + height - 0.4, title, ha='center', va='center', color='white', fontsize=16, fontweight='bold', zorder=3)
    ax.text(x + width/2, y + height/2 - 0.4, subtitle, ha='center', va='center', color='black', fontsize=12, zorder=3, wrap=True)
    return (x + width/2, y + height/2)

# --- Nodes ---
# 1. Input Box (Top Left)
cx_in, cy_in = draw_box(ax, 0.5, 7.5, 3.5, 1.5, C_INPUT, "PROMPT / OBSERVATION", "Initial user request or system trigger.")

# 2. Thought Box (Top Right)
cx_t, cy_t = draw_box(ax, 6.0, 7.0, 3.5, 2.0, C_THOUGHT, "THOUGHT (Reasoning)", "Agent model analyzes context and decides what tool to use next.")

# 3. Action Box (Bottom Right)
cx_a, cy_a = draw_box(ax, 6.0, 3.0, 3.5, 2.0, C_ACTION, "ACTION (Tool Execution)", "Agent executes the selected tool (Search, Math, Code, RAG, etc.)")

# 4. Observation Box (Bottom Left)
cx_o, cy_o = draw_box(ax, 1.0, 3.0, 3.5, 2.0, C_OBS, "OBSERVATION (Output)", "Raw result/data returned from the environment or tool.")

# --- Arrows (The Loop) ---
style = "Simple, tail_width=2, head_width=10, head_length=12"
kw = dict(arrowstyle=style, color='black', alpha=0.7, zorder=1)

# Input -> Thought
a1 = FancyArrowPatch((cx_in+1.75, cy_in), (cx_t-1.75, cy_t), connectionstyle="arc3,rad=-0.2", **kw)
ax.add_patch(a1)

# Thought -> Action
a2 = FancyArrowPatch((cx_t, cy_t-1.0), (cx_a, cy_a+1.0), connectionstyle="arc3,rad=-0.3", **kw)
ax.add_patch(a2)
ax.text(7.8, 6.0, "Decides Tool", rotation=-90, fontsize=11, fontweight='bold', color=C_THOUGHT)

# Action -> Observation
a3 = FancyArrowPatch((cx_a-1.75, cy_a), (cx_o+1.75, cy_o), connectionstyle="arc3,rad=-0.2", **kw)
ax.add_patch(a3)
ax.text(5.0, 3.5, "Executes Tool", fontsize=11, fontweight='bold', color=C_ACTION)

# Observation -> Thought (The Iterative Loop)
a4 = FancyArrowPatch((cx_o, cy_o+1.0), (cx_t-1.75, cy_t-0.5), connectionstyle="arc3,rad=0.3", **kw)
ax.add_patch(a4)
ax.text(2.5, 6.0, "Feeds output context\nback into Reasoning Loop", fontsize=11, fontweight='bold', color=C_OBS, ha='center')

# --- Final Answer (BAM!) ---
# Bam Starburst
star = RegularPolygon((9.0, 9.0), numVertices=12, radius=0.8, orientation=0.2, facecolor=C_BAM, edgecolor='black', lw=2, zorder=4)
ax.add_patch(star)
ax.text(9.0, 9.0, "BAM!\nFinal\nAnswer", ha='center', va='center', fontsize=12, fontweight='bold', zorder=5)

# Thought -> BAM
a5 = FancyArrowPatch((cx_t+1.0, cy_t+1.0), (8.5, 8.5), connectionstyle="arc3,rad=-0.2", arrowstyle="Simple, tail_width=2, head_width=10, head_length=12", color=C_THOUGHT, zorder=1)
ax.add_patch(a5)
ax.text(8.3, 8.0, "Done", fontsize=12, fontweight='bold', color=C_THOUGHT, rotation=45)

# --- Title ---
plt.suptitle("The Agentic AI Loop: Iterative Reasoning and Acting", fontsize=22, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('assets/react_agent_loop.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/react_agent_loop.png successfully!")
