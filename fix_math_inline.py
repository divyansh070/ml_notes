import re

with open('ml_notes_revision.md', 'r') as f:
    content = f.read()

def replace_inline(match):
    eq = match.group(1)
    return f"$`{eq}`$"

# Matches inline math $...$ that doesn't span lines and doesn't contain backticks
# and avoids double $$
new_content = re.sub(r'(?<!\$)\$([^\$\`\n]+?)\$(?!\$)', replace_inline, content)

with open('ml_notes_revision.md', 'w') as f:
    f.write(new_content)

print("Fixed inline math formatting for GitHub!")
