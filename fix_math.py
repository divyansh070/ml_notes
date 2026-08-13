import re

with open('ml_notes_revision.md', 'r') as f:
    content = f.read()

# Regex to find lines that start with $$ and end with $$ (with optional whitespace)
# We will replace them with:
# ```math
# <equation>
# ```
def replace_math(match):
    eq = match.group(1).strip()
    return f"```math\n{eq}\n```"

# Matches: ^\s*\$\$(.*?)\$\$\s*$ across multiple lines
new_content = re.sub(r'^\s*\$\$(.*?)\$\$\s*$', replace_math, content, flags=re.MULTILINE)

with open('ml_notes_revision.md', 'w') as f:
    f.write(new_content)

print("Fixed math blocks!")
