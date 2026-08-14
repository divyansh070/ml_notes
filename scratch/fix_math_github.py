import re

filepath = '/Users/divyanshverma/Desktop/ml_interview_questions/transformer_and_beyond.md'

with open(filepath, 'r') as f:
    content = f.read()

def repl(match):
    indent = match.group(1) or ''
    math_content = match.group(2).strip()
    return f"{indent}$$\n{indent}{math_content}\n{indent}$$"

# Revert ```math back to $$ but keep them on their own lines
new_content = re.sub(r'(?m)^([ \t]*)```math\s*(.*?)\s*```', repl, content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(new_content)

print("Reverted to $$ format.")
