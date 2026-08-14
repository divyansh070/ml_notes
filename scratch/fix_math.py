import re
import os

filepath = '/Users/divyanshverma/Desktop/ml_interview_questions/transformer_and_beyond.md'

with open(filepath, 'r') as f:
    content = f.read()

def repl(match):
    indent = match.group(1) or ''
    math_content = match.group(2).strip()
    return f"{indent}```math\n{indent}{math_content}\n{indent}```"

# Match optional leading whitespace, $$, any content, $$
new_content = re.sub(r'(?m)^([ \t]*)\$\$\s*(.*?)\s*\$\$', repl, content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(new_content)

print("Math blocks fixed.")
