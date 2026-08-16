import re

def create_slug(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = text.replace(' ', '-')
    return text

def update_file_toc(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find where TOC should start and end
    toc_start_idx = -1
    first_header_idx = -1

    for i, line in enumerate(lines):
        if "**Table of Contents:**" in line or "## Table of Contents" in line:
            toc_start_idx = i
        elif line.startswith("## ") and first_header_idx == -1 and i > toc_start_idx:
            # But wait, there might be a separator `---` before the first header
            first_header_idx = i
            break
            
    # if no explicit TOC start found, just put it after the main title
    if toc_start_idx == -1:
        for i, line in enumerate(lines):
            if line.startswith("# "):
                toc_start_idx = i + 1
                break

    if first_header_idx == -1:
        for i in range(toc_start_idx + 1, len(lines)):
            if lines[i].startswith("---") or lines[i].startswith("## "):
                first_header_idx = i
                break

    if toc_start_idx == -1 or first_header_idx == -1:
        print(f"Could not find TOC injection points in {filepath}")
        return

    # Extract headers from the rest of the file
    headers = []
    for line in lines[first_header_idx:]:
        if line.startswith("## ") or line.startswith("### "):
            headers.append(line.strip())

    # Build TOC string
    toc_lines = ["\n**Table of Contents:**\n\n"]
    for h in headers:
        if h.startswith("## "):
            title = h[3:].strip()
            slug = create_slug(title)
            toc_lines.append(f"- [{title}](#{slug})\n")
        elif h.startswith("### "):
            title = h[4:].strip()
            slug = create_slug(title)
            toc_lines.append(f"  - [{title}](#{slug})\n")

    toc_lines.append("\n---\n\n")

    # Replace the old TOC with the new TOC
    # The old TOC is between toc_start_idx and first_header_idx (not including first_header_idx)
    # Wait, in ml_notes_revision, there are `---` before the first `## Part 1:`
    # Let's just find the exact `---` or first `##`
    
    # We need to find the `---` that precedes the first `## `
    # or just replace everything between toc_start_idx and first_header_idx.
    # Let's adjust first_header_idx to be the `---` if it exists.
    actual_end_of_toc = first_header_idx
    for i in range(first_header_idx - 1, toc_start_idx, -1):
        if lines[i].strip() == "---":
            actual_end_of_toc = i
            break
            
    new_content = lines[:toc_start_idx] + toc_lines + lines[actual_end_of_toc + 1:]
    
    with open(filepath, 'w') as f:
        f.writelines(new_content)
        
    print(f"Successfully updated TOC for {filepath}")

update_file_toc('/Users/divyanshverma/Desktop/ml_interview_questions/ml_notes_revision.md')
update_file_toc('/Users/divyanshverma/Desktop/ml_interview_questions/transformer_and_beyond.md')
