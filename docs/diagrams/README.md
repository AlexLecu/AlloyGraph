# AlloyGraph Diagrams

Architecture diagrams for the AlloyGraph platform in Mermaid format.

## Available Diagrams

| Diagram | File | Description |
|---------|------|-------------|
| **Evaluation Pipeline** | `evaluation_pipeline.md` | 5-agent workflow for property prediction |
| **Design Pipeline** | `design_pipeline.md` | 7-agent iterative design loop |
| **RAG Chat** | `rag_chat.md` | Research assistant with intent classification |
| **System Overview** | `system_overview.md` | Full platform architecture |

---

## Viewing Diagrams

### GitHub
Mermaid diagrams render automatically in GitHub markdown preview.

### VSCode
Install the **"Markdown Preview Mermaid Support"** extension, then use `Cmd+Shift+V` to preview.

### Online Editor
Copy the Mermaid code to [mermaid.live](https://mermaid.live) for interactive editing and export.

---

## Export to PNG/SVG (for papers)

### Option 1: Mermaid CLI

```bash
# Install
npm install -g @mermaid-js/mermaid-cli

# Export to PNG
mmdc -i evaluation_pipeline.md -o evaluation_pipeline.png -b white

# Export to SVG
mmdc -i evaluation_pipeline.md -o evaluation_pipeline.svg -b white

# Export all
for f in *.md; do
  [[ "$f" != "README.md" ]] && mmdc -i "$f" -o "${f%.md}.png" -b white
done
```

### Option 2: mermaid.live
1. Go to [mermaid.live](https://mermaid.live)
2. Paste the diagram code
3. Click "Export" → PNG or SVG

---

## Diagram Structure

Each file contains:
1. **Full Detail Diagram** - Shows agent roles, tools, and computations
2. **Paper Version** - Simplified for publication
3. **Tables** - Agent interactions, tool mappings
4. **Example** - Concrete walkthrough with values
