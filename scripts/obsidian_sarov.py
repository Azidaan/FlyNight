"""Create the editable Sarov conference board through build_obsidian_vault.py."""
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "experiments/sarov_conference.md"
NOTES = "Саров"


def create_board(vault: Path) -> Path:
    sections = SOURCE.read_text(encoding="utf-8").strip().split("\n## ")
    board_name, introduction = sections[0].split("\n\n", 1)
    board_name = board_name.removeprefix("# ")
    topics = [(board_name.removeprefix("Граф "), introduction)]
    for section in sections[1:]:
        title, body = section.split("\n\n", 1)
        topics.append((title, body))

    nodes, edges, files = [], [], {}
    for index, (title, body) in enumerate(topics):
        question = body.strip().split("\n\n", 1)[0]
        note_link = f"[[{NOTES}/{title}|{title}]]"
        nodes.append({
            "id": f"sarov-{index}", "type": "text",
            "text": f"### {note_link}\n\n{question}",
            "x": 380 * index, "y": 0, "width": 316, "height": 154,
        })
        note = f"# {title}\n\n{body.strip()}\n"
        if index + 1 < len(topics):
            following = topics[index + 1][0]
            note += f"\nДальше по линии — [[{NOTES}/{following}|{following}]].\n"
            edges.append({
                "id": f"link-{index}-{index + 1}",
                "fromNode": f"sarov-{index}", "fromSide": "right", "fromEnd": "none",
                "toNode": f"sarov-{index + 1}", "toSide": "left", "toEnd": "none",
            })
        files[vault / NOTES / f"{title}.md"] = note

    board = vault / f"{board_name}.canvas"
    files[board] = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2) + "\n"
    # A later rebuild can restore missing files without erasing user-added branches.
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        except FileExistsError:
            continue
    return board
