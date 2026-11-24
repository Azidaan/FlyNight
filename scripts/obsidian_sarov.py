"""Build the Sarov conference chain for Obsidian's global graph."""
import json
from pathlib import Path
from scripts.obsidian_sarov_branch import create_sam2_branch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/sarov_conference.md"
NOTES = "Саров"


def read_topics():
    sections = SOURCE.read_text(encoding="utf-8").strip().split("\n## ")
    board_name, introduction = sections[0].split("\n\n", 1)
    board_name = board_name.removeprefix("# ")
    topics = [(board_name, introduction)]
    for section in sections[1:]:
        title, body = section.split("\n\n", 1)
        topics.append((title, body))
    return topics


def include_in_graph(vault: Path, manifest: dict) -> None:
    """Include this independent component in the global graph's existing search."""
    if not (vault / NOTES).is_dir():
        return
    spine = [f"{NOTES}/{title}.md" for title, _ in read_topics()]
    files = {p.relative_to(vault).as_posix() for p in (vault / NOTES).rglob("*.md")}
    if not set(spine) <= files:
        raise ValueError("В цепочке «Саров» отсутствуют заметки: " + ", ".join(sorted(set(spine) - files)))
    manifest["sequences"] = {**manifest.get("sequences", {}), NOTES: spine}
    manifest["files"] = sorted(set(manifest["files"]) | files)
    selector = rf"path:/^{NOTES}\/.+\.md$/"
    if selector not in manifest["search"]:
        manifest["search"] = f'({manifest["search"]}) OR {selector}'


def create_notes(vault: Path) -> Path:
    topics = read_topics()
    board_name = topics[0][0]
    board = vault / f"{board_name}.canvas"
    old_title = board_name.removeprefix("Граф ")
    old_root = vault / NOTES / f"{old_title}.md"
    new_root = vault / NOTES / f"{board_name}.md"
    if old_root.exists():
        old_text = old_root.read_text(encoding="utf-8")
        if new_root.exists() or not old_text.startswith(f"# {old_title}\n"):
            raise ValueError(f"Проверьте две версии корневой заметки в {old_root.parent}")
        old_root.rename(new_root)
        new_text = old_text.replace(f"# {old_title}\n", f"# {board_name}\n", 1)
        new_text = new_text.replace(
            "Открой нужную тему на холсте. Линия слева направо задаёт порядок обсуждения. "
            "Под каждой темой можно разместить её будущий подграф.",
            topics[0][1].strip().split("\n\n")[-1],
        )
        new_root.write_bytes(new_text.encode("utf-8"))
    if board.exists():
        old_link = f"[[{NOTES}/{old_title}|{old_title}]]"
        original = board.read_text(encoding="utf-8")
        if old_link in original:
            board.write_bytes(original.replace(old_link, f"[[{NOTES}/{board_name}|{board_name}]]").encode("utf-8"))

    files = {}
    for index, (title, body) in enumerate(topics):
        note = f"# {title}\n\n{body.strip()}\n"
        if index + 1 < len(topics):
            following = topics[index + 1][0]
            note += f"\nДальше по линии — [[{NOTES}/{following}|{following}]].\n"
        files[vault / NOTES / f"{title}.md"] = note

    # A later rebuild can restore missing files without erasing user-added branches.
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        except FileExistsError:
            continue
    create_sam2_branch(vault / NOTES)
    return new_root


def update_graph_filter(vault: Path) -> None:
    """Update just membership and search on an already built project graph."""
    manifest_path = vault / ".graph-visible.json"
    if not manifest_path.is_file():
        return
    config_path = vault / ".obsidian/graph.json"
    manifest = json.loads(manifest_path.read_bytes())
    config = json.loads(config_path.read_bytes())
    include_in_graph(vault, manifest)
    config["search"] = manifest["search"]
    for path, obj in [(manifest_path, manifest), (config_path, config)]:
        if json.loads(path.read_bytes()) != obj:
            path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
