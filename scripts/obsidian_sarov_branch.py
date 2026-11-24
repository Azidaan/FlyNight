"""Update the generated SAM 2 branch of the Sarov notes."""
from pathlib import Path
import re


SOURCE = Path(__file__).resolve().parents[1] / "experiments/sarov_sam2.md"
PARENT = "1.6 Диффузия и SAM 3D"
START = "<!-- sarov-sam2:start -->"
END = "<!-- sarov-sam2:end -->"


def write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8", newline="\n")


def create_sam2_branch(notes: Path) -> Path:
    source = SOURCE.read_text(encoding="utf-8").strip()
    heading, decision, body = source.split("\n\n", 2)
    if not heading.startswith("# 1.6.1 "):
        raise ValueError("У ветви SAM 2 должен быть номер 1.6.1")
    title = heading[2:]
    parent = notes / f"{PARENT}.md"
    child = notes / f"{title}.md"
    block = f"{START}\n{decision} Продолжение — [[{notes.name}/{title}|{title}]].\n{END}"
    previous = parent.read_text(encoding="utf-8")
    managed = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if START in previous or END in previous:
        if previous.count(START) != 1 or previous.count(END) != 1 or not managed.search(previous):
            raise ValueError(f"Повреждён блок перехода в {parent}")
        parent_text = managed.sub(lambda _: block, previous)
    else:
        main, line, following = previous.partition("\nДальше по линии")
        parent_text = main.rstrip() + "\n\n" + block + "\n"
        if line:
            parent_text += line + following
    write_if_changed(child, heading + "\n\n" + body + "\n")
    write_if_changed(parent, parent_text)
    return child
