from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def progress(note: str = "processing"):
    return Progress(
        TextColumn(f"{note} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
