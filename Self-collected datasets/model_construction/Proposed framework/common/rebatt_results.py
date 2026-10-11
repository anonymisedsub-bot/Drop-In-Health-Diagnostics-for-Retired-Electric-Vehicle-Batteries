"""Names for Proposed model training rounds."""
from operator import index


def iteration_name(round_number) -> str:
    """Name one training round; iter_1 uses only the initial labelled set."""
    try:
        number = index(round_number)
    except TypeError as exc:
        raise ValueError("round_number must be a positive integer") from exc
    if number < 1:
        raise ValueError("round_number must be a positive integer")
    return f"iter_{number}"
