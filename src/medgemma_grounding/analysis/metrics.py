"""Classification outcome labels derived from fixed Yes/No score decisions."""


from __future__ import annotations


def outcome(record):
    return ("T" if (record["answer_score"]["decision"] == "Yes") == bool(record["label"]) else "F") + (
        "P" if record["answer_score"]["decision"] == "Yes" else "N")
