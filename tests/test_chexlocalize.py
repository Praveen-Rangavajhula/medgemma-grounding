import csv
import json

from PIL import Image

from medgemma_grounding.chexlocalize import find_positive_case, inspect_case


def test_loads_one_positive_localized_case(tmp_path) -> None:
    image_path = tmp_path / "CheXpert" / "val" / "patient1" / "study1" / "view1_frontal.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (8, 6)).save(image_path)

    labels_path = tmp_path / "CheXpert" / "val_labels.csv"
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Path", "Pleural Effusion"])
        writer.writeheader()
        writer.writerow(
            {
                "Path": "CheXpert-v1.0/valid/patient1/study1/view1_frontal.jpg",
                "Pleural Effusion": "1.0",
            }
        )

    annotations_path = tmp_path / "CheXlocalize" / "gt_annotations_val.json"
    annotations_path.parent.mkdir()
    annotations_path.write_text(
        json.dumps(
            {
                "patient1_study1_view1_frontal": {
                    "img_size": [6, 8],
                    "Pleural Effusion": [[[1, 2], [3, 4], [5, 2]]],
                }
            }
        )
    )

    case = find_positive_case(tmp_path)

    assert case.label == 1.0
    assert case.image_path == image_path
    assert case.contours == (((1.0, 2.0), (3.0, 4.0), (5.0, 2.0)),)
    assert inspect_case(case) == (8, 6)
