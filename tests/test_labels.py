from src.detection.labels import resolve_class_names

CONFIG = {0: "bus", 1: "car", 2: "motorcycle", 3: "person", 4: "rickshaw", 5: "truck"}


def test_dataset_names_map_to_canonical_config_names():
    model = {0: "Bus", 1: "Car", 2: "Motor Cycle", 3: "Person", 4: "Rickshaw", 5: "Truck"}
    assert resolve_class_names(model, CONFIG) == CONFIG


def test_coco_model_keeps_its_own_names():
    coco = {0: "person", 1: "bicycle", 2: "car"}
    assert resolve_class_names(coco, CONFIG) == coco


def test_same_size_but_different_classes_is_not_treated_as_project_model():
    other = {0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "cow", 5: "pig"}
    assert resolve_class_names(other, CONFIG) == other
