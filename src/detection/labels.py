def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")


def resolve_class_names(model_names: dict, config_names: dict) -> dict:
    model_names = dict(model_names)
    same_size = len(model_names) == len(config_names)
    if same_size and all(
        _norm(model_names.get(i, "")) == _norm(name)
        for i, name in config_names.items()
    ):
        return dict(config_names)
    return model_names
