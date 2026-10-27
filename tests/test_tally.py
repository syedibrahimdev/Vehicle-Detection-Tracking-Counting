from scripts.tally_counts import key_to_event, summarize


def test_lowercase_is_in_uppercase_is_out():
    assert key_to_event(ord("c")) == ("car", "IN")
    assert key_to_event(ord("M")) == ("motorcycle", "OUT")
    assert key_to_event(ord("r")) == ("rickshaw", "IN")
    assert key_to_event(ord("T")) == ("truck", "OUT")
    assert key_to_event(ord("b")) == ("bus", "IN")


def test_other_keys_and_no_key_are_ignored():
    assert key_to_event(-1) is None
    assert key_to_event(255) is None
    assert key_to_event(ord("x")) is None
    assert key_to_event(ord(" ")) is None


def test_summarize_matches_evaluate_counts_format():
    ev = [{"cls": "car", "dir": "IN"}, {"cls": "car", "dir": "OUT"},
          {"cls": "car", "dir": "IN"}, {"cls": "bus", "dir": "OUT"}]
    assert summarize(ev) == {"car": {"IN": 2, "OUT": 1}, "bus": {"IN": 0, "OUT": 1}}