from ai_test.device import get_device


def test_device():
    assert get_device().type in {"cuda", "cpu"}
