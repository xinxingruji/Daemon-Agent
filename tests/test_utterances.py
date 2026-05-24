from utterances import SMALL, LARGE


def test_small_not_empty():
    assert len(SMALL) > 0


def test_large_not_empty():
    assert len(LARGE) > 0


def test_small_no_duplicates():
    assert len(SMALL) == len(set(SMALL))


def test_large_no_duplicates():
    assert len(LARGE) == len(set(LARGE))
