from evals.scoring import is_hit, reciprocal_rank, summarize_scores


def test_a_hit_needs_one_expected_document():
    assert is_hit(["A", "B"], ["B"])
    assert not is_hit(["A"], ["B"])


def test_reciprocal_rank_rewards_a_high_position():
    assert reciprocal_rank(["B", "A"], ["B"]) == 1.0
    assert reciprocal_rank(["A", "B"], ["B"]) == 0.5
    assert reciprocal_rank(["A"], ["B"]) == 0.0


def test_scores_are_averaged_across_questions():
    assert summarize_scores([True, False], [1.0, 0.0]) == {
        "hit_rate": 0.5,
        "mean_reciprocal_rank": 0.5,
    }


def test_no_questions_gives_zero_rather_than_an_error():
    assert summarize_scores([], [])["hit_rate"] == 0.0
