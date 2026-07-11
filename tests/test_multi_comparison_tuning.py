from types import SimpleNamespace


class FakeComparisonResult:
    """Record tune_best arguments for one output target."""

    def __init__(self, target: str) -> None:
        """Initialize a target-specific call log."""

        self.target = target
        self.calls = []
        self.best_model_name = f"best-{target}"
        self.best_model = SimpleNamespace(target=target)
        self.best_params = None
        self.best_is_tuned = False

    def tune_best(self, *, n_trials=30, verbose=0, evaluate=True):
        """Record tuning settings and return one fitted-model placeholder."""

        self.calls.append(
            {
                "n_trials": n_trials,
                "verbose": verbose,
                "evaluate": evaluate,
            }
        )
        self.best_is_tuned = True
        self.best_params = {"n_trials": n_trials}
        return SimpleNamespace(target=self.target, n_trials=n_trials)


def test_multi_output_result_tunes_selected_targets_with_individual_trials() -> None:
    """Deferred tuning should support target filtering and trial mappings."""

    from malchan.models.compare import MultiOutputModelComparisonResult

    strength = FakeComparisonResult("strength")
    cost = FakeComparisonResult("cost")
    result = MultiOutputModelComparisonResult(
        results={
            "strength": strength,
            "cost": cost,
        }
    )

    tuned = result.tune_best(
        targets=["strength"],
        n_trials={"strength": 80},
        verbose=1,
        evaluate=False,
    )

    assert list(tuned) == ["strength"]
    assert tuned["strength"].n_trials == 80
    assert strength.calls == [
        {"n_trials": 80, "verbose": 1, "evaluate": False}
    ]
    assert cost.calls == []
    assert result.best_are_tuned == {
        "strength": True,
        "cost": False,
    }
