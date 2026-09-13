from scripts.evaluate_botsort_video_rate import botsort_args


def test_botsort_arguments_disable_reid() -> None:
    args = botsort_args("sparseOptFlow")
    assert args.gmc_method == "sparseOptFlow"
    assert args.with_reid is False
    assert args.model == "auto"
