from scripts.sweep_bytetrack_video_rate import config_name, token


def test_configuration_name_is_path_safe() -> None:
    assert token(0.05) == "0p05"
    assert config_name(0.2, 0.05, 60, 0.8) == "h0p2_l0p05_b60_m0p8"
