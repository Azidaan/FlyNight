extends RefCounted

func run(t: AssertBag) -> void:
	t.eq(PN.Potion.size(), 3, "три типа зелий")
	t.eq(PN.TrackState.size(), 3, "три состояния трека")
	t.eq(PN.ActionKind.size(), 5, "пять карт")
	t.eq(PN.Cloak.size(), 5, "пять цветов плаща")

	t.eq(PN.potion_name(PN.Potion.HEALING), "лечения", "имя зелья лечения")
	t.eq(PN.cloak_name(PN.Cloak.RED), "красный", "имя красного плаща")

	t.eq(PN.state_from_conf(0.95), PN.TrackState.LOCKED, "высокая уверенность — захват")
	t.eq(PN.state_from_conf(PN.CONF_LOCKED), PN.TrackState.LOCKED, "порог захвата включительно")
	t.eq(PN.state_from_conf(0.5), PN.TrackState.DRIFTING, "средняя — дрейф")
	t.eq(PN.state_from_conf(PN.CONF_DRIFTING), PN.TrackState.DRIFTING, "порог дрейфа включительно")
	t.eq(PN.state_from_conf(0.1), PN.TrackState.LOST, "низкая — потеря")
	t.eq(PN.state_from_conf(0.0), PN.TrackState.LOST, "ноль — потеря")
