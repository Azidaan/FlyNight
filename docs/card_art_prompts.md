# Промпты для генерации карт

## Как этим пользоваться

**Текст на картах не генерируем.** Модели изображений коверкают кириллицу — буквы
получаются кривые и наполовину латиницей. Генерируем только раму и иллюстрации,
а названия и описания накладываются шрифтом в Godot. Заодно текст можно править
без перерисовки.

Порядок:

1. Один раз сгенерировать **пустую раму** (промпт 1). Она общая для всех карт.
2. Сгенерировать **пять иллюстраций** (промпты 2.1–2.5), по одной на действие.
3. Прислать мне файлы — я нарежу раму под 9-patch и соберу панель.

Размеры: рама 512×768 (соотношение 2:3), иллюстрации квадратные 512×512.

---

## Промпт 1 — пустая рама карты

```
Empty fantasy card frame for a video game UI, vertical 2:3 aspect ratio.
Ornate carved wooden border with aged brass corner fittings and small rivets.
Empty rectangular illustration window in the upper two thirds, and an empty
parchment panel in the lower third for description text.
A circular socket in the top-left corner for a cost number.
A narrow horizontal banner across the middle for a title.

Style: stylized low-poly game art, flat colours, thick clean outlines, no gradients,
soft matte shading. Muted night palette: dark walnut wood, tarnished brass, bone
white parchment. Cohesive with KayKit and Kenney low-poly asset packs.

IMPORTANT: absolutely no text, no letters, no numbers, no runes anywhere.
All panels and the banner must be completely empty.
Plain flat neutral grey background, no scene, no shadows outside the card.
Centred, straight on, no perspective, no tilt.
```

Если рама выйдет слишком нарядной для нашей ночи — добавить в конец:
`Darker and more restrained, less gold, more worn iron and dark wood.`

---

## Промпт 2 — иллюстрации действий

Общая часть, приписывать к каждой:

```
Square game card illustration, centred single subject on a simple dark background.
Style: stylized low-poly game art, flat colours, thick clean outlines, no gradients,
soft matte shading, muted night palette with a warm light source.
Cohesive with KayKit and Kenney low-poly asset packs.
No text, no letters, no numbers. No border, no frame — artwork only.
```

### 2.1 Отдать зелье

```
A shopkeeper's hand in a rough sleeve holding out a glowing green potion bottle
with a cork and a small tag, offering it across a wooden counter edge at the bottom
of the frame. The potion glows softly and lights the hand from below.
```

### 2.2 Достать из сундука

```
An open wooden treasure chest with iron bands, seen slightly from above,
filled with glowing potion bottles in green, red and blue. Warm light spills
out of the chest. Straw and a folded cloth around the bottles.
```

### 2.3 Фонарь

```
An old iron lantern with warm orange flame inside, hanging from a hook,
casting a bright glow outward into surrounding darkness. Rain streaks visible
in the light. The lantern is the only light source in the frame.
```

### 2.4 Камень

```
A rough grey stone held ready to throw, seen from behind the thrower's hand,
with a dim skeletal silhouette and two glowing red eyes in the dark distance ahead.
Cold blue moonlight, the stone lit from the side.
```

### 2.5 Перезахват

```
A green targeting reticle with corner brackets snapping onto a hooded traveller
silhouette in the rain at night. Thin scan lines and a faint green glow around
the brackets. The figure is dark, only the reticle is bright.
```

---

## Что важно проверить в результате

- **Ни одной буквы и цифры.** Если модель всё же нарисовала — перегенерировать,
  а не замазывать: текст всплывёт по краям.
- **Один и тот же стиль у всех пяти.** Если пятая заметно отличается —
  перегенерировать её с приписанной общей частью.
- **Фон однородный**, чтобы иллюстрация чисто легла в окно рамы.
- **Никакой перспективы и наклона у рамы** — её резать под 9-patch,
  а наклонённую нарезать нельзя.

## Если понадобятся ещё карты

Действий пока пять, но при появлении новых общая часть промпта та же —
меняется только описание предмета. Держать один стиль важнее, чем разнообразие:
карты лежат рядом и любое расхождение видно сразу.
