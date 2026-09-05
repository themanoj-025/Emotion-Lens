"""Tests for model utility constants and helpers."""


from utils.model_utils import EMOTION_CONFIG, EMOTIONS, MOOD_MUSIC_MAP


class TestEmotions:
    """Tests for EMOTIONS constant."""

    def test_has_7_emotions(self) -> None:
        assert len(EMOTIONS) == 7

    def test_emotion_names(self) -> None:
        expected = {"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"}
        assert set(EMOTIONS) == expected


class TestEmotionConfig:
    """Tests for EMOTION_CONFIG."""

    def test_all_emotions_have_config(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in EMOTION_CONFIG

    def test_config_has_required_keys(self) -> None:
        """model_utils.EMOTION_CONFIG has color, emoji, bg — NOT valence/arousal."""
        for emotion in EMOTIONS:
            config = EMOTION_CONFIG[emotion]
            assert "color" in config
            assert "emoji" in config
            assert "bg" in config

    def test_colors_are_hex(self) -> None:
        for emotion in EMOTIONS:
            color = EMOTION_CONFIG[emotion]["color"]
            assert color.startswith("#")
            assert len(color) == 7


class TestMoodMusicMap:
    """Tests for MOOD_MUSIC_MAP."""

    def test_all_emotions_have_music(self) -> None:
        for emotion in EMOTIONS:
            assert emotion in MOOD_MUSIC_MAP

    def test_music_has_spotify_and_youtube(self) -> None:
        for emotion in EMOTIONS:
            music = MOOD_MUSIC_MAP[emotion]
            assert "spotify" in music
            assert "youtube" in music
