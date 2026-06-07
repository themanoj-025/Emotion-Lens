"""CSV/JSON/PDF export helpers for session data and predictions."""
from __future__ import annotations
import csv
import json
import io
from datetime import datetime
from typing import Optional
import pandas as pd
from utils.config import EMOTIONS, EMOTION_CONFIG


def export_predictions_csv(predictions: list[dict]) -> Optional[str]:
    """Export predictions as CSV string.

    Args:
        predictions: List of prediction dicts with keys:
            emotion, confidence, probabilities (list of 7), timestamp, source.

    Returns:
        CSV string content or None if no data.
    """
    if not predictions:
        return None

    output = io.StringIO()
    fieldnames = ['timestamp', 'source', 'emotion', 'confidence', 'positivity_score'] + EMOTIONS
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for p in predictions:
        probs = p.get('probabilities', [0] * 7)
        from utils.config import positivity_score
        row = {
            'timestamp': p.get('timestamp', ''),
            'source': p.get('source', 'live'),
            'emotion': p.get('emotion', ''),
            'confidence': round(p.get('confidence', 0), 4),
            'positivity_score': round(positivity_score(probs), 4),
        }
        for i, e in enumerate(EMOTIONS):
            row[e] = round(probs[i], 4)
        writer.writerow(row)

    return output.getvalue()


def export_predictions_json(predictions: list[dict]) -> Optional[str]:
    """Export predictions as formatted JSON string."""
    if not predictions:
        return None
    return json.dumps(predictions, indent=2, default=str)


def export_session_report(predictions: list[dict]) -> Optional[str]:
    """Generate a text summary report of the session.

    Returns:
        Plain text report string.
    """
    if not predictions:
        return None

    from collections import Counter
    counts = Counter(p['emotion'] for p in predictions)
    total = len(predictions)
    dominant = counts.most_common(1)[0][0] if counts else 'N/A'

    report = f"""╔══════════════════════════════════════╗
║   EmotionLens 🎭 Session Report      ║
╠══════════════════════════════════════╣
║ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}       ║
╚══════════════════════════════════════╝

📊 Summary
──────────
Total Predictions: {total}
Dominant Emotion: {EMOTION_CONFIG.get(dominant, {}).get('emoji', '')} {dominant}

📈 Emotion Breakdown
────────────────────
"""
    for emotion, count in counts.most_common():
        pct = count / total * 100
        emoji = EMOTION_CONFIG.get(emotion, {}).get('emoji', '')
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        report += f"  {emoji} {emotion:10s} {bar} {pct:5.1f}% ({count})\n"

    # Average confidence
    avg_conf = sum(p.get('confidence', 0) for p in predictions) / total
    report += f"\n📉 Average Confidence: {avg_conf*100:.1f}%\n"

    return report


def predictions_to_dataframe(predictions: list[dict]) -> pd.DataFrame:
    """Convert predictions list to a pandas DataFrame.

    Adds positivity_score as an extra column.
    """
    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame(predictions)
    from utils.config import positivity_score
    df['positivity_score'] = df['probabilities'].apply(
        lambda probs: positivity_score(probs) if isinstance(probs, list) and len(probs) == 7 else 0
    )
    return df
