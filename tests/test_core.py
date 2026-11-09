import unittest
from app.main import detect_highlights

class CoreTests(unittest.TestCase):
    def test_highlight_candidate_is_bounded(self):
        transcript = [{"start": 5.0, "end": 10.0, "text": "Ternyata ini rahasia yang sangat penting!", "words": []}]
        result = detect_highlights(transcript, 30.0)
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0]["start"], 0)
        self.assertLessEqual(result[0]["end"], 30)
        self.assertGreater(result[0]["score"], 1)

    def test_empty_transcript_has_no_fake_highlights(self):
        self.assertEqual(detect_highlights([], 20.0), [])

if __name__ == "__main__":
    unittest.main()
