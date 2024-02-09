import unittest

from sign_language_recognition.kaggle_asl_signs import prob_to_label


class KaggleASLSignsCase(unittest.TestCase):
    def test_label_mapping(self):
        label = prob_to_label([0.1, 0.2, 0.7])
        self.assertEqual(label, "airplane")


if __name__ == '__main__':
    unittest.main()
