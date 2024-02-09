# Sign Language Recognition Models

Sign language recognition labels lexical signs from an isolated sign video.

## Usage

```bash
pip install git+https://github.com/sign-language-processing/recognition
```

### [Kaggle ASL Signs](sign_language_recognition/kaggle_asl_signs)

The winning entry for the ASL Signs Kaggle competition.

```py
from sign_language_recognition.kaggle_asl_signs import predict

pose = ... # Load pose from a file
class_probabilities = predict(pose)
gloss = predict(pose, label=True)
```

Or in CLI, given a `.pose` file, and an ELAN file with a `SIGN` tier:

```bash
sign_language_recognition --model="kaggle_asl_signs" --pose="sign.pose" --elan="sign.eaf"
```
