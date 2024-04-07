from PIL import Image
from pathlib import Path
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from feature_extractor import FeatureExtractor

if __name__ == "__main__":
    fe = FeatureExtractor()

    for img_path in sorted(Path("./img").glob("*.jpg")):
        print(img_path)

        # extract deep features
        feature = fe.extract(img=Image.open(img_path))

        feature_path = Path("./feature")/(img_path.stem + ".npy")
        print(feature_path)

        # save the feature
        np.save(feature_path, feature)
        