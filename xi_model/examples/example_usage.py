import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xi_model import load_default_model


def main() -> None:
    model = load_default_model()
    result = model.predict(hstar=0.2, vw=0.5, theta0=1.2, beta_over_h=6.0, clip=True)
    print(result.to_dict())


if __name__ == "__main__":
    main()
