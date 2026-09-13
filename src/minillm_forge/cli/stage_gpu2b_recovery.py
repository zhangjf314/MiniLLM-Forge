from __future__ import annotations

import json

from minillm_forge.design.gpu2b_recovery import validate


def main() -> None:
    print(json.dumps(validate(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
