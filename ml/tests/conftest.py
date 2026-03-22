"""
ml/tests shared configuration.

sys.path 부트스트랩 및 pytest 마커를 설정합니다.
테스트 파일마다 반복되던 sys.path.insert 코드를 여기서 한 번만 수행합니다.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
