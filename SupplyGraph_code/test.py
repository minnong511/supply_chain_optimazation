import os 

# 현재 스크립트가 실행되는 경로 (보통 .py가 있는 위치)
base_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Base directory: {base_dir}")