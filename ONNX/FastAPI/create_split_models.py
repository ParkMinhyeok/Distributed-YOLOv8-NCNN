import onnx
import os

# --- 설정 ---
INPUT_MODEL = "yolov8n.onnx"
OUTPUT_DIR = "split_models"

# 사용자가 찾은 분할 가능한 레이어 인덱스 리스트
SPLITTABLE_INDICES = [
    0, 2, 3, 5, 6, 8, 17, 18, 20, 21, 23, 24, 26, 42, 43, 45, 215
]

# --- 메인 스크립트 ---
def create_all_split_models():
    """지정된 인덱스를 기반으로 모든 백본/헤드 모델을 생성합니다."""
    
    # 1. 원본 ONNX 모델 로드
    try:
        print(f"원본 모델 로드 중: {INPUT_MODEL}")
        model = onnx.load(INPUT_MODEL)
    except FileNotFoundError:
        print(f"[ERROR] 원본 모델 파일 '{INPUT_MODEL}'을 찾을 수 없습니다.")
        return

    # 2. 메인 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"분할된 모델들은 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

    # 3. 각 레이어의 인덱스와 출력 이름을 매핑
    index_to_output_name = {
        i: node.output[0] for i, node in enumerate(model.graph.node) if node.output
    }
    
    # 4. 분할 가능한 각 인덱스에 대해 모델 분할 실행
    final_output_name = model.graph.output[0].name # 모델의 최종 출력 이름 (e.g., 'output0')
    
    for idx in SPLITTABLE_INDICES:
        if idx not in index_to_output_name:
            print(f"[WARNING] 인덱스 {idx}에 해당하는 레이어를 찾을 수 없어 건너뜁니다.")
            continue
            
        split_output_name = index_to_output_name[idx]
        
        # 분할 지점별로 하위 폴더 생성
        split_subdir = os.path.join(OUTPUT_DIR, f"split_layer_{idx}")
        os.makedirs(split_subdir, exist_ok=True)
        
        backbone_file = os.path.join(split_subdir, "backbone.onnx")
        head_file = os.path.join(split_subdir, "head.onnx")

        print(f"\n--- Layer {idx} ({split_output_name}) 분할 작업 중 ---")

        try:
            # 5. Backbone (Client) 모델 생성
            print(f"  -> Backbone 모델 생성 중...")
            onnx.utils.extract_model(
                INPUT_MODEL,
                backbone_file,
                input_names=["images"],        # 원본 모델의 입력
                output_names=[split_output_name], # 분할 지점의 출력
            )

            # 6. Head (Server) 모델 생성
            print(f"  -> Head 모델 생성 중...")
            onnx.utils.extract_model(
                INPUT_MODEL,
                head_file,
                input_names=[split_output_name], # 분할 지점의 출력을 입력으로
                output_names=[final_output_name], # 원본 모델의 최종 출력
            )
            
            print(f"[SUCCESS] 저장 완료:")
            print(f"  - Backbone: {backbone_file}")
            print(f"  - Head: {head_file}")

        except Exception as e:
            print(f"[FAILED ] Layer {idx} ({split_output_name}) 분할 중 에러 발생: {e}")

    print("\n모든 분할 작업이 완료되었습니다.")

if __name__ == "__main__":
    create_all_split_models()