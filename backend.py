import os
import re
import sys
import json
import math
import types
import tempfile
import traceback

# ══════════════════════════════════════════════════════════
# 🔧 mmaction2 drn 모듈 버그 패치
# ══════════════════════════════════════════════════════════
def patch_mmaction_drn():
    try:
        drn_pkg = types.ModuleType("mmaction.models.localizers.drn")
        drn_drn = types.ModuleType("mmaction.models.localizers.drn.drn")
        class DRN: pass
        drn_drn.DRN = DRN
        drn_pkg.drn = drn_drn
        sys.modules["mmaction.models.localizers.drn"] = drn_pkg
        sys.modules["mmaction.models.localizers.drn.drn"] = drn_drn
        print("✅ mmaction drn 모듈 패치 완료")
    except Exception as e:
        print(f"⚠️ drn 패치 실패: {e}")

patch_mmaction_drn()

import torch
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.config import Config

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════
# 📂 경로 설정
# ══════════════════════════════════════════════════════════
BASE_DIR = "/home/ubuntu/ai-muncheol"

MODELS_CONFIG = {
    "model1": {
        "config": os.path.join(BASE_DIR, "model1_config.py"),
        "checkpoint": os.path.join(BASE_DIR, "model1.pth"),
        "label": "장소/배경",
        "db_map": "place",
    },
    "model2": {
        "config": os.path.join(BASE_DIR, "model2_config.py"),
        "checkpoint": os.path.join(BASE_DIR, "model2.pth"),
        "label": "사고 유형",
        "db_map": "type",
    },
    "model3": {
        "config": os.path.join(BASE_DIR, "model3_config.py"),
        "checkpoint": os.path.join(BASE_DIR, "model3.pth"),
        "label": "가해 차량(A)",
        "db_map": "action",
    },
    "model4": {
        "config": os.path.join(BASE_DIR, "model4_config.py"),
        "checkpoint": os.path.join(BASE_DIR, "model4.pth"),
        "label": "피해 차량(B)",
        "db_map": "action",
    },
}

# ══════════════════════════════════════════════════════════
# 🗺️ 모델 인덱스 → DB ID 매핑
# ══════════════════════════════════════════════════════════
MAP_MODEL1 = {i: v for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 13])}
MAP_MODEL2 = {i: v for i, v in enumerate([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 20, 21, 22, 23, 24,
    37, 38, 39, 40, 41, 45, 48, 49, 50, 59, 60
])}
MAP_MODEL3 = {i: v for i, v in enumerate([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 88, 89,
    90, 91, 133, 134, 135, 138, 139, 140, 144, 147, 148, 154, 169, 170, 171,
    172, 173, 174, 175, 176, 177, 178, 179
])}
MAP_MODEL4 = {i: v for i, v in enumerate([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 45, 46, 47, 50,
    52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74,
    87, 88, 89, 90, 91, 92, 139, 140, 142, 143, 146, 147, 150, 151, 165, 166,
    167, 168, 169, 170, 171, 172, 173
])}

MODEL_MAPS = {
    "model1": MAP_MODEL1,
    "model2": MAP_MODEL2,
    "model3": MAP_MODEL3,
    "model4": MAP_MODEL4,
}

# ══════════════════════════════════════════════════════════
# 📊 라벨 맵
# ══════════════════════════════════════════════════════════
LABEL_MAP_PLACE = {
    0: "직선 도로", 1: "신호 없는 교차로", 2: "신호 있는 교차로",
    3: "t자형 도로", 4: "기타 도로", 5: "주차장",
    6: "회전 교차로", 7: "고속도로"
}

LABEL_MAP_TYPE = {}
LABEL_MAP_ACTION = {}
CRASH_DF = pd.DataFrame()

def load_csv_labels():
    global CRASH_DF, LABEL_MAP_TYPE, LABEL_MAP_ACTION
    
    # 1. 가장 데이터가 확실한 파일을 리스트 맨 앞으로 (순서 중요)
    csv_candidates = [
        os.path.join(BASE_DIR, "matching.csv")
    ]
    
    df = pd.DataFrame()
    final_path = None

    for p in csv_candidates:
        if not os.path.exists(p):
            continue
            
        # 2. 인코딩 시도
        for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                temp = pd.read_csv(p, encoding=enc)
                # 3. 컬럼명 공백 제거 (매우 중요)
                temp.columns = temp.columns.str.strip()
                
                # 4. 필수 컬럼이 있는지 확인 (과실비율A가 있어야 진짜 데이터셋)
                if "과실비율A" in temp.columns and "사고장소특징_ID" in temp.columns:
                    df = temp
                    final_path = p
                    break
            except Exception:
                continue
        if not df.empty:
            break

    if df.empty:
        print("⚠️ '과실비율A' 컬럼이 포함된 유효한 CSV 파일을 찾을 수 없습니다.")
        return

    # 5. ID 컬럼 숫자형 변환
    for col in ["사고장소특징_ID", "A진행방향_ID", "B진행방향_ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    CRASH_DF = df
    
    # 6. 라벨 맵 생성
    if "사고장소특징_ID" in df.columns and "사고장소특징" in df.columns:
        LABEL_MAP_TYPE = df.groupby("사고장소특징_ID")["사고장소특징"].first().to_dict()

    if "A진행방향_ID" in df.columns:
        # A, B 진행방향 통합 맵핑
        map_a = df[["A진행방향_ID", "A진행방향"]].dropna().drop_duplicates()
        map_b = df[["B진행방향_ID", "B진행방향"]].dropna().drop_duplicates()
        map_a.columns = ["ID", "Label"]
        map_b.columns = ["ID", "Label"]
        combined = pd.concat([map_a, map_b]).drop_duplicates(subset="ID")
        LABEL_MAP_ACTION = combined.set_index("ID")["Label"].to_dict()

    print(f"✅ CSV 로드 완료 ({os.path.basename(final_path)}): {len(df)}행, 사고유형 {len(LABEL_MAP_TYPE)}개, 진행방향 {len(LABEL_MAP_ACTION)}개")

LABEL_MAPS = {
    "place": LABEL_MAP_PLACE,
    "type": LABEL_MAP_TYPE,
    "action": LABEL_MAP_ACTION,
}

# ══════════════════════════════════════════════════════════
# 🔧 Config 로드
# ══════════════════════════════════════════════════════════
def safe_load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"custom_imports\s*=\s*dict\(.*?\)\s*\n", "", text, flags=re.DOTALL)
    text = re.sub(
        r"loss_cls=dict\(\s*alpha=[\s\S]*?type='mmdet\.FocalLoss'[\s\S]*?\)",
        "loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)",
        text,
    )
    text = re.sub(r"load_from\s*=\s*'[^']*'", "load_from = None", text)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    cfg = Config.fromfile(tmp_path)
    os.remove(tmp_path)
    return cfg


# ══════════════════════════════════════════════════════════
# 🧠 Top-K 추출 (mmaction2 1.2.0 호환)
# ══════════════════════════════════════════════════════════
def extract_top_k(res, model_name="", k=3):
    if isinstance(res, (list, tuple)):
        res = res[0]

    scores = None
    attrs = [a for a in dir(res) if not a.startswith('_')]

    # 방법 1: pred_score
    if hasattr(res, 'pred_score') and scores is None:
        val = getattr(res, 'pred_score')
        if torch.is_tensor(val):
            scores = val

    # 방법 2: pred_scores → LabelData
    if hasattr(res, 'pred_scores') and scores is None:
        pred_scores = getattr(res, 'pred_scores')
        if torch.is_tensor(pred_scores):
            scores = pred_scores
        else:
            if hasattr(pred_scores, 'keys'):
                try:
                    for key in pred_scores.keys():
                        val = pred_scores[key]
                        if torch.is_tensor(val):
                            scores = val
                            break
                except Exception:
                    pass
            if scores is None and hasattr(pred_scores, 'values'):
                try:
                    for val in pred_scores.values():
                        if torch.is_tensor(val):
                            scores = val
                            break
                except Exception:
                    pass
            for attr in ['data', 'score', 'scores', 'label']:
                if scores is not None:
                    break
                if hasattr(pred_scores, attr):
                    val = getattr(pred_scores, attr)
                    if torch.is_tensor(val):
                        scores = val

    # 방법 3: fallback
    if scores is None:
        for attr_name in attrs:
            if 'score' in attr_name.lower():
                val = getattr(res, attr_name, None)
                if torch.is_tensor(val) and val.dim() >= 1:
                    scores = val
                    break

    if scores is None:
        raise ValueError(f"[{model_name}] scores 추출 실패!")

    if scores.dim() > 1:
        scores = scores.squeeze()
    scores = scores.cpu().to(torch.float64)

    print(f"  📊 [{model_name}] scores shape: {scores.shape}")
    top5 = scores.topk(min(5, len(scores)))
    print(f"  📊 [{model_name}] 상위5 값: {[f'{v:.4f}' for v in top5.values.tolist()]}")
    print(f"  📊 [{model_name}] 상위5 idx: {top5.indices.tolist()}")

    if scores.min() >= 0 and scores.max() <= 1 and scores.sum() > 0.5:
        probs = scores / scores.sum()
    else:
        probs = torch.nn.functional.softmax(scores, dim=0)

    topk_vals, topk_inds = torch.topk(probs, min(k, len(probs)))
    return topk_inds.tolist(), topk_vals.tolist()


# ══════════════════════════════════════════════════════════
# 🚀 모델 로드
# ══════════════════════════════════════════════════════════
loaded_models = {}

def load_all_models():
    global loaded_models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  디바이스: {device}")
    for key, info in MODELS_CONFIG.items():
        config_path = info["config"]
        ckpt_path = info["checkpoint"]
        if not os.path.exists(config_path):
            print(f"❌ {key}: config 없음 → {config_path}")
            continue
        if not os.path.exists(ckpt_path):
            print(f"❌ {key}: checkpoint 없음 → {ckpt_path}")
            continue
        try:
            print(f"📦 {key} ({info['label']}) 로딩 중...")
            cfg = safe_load_config(config_path)
            if not hasattr(cfg, "test_pipeline") or cfg.test_pipeline is None:
                if hasattr(cfg, "val_pipeline"):
                    cfg.test_pipeline = cfg.val_pipeline
            model = init_recognizer(cfg, ckpt_path, device=device)
            loaded_models[key] = model
            print(f"✅ {key} 로드 완료 (클래스 수: {cfg.model.cls_head.num_classes})")
        except Exception as e:
            print(f"❌ {key} 로드 실패: {e}")
            traceback.print_exc()
    print(f"\n🎉 총 {len(loaded_models)}/4 모델 로드 완료")


# ══════════════════════════════════════════════════════════
# 🌐 API
# ══════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(loaded_models.keys()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """SSE 스트리밍 방식 — 모델 하나 완료될 때마다 이벤트 전송"""
    if "video" not in request.files:
        return jsonify({"error": "영상 파일이 필요합니다"}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()
    video_path = tmp.name

    print(f"\n{'='*50}")
    print(f"📹 새 분석 요청: {video_file.filename}")
    print(f"   임시 파일: {video_path}")
    print(f"   파일 크기: {os.path.getsize(video_path) / 1024:.1f} KB")
    print(f"{'='*50}")

    def generate():
        try:
            results = {}
            candidates_data = {}
            model_keys = list(MODELS_CONFIG.keys())
            total = len(model_keys)

            for i, key in enumerate(model_keys):
                info = MODELS_CONFIG[key]
                model = loaded_models.get(key)

                # ── 모델 시작 이벤트 ──
                start_evt = {
                    "type": "model_start",
                    "model_index": i,
                    "model_key": key,
                    "model_label": info["label"],
                    "progress": int((i / total) * 100),
                }
                yield f"data: {json.dumps(start_evt, ensure_ascii=False)}\n\n"

                if not model:
                    results[key] = {"label": info["label"], "error": "모델 미로드"}
                    done_evt = {
                        "type": "model_done",
                        "model_index": i,
                        "model_key": key,
                        "model_label": info["label"],
                        "progress": int(((i + 1) / total) * 100),
                        "error": "모델 미로드",
                    }
                    yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"
                    continue

                try:
                    print(f"\n🔍 {key} ({info['label']}) 추론 시작...")
                    res = inference_recognizer(model, video_path)
                    print(f"  ✅ 추론 완료")

                    inds, probs = extract_top_k(res, model_name=key, k=3)
                    mapping = MODEL_MAPS.get(key, {})
                    db_key = info["db_map"]

                    candidates = []
                    for idx, prob in zip(inds, probs):
                        code = mapping.get(idx, idx)
                        label_map = LABEL_MAPS.get(db_key, {})
                        if code in label_map:
                            korean_label = f"{label_map[code]} ({code})"
                        else:
                            korean_label = f"Class {code}"
                        candidates.append({
                            "code": int(code),
                            "prob": float(prob),
                            "label": korean_label,
                        })

                    print(f"  🏆 [{key}] Top-1: {candidates[0]['label']} ({candidates[0]['prob']*100:.1f}%)")

                    results[key] = {"label": info["label"], "top": candidates}
                    candidates_data[key] = candidates

                    # ── 모델 완료 이벤트 (결과 포함) ──
                    done_evt = {
                        "type": "model_done",
                        "model_index": i,
                        "model_key": key,
                        "model_label": info["label"],
                        "progress": int(((i + 1) / total) * 100),
                        "result": {"label": info["label"], "top": candidates},
                    }
                    yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

                except Exception as e:
                    print(f"  ❌ [{key}] 추론 실패: {e}")
                    traceback.print_exc()
                    results[key] = {"label": info["label"], "error": str(e)}
                    done_evt = {
                        "type": "model_done",
                        "model_index": i,
                        "model_key": key,
                        "progress": int(((i + 1) / total) * 100),
                        "error": str(e),
                    }
                    yield f"data: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

            # ═══ 과실비율 계산 ═══
            fault_result = None
            alt_faults = []

            if (
                not CRASH_DF.empty
                and all(k in candidates_data for k in ["model2", "model3", "model4"])
            ):
                cand_type = candidates_data["model2"]
                cand_a = candidates_data["model3"]
                cand_b = candidates_data["model4"]

                eps = 1e-12
                combinations = []
                for t in cand_type:
                    for a in cand_a:
                        for b in cand_b:
                            log_score = (
                                math.log(max(float(t["prob"]), eps))
                                + math.log(max(float(a["prob"]), eps))
                                + math.log(max(float(b["prob"]), eps))
                            )
                            combinations.append({
                                "type": t["code"], "a": a["code"], "b": b["code"],
                                "score": math.exp(log_score),
                                "log_score": log_score,
                                "desc": f"[{t['label']}] + A[{a['label']}] + B[{b['label']}]",
                            })

                log_scores = torch.tensor([c["log_score"] for c in combinations], dtype=torch.float64)
                norm = torch.nn.functional.softmax(log_scores, dim=0).tolist()
                for c, p in zip(combinations, norm):
                    c["norm_conf"] = p

                combinations.sort(key=lambda x: x["norm_conf"], reverse=True)

                for combo in combinations:
                    match_rows = CRASH_DF[
                        (CRASH_DF["사고장소특징_ID"] == combo["type"])
                        & (CRASH_DF["A진행방향_ID"] == combo["a"])
                        & (CRASH_DF["B진행방향_ID"] == combo["b"])
                    ]
                    if not match_rows.empty:
                        row = match_rows.iloc[0]
                        fa = int(row["과실비율A"])
                        fb = int(row["과실비율B"])
                        entry = {
                            "fa": fa, "fb": fb,
                            "role_a": "가해자" if fa > fb else "피해자",
                            "role_b": "피해자" if fa > fb else "가해자",
                            "confidence": round(combo["norm_conf"] * 100, 2),
                            "desc": combo["desc"],
                            "accident_place": str(row.get("사고장소", "")),
                            "accident_feature": str(row.get("사고장소특징", "")),
                        }
                        if fault_result is None:
                            fault_result = entry
                        elif len(alt_faults) < 3:
                            alt_faults.append(entry)

            if fault_result:
                print(f"\n⚖️ 과실비율: A={fault_result['fa']}% / B={fault_result['fb']}%")
            else:
                print(f"\n⚠️ 과실비율 매칭 실패")

            # ── 최종 결과 이벤트 ──
            final_evt = {
                "type": "complete",
                "progress": 100,
                "models": results,
                "fault": fault_result,
                "alt_faults": alt_faults,
            }
            yield f"data: {json.dumps(final_evt, ensure_ascii=False)}\n\n"

        except Exception as e:
            traceback.print_exc()
            error_evt = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n"
        finally:
            try:
                os.remove(video_path)
            except Exception:
                pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════
# 🏁 서버 시작
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI 문철 백엔드 서버 v3 (SSE 실시간 진행)")
    print("=" * 60)
    load_csv_labels()
    LABEL_MAPS["type"] = LABEL_MAP_TYPE
    LABEL_MAPS["action"] = LABEL_MAP_ACTION
    load_all_models()
    print("\n" + "=" * 60)
    print("🌐 서버 실행: http://localhost:5002")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5002, debug=False)