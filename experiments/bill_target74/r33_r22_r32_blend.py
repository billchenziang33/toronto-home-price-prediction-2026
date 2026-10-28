"""R33: fixed, no-refit equal blend of R22 and R32."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json

CARD = "R33_R22_R32_fixed_half_blend"
SCREEN_MONTHS = ("2023-03", "2024-06")
VARIANTS = {
    "off_market": ("bill_target74_r32_comparable_r22_20261025T024940Z", "r32_screen_results.json"),
    "list_aware": ("bill_target74_r32_comparable_r22_20261025T024940Z", "r32_screen_results.json"),
}

def _batch(campaign, run_id: str) -> Path:
    path = campaign.CAMPAIGN_ROOT / "batches" / run_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path

def _load_variant(campaign, run_id: str, result_name: str):
    folder = _batch(campaign, run_id)
    result_path = folder / result_name
    paired_path = folder / "paired_predictions.csv"
    manifest_path = folder / "run_manifest.json"
    for path in (result_path, paired_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("result_sha256") and manifest["result_sha256"] != _sha256(result_path):
        raise RuntimeError(f"variant result hash mismatch: {run_id}")
    expected = manifest.get("paired_predictions_sha256", manifest.get("paired_prediction_csv_sha256"))
    if not expected or expected != _sha256(paired_path):
        raise RuntimeError(f"variant prediction hash mismatch: {run_id}")
    return result, manifest, pd.read_csv(paired_path, dtype={"ml_num":str})

def run_r33_screen() -> dict:
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_greedy import runner as greedy

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R33 must be preregistered before scoring")
    parent_result, parent_path, parent_manifest = r22._verified_screen_result(campaign)
    parent_rows = pd.read_csv(parent_path.parent / "paired_predictions.csv", dtype={"ml_num":str})
    source = budget._source_identity()
    run_id = "bill_target74_r33_r22_r32_blend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    variants = {}
    for track, (variant_id, result_name) in VARIANTS.items():
        result, manifest, frame = _load_variant(campaign, variant_id, result_name)
        if result.get("data_sha256") != source["data_sha256"] or manifest.get("data_sha256") != source["data_sha256"]:
            raise RuntimeError(f"data hash mismatch for {track} variant {variant_id}")
        if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
            raise RuntimeError(f"variant {variant_id} touched a sealed label set")
        variants[track] = (variant_id, result, manifest, frame)
    campaign._append({"event":"card_start","card":CARD,"run_id":run_id,
                      "parent_candidate":parent_result["run_id"],"screen_months":list(SCREEN_MONTHS),
                      "expected_fit_starts":0,"blend_weight":{"R22":0.5,"variant":0.5},
                      "variant_runs":{track:info[0] for track,info in variants.items()},
                      "parent_result_sha256":_sha256(parent_path),
                      "parent_predictions_sha256":parent_manifest["paired_predictions_sha256"],
                      "data_sha256":source["data_sha256"],
                      "reserved_months_labels_opened":False,"hidden_labels_opened":False})
    rows_by_track={track:[] for track in VARIANTS}
    alignment_manifest=[]
    for month in SCREEN_MONTHS:
        _, _, _, truth, _ = greedy._load_fold(month)
        for track in VARIANTS:
            ref=shared.r22_reference_frame(parent_rows,truth,month,track)
            variant_id, variant_result, variant_manifest, variant_rows=variants[track]
            alt=variant_rows.loc[(variant_rows.month==month)&(variant_rows.track==track)].copy()
            if alt.ml_num.astype(str).duplicated().any():
                raise RuntimeError(f"duplicate IDs in {variant_id}/{month}/{track}")
            expected_ids=ref.ml_num.astype(str).to_numpy()
            if not np.array_equal(alt.ml_num.astype(str).to_numpy(),expected_ids):
                raise RuntimeError(f"listing order or coverage mismatch in {variant_id}/{month}/{track}")
            if not np.array_equal(alt.sold_price.to_numpy(float),ref.sold_price.to_numpy(float)):
                raise RuntimeError(f"truth labels mismatch in {variant_id}/{month}/{track}")
            if not np.array_equal(alt.parent_prediction.to_numpy(float),ref.predicted_price.to_numpy(float)):
                raise RuntimeError(f"variant parent predictions do not reproduce R22 for {month}/{track}")
            prediction=np.round(0.5*ref.predicted_price.to_numpy(float)+0.5*alt.candidate_prediction.to_numpy(float))
            candidate=pd.DataFrame({"ml_num":expected_ids,"sold_price":ref.sold_price.to_numpy(float),"predicted_price":prediction})
            paired=helpers._candidate_frame(month,track,truth,ref[["ml_num","sold_price","predicted_price"]],candidate)
            rows_by_track[track].append(paired)
            alignment_manifest.append({"track":track,"month":month,"variant_run_id":variant_id,
                                       "rows":int(len(ref)),"r22_parent_predictions_equal":True,
                                       "variant_paired_sha256":variant_manifest.get("paired_predictions_sha256",variant_manifest.get("paired_prediction_csv_sha256"))})
    paired=pd.concat([frame for track_frames in rows_by_track.values() for frame in track_frames],ignore_index=True)
    paired_path=output/"paired_predictions.csv"
    paired.to_csv(paired_path,index=False)
    tracks={track:shared._track_result(pd.concat(frames,ignore_index=True)) for track,frames in rows_by_track.items()}
    result={"campaign_id":campaign.PROTOCOL["campaign_id"],"run_id":run_id,"card":CARD,
            "parent_candidate":parent_result["run_id"],"screen_months":list(SCREEN_MONTHS),
            "fit_starts":0,"blend_weight":{"parent_R22":0.5,"candidate_variant":0.5},
            "variant_runs":{track:info[0] for track,info in variants.items()},"tracks":tracks,
            "reserved_months_labels_opened":False,"hidden_labels_opened":False,
            "data_sha256":source["data_sha256"],"alignment":alignment_manifest}
    result_path=output/"r33_screen_results.json"
    _write_json(result_path,result)
    manifest={"campaign_id":campaign.PROTOCOL["campaign_id"],"run_id":run_id,
              "source_identity":source,"variant_manifests":{track:{"run_id":info[0],"manifest_sha256":_sha256(_batch(campaign,info[0])/"run_manifest.json"),"paired_predictions_sha256":info[2].get("paired_predictions_sha256",info[2].get("paired_prediction_csv_sha256")),"result_sha256":info[2].get("result_sha256")} for track,info in variants.items()},
              "parent_result_sha256":_sha256(parent_path),"parent_predictions_sha256":parent_manifest["paired_predictions_sha256"],
              "paired_predictions_sha256":_sha256(paired_path),"result_sha256":_sha256(result_path),
              "fit_starts":0,"alignment":alignment_manifest,
              "reserved_months_labels_opened":False,"hidden_labels_opened":False}
    _write_json(output/"run_manifest.json",manifest)
    lines=["# R33：冻结候选 50/50 价格空间混合","",f"run `{run_id}`；父候选 R22 `{parent_result['run_id']}`。",
           "不重新训练、不调整权重。两个赛道均将 R22 与 R32 七项可比房源候选按固定 50/50 权重混合；逐行核对 ID、售价、R22 父预测。","",
           "| 赛道 | R22 MdAPE | 混合 MdAPE | 相对改善 | MAPE变化 | 两月均改善 | 通过 |","|---|---:|---:|---:|---:|---|---|"]
    for track,score in tracks.items():
        both=all(score["monthly"][month]["relative_mdape_improvement_percent"]>0 for month in SCREEN_MONTHS)
        lines.append(f"| {track} | {score['parent']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | {score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | {'是' if both else '否'} | {'是' if score['qualifies_screen'] else '否'} |")
    lines.extend(["","逐月变化：",""])
    for track,score in tracks.items():
        for month,values in score["monthly"].items():
            lines.append(f"- {track} / {month}: MdAPE 相对改善 {values['relative_mdape_improvement_percent']:+.3f}%")
    report=output/"research_report.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    passed={track:score["qualifies_screen"] for track,score in tracks.items()}
    campaign._append({"event":"card_complete","card":CARD,"run_id":run_id,"fit_starts":0,
                      "result_path":str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),
                      "qualifies_by_track":passed,"reserved_months_labels_opened":False,"hidden_labels_opened":False})
    campaign._write_progress("r33_screen_complete",next_action=("run the preregistered R33 development confirmation for passing tracks" if any(passed.values()) else "R33 fixed blends failed; retain R22 and register a distinct experiment"),r33_result=str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),latest_fit_count=0)
    return result

def main():
    result=run_r33_screen()
    for track,score in result["tracks"].items():
        print(f"{track}: R22={score['parent']['mdape_percent']:.5f}% R33={score['candidate']['mdape_percent']:.5f}% gain={score['relative_mdape_improvement_percent']:+.3f}% screen_pass={score['qualifies_screen']}",flush=True)
    print('run_id='+result['run_id'],flush=True)

if __name__=="__main__":
    main()

