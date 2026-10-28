"""Eight-month confirmation for the fixed R22/R32 off-market blend."""
from __future__ import annotations
import json, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.bill_target74.r19_expert_weight import _sha256, _write_json

CARD="R33_R22_R32_fixed_half_blend"
VARIANT="R32_r22_comparable_features"
TRACK="off_market"
SCREEN_RUN_ID="bill_target74_r33_r22_r32_blend_20261025T025855Z"
R22_DEV_RUN_ID="bill_target74_r22_development_20261024T193459Z"
R32_SCREEN_RUN_ID="bill_target74_r32_comparable_r22_20261025T024940Z"
BLEND_WEIGHT=0.5

def run_development_confirmation() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_greedy import runner as greedy

    card=campaign.PROTOCOL["cards"].get(CARD,{})
    if card.get("status")!="development_confirmation_running":
        raise RuntimeError("R33 off-market development confirmation is not registered")
    if card.get("qualifies_by_track",{}).get(TRACK) is not True:
        raise RuntimeError("off-market did not pass the frozen R33 screen")
    screen_run_id=card.get("screen_run_id")
    if screen_run_id!=SCREEN_RUN_ID:
        raise RuntimeError("R33 screen run differs from the preregistered run")

    screen_folder=campaign.CAMPAIGN_ROOT/"batches"/SCREEN_RUN_ID
    screen_path=screen_folder/"r33_screen_results.json"
    screen_pairs_path=screen_folder/"paired_predictions.csv"
    screen_manifest_path=screen_folder/"run_manifest.json"
    for path in (screen_path,screen_pairs_path,screen_manifest_path):
        if not path.is_file(): raise FileNotFoundError(path)
    screen=json.loads(screen_path.read_text(encoding="utf-8"))
    screen_manifest=json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    if screen.get("run_id")!=SCREEN_RUN_ID or screen.get("tracks",{}).get(TRACK,{}).get("qualifies_screen") is not True:
        raise RuntimeError("R33 screen result does not verify the off-market pass")
    if screen_manifest.get("result_sha256")!=_sha256(screen_path) or screen_manifest.get("paired_predictions_sha256")!=_sha256(screen_pairs_path):
        raise RuntimeError("R33 screen artifacts fail their hashes")
    screen_pairs=pd.read_csv(screen_pairs_path,dtype={"ml_num":str})
    screen_pairs=screen_pairs.loc[screen_pairs.track==TRACK].copy()

    r22_dev,r22_path,r22_manifest=r22._verified_development_result(campaign,R22_DEV_RUN_ID)
    if TRACK not in r22_dev.get("eligible_tracks",[]):
        raise RuntimeError("verified R22 development result does not include off-market")
    r22_pairs_path=r22_path.parent/"paired_predictions.csv"
    r22_pairs=pd.read_csv(r22_pairs_path,dtype={"ml_num":str})
    r22_pairs=r22_pairs.loc[r22_pairs.track==TRACK].copy()
    dev_months=tuple(campaign.PROTOCOL["development_months"])
    screen_months=tuple(screen["screen_months"])
    if set(screen_months)-set(dev_months): raise RuntimeError("screen months fall outside development window")
    source=budget._source_identity()
    run_id="bill_target74_r33_offmarket_development_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output=campaign.CAMPAIGN_ROOT/"batches"/run_id
    output.mkdir(parents=True,exist_ok=False)
    source_files=[campaign.PROTOCOL_PATH,campaign.PACKAGE/"r33_r22_r32_development.py",
      campaign.PACKAGE/"r33_r22_r32_blend.py",campaign.PACKAGE/"r32_comparable_r22.py",
      campaign.PACKAGE/"r22_remarks_tfidf.py",campaign.PACKAGE/"r23_word_char_tfidf.py",
      campaign.PACKAGE/"r03_catboost.py",campaign.PACKAGE/"r04_ablation.py",
      campaign.PACKAGE/"runner.py",campaign.ROOT/"experiments/bill_greedy/runner.py",
      campaign.ROOT/"experiments/bill_greedy/history.py",campaign.ROOT/"experiments/bill_greedy/comparables.py",
      campaign.ROOT/"model_candidate_v2/features.py",campaign.ROOT/"experiments/model_optimization_v2/learners.py"]
    code_hashes={str(path.relative_to(campaign.ROOT)).replace("\\","/"):_sha256(path) for path in source_files}
    provenance=output/"provenance"; provenance.mkdir()
    shutil.copy2(campaign.PROTOCOL_PATH,provenance/"r33_preregistered_campaign_protocol.json")
    shutil.copy2(screen_path,provenance/"r33_screen_results.json")
    shutil.copy2(screen_pairs_path,provenance/"r33_screen_paired_predictions.csv")
    shutil.copy2(r22_path,provenance/"r22_development_results.json")
    shutil.copy2(r22_pairs_path,provenance/"r22_development_paired_predictions.csv")

    fit_plan=[(month,component) for month in dev_months if month not in screen_months for component in ("base","expert")]
    if len(fit_plan)!=12: raise RuntimeError(f"expected 12 non-screen off-market component fits, got {len(fit_plan)}")
    campaign._append({"event":"card_continuation_start","card":CARD,"stage":"development_confirmation",
      "run_id":run_id,"screen_run_id":SCREEN_RUN_ID,"parent_development_run_id":R22_DEV_RUN_ID,
      "track":TRACK,"development_months":list(dev_months),"new_fit_starts_expected":len(fit_plan),
      "screen_predictions_reused":True,"reserved_months_labels_opened":False,"hidden_labels_opened":False})

    fit_details=[]; component_manifest=[]; active={}
    old_fit_record=greedy._fit_record; old_variant=greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT]={"model_config_overrides":{},"include_neighbor_comparables":True}
    fit_start_count=0
    def record_fit(_fit_id_original,fit_fn):
        nonlocal fit_start_count
        month,component=active["month"],active["component"]
        fit_id=f"bill_target74:{CARD}:development:{TRACK}:{month}:{component}:{run_id}"
        fit_start_count+=1
        campaign._append({"event":"fit_start","card":CARD,"stage":"development_confirmation","run_id":run_id,
          "fit_id":fit_id,"track":TRACK,"month":month,"component":component,"data_sha256":source["data_sha256"]})
        started=time.perf_counter()
        try: fitted=fit_fn()
        except Exception as exc:
          campaign._append({"event":"fit_complete","card":CARD,"stage":"development_confirmation","run_id":run_id,
            "fit_id":fit_id,"track":TRACK,"month":month,"component":component,"status":"failed",
            "seconds":time.perf_counter()-started,"error":f"{type(exc).__name__}: {exc}"})
          raise
        component_dir=output/"component_predictions"/TRACK/month
        component_dir.mkdir(parents=True,exist_ok=True)
        component_path=component_dir/f"{component}.csv"
        pd.DataFrame({"ml_num":active["evaluation_ids"],"predicted_price":np.asarray(fitted["predictions"],dtype=float)}).to_csv(component_path,index=False)
        detail={"fit_id":fit_id,"track":TRACK,"month":month,"component":component,
          "seconds":time.perf_counter()-started,"model_family":fitted["model_family"],"model_config":fitted["model_config"],
          "target":fitted["target"],"actual_objective":fitted.get("actual_objective"),
          "training_rows":fitted["training_rows"],"evaluation_rows":fitted["evaluation_rows"],
          "tfidf_feature_count":sum(n.startswith("remarks_tfidf_") for n in fitted["feature_columns"]),
          "neighbor_comparable_feature_count":sum(n.startswith("nn_") for n in fitted["feature_columns"]),
          "feature_columns":fitted["feature_columns"],"prediction_sha256":fitted["prediction_sha256"],
          "saved_prediction_path":str(component_path.relative_to(campaign.ROOT)).replace("\\","/"),
          "saved_prediction_sha256":_sha256(component_path)}
        fit_details.append(detail)
        component_manifest.append({k:v for k,v in detail.items() if k not in ("seconds","fit_id")})
        campaign._append({"event":"fit_complete","card":CARD,"stage":"development_confirmation","run_id":run_id,**detail,"status":"complete"})
        return fitted

    greedy._fit_record=record_fit
    rows=[]
    try:
      for month in dev_months:
        listings,train,_,truth,heldout=greedy._load_fold(month)
        cutoff=pd.Period(month,freq="M").start_time
        if (pd.to_datetime(train.date_sold,errors="raise")>=cutoff).any():
          raise RuntimeError(f"R33 development training rows include sales at or after cutoff {month}")
        homes=hide_answers(heldout,TRACK)
        ref=r22_pairs.loc[r22_pairs.month.astype(str)==month].copy()
        if ref.ml_num.astype(str).duplicated().any() or not np.array_equal(ref.ml_num.astype(str).to_numpy(),homes.ml_num.astype(str).to_numpy()):
          raise RuntimeError(f"R22 development IDs do not align for {month}")
        if not np.array_equal(ref.sold_price.to_numpy(float),truth.sold_price.to_numpy(float)):
          raise RuntimeError(f"R22 development labels do not align for {month}")
        r22_prediction=ref.candidate_prediction.to_numpy(float)
        if month in screen_months:
          alt=screen_pairs.loc[screen_pairs.month.astype(str)==month].copy()
          if not np.array_equal(alt.ml_num.astype(str).to_numpy(),homes.ml_num.astype(str).to_numpy()):
            raise RuntimeError(f"R33 screen IDs do not align for {month}")
          if not np.array_equal(alt.sold_price.to_numpy(float),truth.sold_price.to_numpy(float)):
            raise RuntimeError(f"R33 screen labels do not align for {month}")
          if not np.array_equal(alt.parent_prediction.to_numpy(float),r22_prediction):
            raise RuntimeError(f"R33 screen parent is not the R22 development prediction for {month}")
          blended=alt.candidate_prediction.to_numpy(float)
          fitted_components=None
        else:
          specialist=greedy._specialist_mask(train,"Detached")
          eval_specialist=greedy._specialist_mask(homes,"Detached")
          fit_jobs=[("base",train,homes),("expert",train.loc[specialist].copy(),homes.loc[eval_specialist].copy())]
          fitted_components={}
          for component,fit_rows,eval_rows in fit_jobs:
            transformer,restore=r22._install_text_features(fit_rows,eval_rows,TRACK)
            active.update(month=month,component=component,evaluation_ids=eval_rows.ml_num.astype(str).to_numpy())
            try:
              fitted=greedy._fit_history_candidate(month,TRACK,train,eval_rows,base_variant="G06_lot_garage",
                    variant=VARIANT,fit_tag=f"R33-{component}-{month}",fit_rows=fit_rows)
            finally: restore()
            if sum(n.startswith("nn_") for n in fitted["feature_columns"])!=7:
              raise RuntimeError(f"R32 comparable block missing from {month}/{component}")
            fitted_components[component]=fitted
          r32_prediction=greedy._blend_specialist_predictions(np.round(fitted_components["base"]["predictions"]),
               np.round(fitted_components["expert"]["predictions"]),homes,"Detached",0.75)
          blended=np.round((r22_prediction+r32_prediction)*0.5)
        paired=pd.DataFrame({"month":month,"track":TRACK,"ml_num":homes.ml_num.astype(str).to_numpy(),
          "sold_price":truth.sold_price.to_numpy(float),"parent_prediction":r22_prediction,
          "candidate_prediction":np.asarray(blended,dtype=float),
          "initial_parent_prediction":ref.parent_prediction.to_numpy(float)})
        paired["parent_ape_percent"]=100*np.abs(paired.parent_prediction-paired.sold_price)/paired.sold_price
        paired["candidate_ape_percent"]=100*np.abs(paired.candidate_prediction-paired.sold_price)/paired.sold_price
        rows.append(paired)
        campaign._append({"event":"development_fold_complete","card":CARD,"stage":"development_confirmation","run_id":run_id,
          "track":TRACK,"month":month,"r22_parent_alignment":True,"screen_prediction_reused":month in screen_months,
          "date_cutoff_exclusive":cutoff.isoformat(),"reserved_months_labels_opened":False,"hidden_labels_opened":False})
    except Exception as exc:
      failure={"run_id":run_id,"card":CARD,"stage":"development_confirmation","error":f"{type(exc).__name__}: {exc}",
        "fit_starts":fit_start_count,"fit_details":fit_details,"completed_months":[str(x.month.iloc[0]) for x in rows],
        "reserved_months_labels_opened":False,"hidden_labels_opened":False}
      _write_json(output/"failure_checkpoint.json",failure)
      campaign._append({"event":"card_continuation_failed","card":CARD,"stage":"development_confirmation","run_id":run_id,
        "fit_starts":fit_start_count,"error":failure["error"]})
      campaign._write_progress("r33_development_confirmation_failed",next_action="audit R33 development checkpoint and fit ledger before any retry",
         r33_development_failure=str((output/"failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\","/"))
      raise
    finally:
      greedy._fit_record=old_fit_record
      if old_variant is None: greedy.VARIANTS.pop(VARIANT,None)
      else: greedy.VARIANTS[VARIANT]=old_variant

    if fit_start_count!=12: raise RuntimeError(f"R33 expected 12 new fits, observed {fit_start_count}")
    paired=pd.concat(rows,ignore_index=True)
    paired_path=output/"paired_predictions.csv"; paired.to_csv(paired_path,index=False)
    score=r22.score_development_confirmation(paired,months=dev_months,target_mdape_strictly_below=7.0,
       bootstrap_replicates=2000,bootstrap_seed=20261024,minimum_relative_gain_percent=0.5,
       minimum_months_improved=6,maximum_worst_month_regression_percent=5.0,maximum_mape_regression_percent=1.0)
    result={"campaign_id":campaign.PROTOCOL["campaign_id"],"run_id":run_id,"card":CARD,"stage":"development_confirmation",
      "screen_run_id":SCREEN_RUN_ID,"parent_development_run_id":R22_DEV_RUN_ID,"variant_screen_run_id":R32_SCREEN_RUN_ID,
      "track":TRACK,"development_months":list(dev_months),"fit_starts":fit_start_count,"fit_details":fit_details,
      "tracks":{TRACK:score},"reserved_months_labels_opened":False,"hidden_labels_opened":False,"data_sha256":source["data_sha256"]}
    result_path=output/"r33_off_market_development_results.json"; _write_json(result_path,result)
    manifest={"campaign_id":campaign.PROTOCOL["campaign_id"],"run_id":run_id,"source_identity":source,
      "code_sha256":code_hashes,"data_sha256":source["data_sha256"],"parent_run_id":R22_DEV_RUN_ID,
      "parent_result_sha256":_sha256(r22_path),"parent_predictions_sha256":r22_manifest["paired_predictions_sha256"],
      "screen_blend_run_id":SCREEN_RUN_ID,"screen_blend_result_sha256":_sha256(screen_path),
      "screen_blend_predictions_sha256":_sha256(screen_pairs_path),"variant_screen_run_id":R32_SCREEN_RUN_ID,
      "variant_screen_manifest_sha256":_sha256(campaign.CAMPAIGN_ROOT/"batches"/R32_SCREEN_RUN_ID/"run_manifest.json"),
      "paired_predictions_sha256":_sha256(paired_path),"result_sha256":_sha256(result_path),
      "fit_starts":fit_start_count,"fit_details":fit_details,"component_manifest":component_manifest,
      "reserved_months_labels_opened":False,"hidden_labels_opened":False}
    _write_json(output/"run_manifest.json",manifest)
    report=["# R33 off-market：R22/R32 固定 50/50 八个月开发确认","",f"运行：`{run_id}`；screen：`{SCREEN_RUN_ID}`。",
       "仅 off-market 进入确认。两个筛选月复用通过哈希核验的 R33 50/50 预测；其余六个月为 R22 + R32 七项可比特征模型，各占 50%。", "",
       f"- R22 MdAPE：{score['parent']['mdape_percent']:.5f}%",
       f"- R33 MdAPE：{score['candidate']['mdape_percent']:.5f}%",
       f"- 相对改善：{score['relative_mdape_improvement_percent']:+.3f}%",
       f"- MAPE 变化：{score['mape_relative_change_percent']:+.3f}%",
       f"- 改善月份：{score['months_improved']}/8；最差月退化：{score['worst_month_regression_percent']:.3f}%",
       f"- MdAPE 差值 95% bootstrap 区间：[{score['bootstrap_mdape_difference_percent']['lower95']:+.5f}, {score['bootstrap_mdape_difference_percent']['upper95']:+.5f}] pp",
       f"- 严格目标 MdAPE <7%：{'达到' if score['target_met'] else '未达到'}",
       f"- 开发确认晋级：{'是' if score['qualifies_development'] else '否'}", "", "| 月份 | R22 MdAPE | R33 MdAPE | 相对改善 |", "|---|---:|---:|---:|"]
    for month,v in score['monthly'].items(): report.append(f"| {month} | {v['parent']['mdape_percent']:.5f}% | {v['candidate']['mdape_percent']:.5f}% | {v['relative_mdape_improvement_percent']:+.3f}% |")
    (output/"research_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    card=campaign.PROTOCOL["cards"][CARD]
    card["development_confirmation"].update({"status":"qualified" if score["qualifies_development"] else "not_qualified",
      "run_id":run_id,"result_path":str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),
      "fit_starts":fit_start_count,"metrics":score})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    campaign._append({"event":"card_continuation_complete","card":CARD,"stage":"development_confirmation","run_id":run_id,
      "fit_starts":fit_start_count,"result_path":str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),
      "qualifies_development":score["qualifies_development"],"target_met":score["target_met"],
      "reserved_months_labels_opened":False,"hidden_labels_opened":False})
    next_action=("run fixed-origin confirmation for off-market R33" if score["qualifies_development"] else
      "R33 off-market failed development promotion; retain R22 and preregister a distinct experiment")
    campaign._write_progress("r33_development_confirmation_complete",next_action=next_action,
      r33_development_result=str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),latest_fit_count=fit_start_count)
    return result

def main():
    result=run_development_confirmation(); score=result['tracks'][TRACK]
    print(f"off_market R22={score['parent']['mdape_percent']:.5f}% R33={score['candidate']['mdape_percent']:.5f}% "
      f"gain={score['relative_mdape_improvement_percent']:+.3f}% target_met={score['target_met']} "
      f"development_pass={score['qualifies_development']}",flush=True)
    print('run_id='+result['run_id'],flush=True)
if __name__=='__main__': main()
