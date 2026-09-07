"""Execute frozen human image-aware open-loop plans, with resumable subsets.

The run cohort is always the five frozen questions. --indices selects work for
this invocation only, so a q1 smoke check can be extended with --smoke --resume.
No planning API calls or previous-step feedback are sent to the grounder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx
import pandas as pd
from openai import OpenAI

import prepare_H2_Human_Image_Aware_Plan_at6 as human
import test_H2_Static_Plan_at6_on_Charxiv as static

VERSION = "h2-human-image-aware-grounding-at6-v1"
MODE = "human-image-aware-plan"
CONDITION = "human-image-aware-plan@6"
K = 6
MAX_TOKENS = 12288
DEFAULT_OUTPUT = human.DEFAULT_ROOT / "Qwen3.5-27B__human-image-aware-plan-at6"
TERMINAL = {"valid", "truncated", "parse_failed"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen_plans(path):
    path = Path(path).resolve()
    if path.name != "plans.json" or not human.verify_frozen(path.parent)["valid"]:
        raise ValueError("requires plans.json in a verified human frozen directory")
    config = read_json(path.parent / "experiment_config.json")
    status = read_json(path.parent / "run_status.json")
    if (config.get("experiment_version") != human.VERSION
            or config.get("mode") != MODE or config.get("plan_only") is not True
            or config.get("budget_k") != K
            or config.get("selected_indices") != list(human.INDICES)
            or not status.get("complete") or status.get("request_failed_samples")
            or not config.get("author_anonymous_id")
            or not all(config.get("author_attestation", {}).get(f) is True
                       for f in human.ATTESTATION_FIELDS)):
        raise ValueError("incompatible or incomplete human frozen plan metadata")
    rows = read_json(path)
    if [row["index"] for row in rows] != list(human.INDICES):
        raise ValueError("frozen plan cohort/order mismatch")
    for row in rows:
        # Do not let the shared static normalizer silently truncate human text.
        if (row.get("status") != "valid"
                or static.normalize_fixed_plan(row.get("steps")) != row.get("steps")):
            raise ValueError(f"invalid or noncanonical six-step plan: q{row['index']}")
    return {row["index"]: row for row in rows}, config


def assert_output_safe(output, source):
    output, source = Path(output).resolve(), Path(source).resolve().parent
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the frozen plan directory")
    forbidden = ("repeated", "static", "oracle", "plan-only", "authoring")
    if any(word in output.name.lower() for word in forbidden):
        raise ValueError("use a separate human grounding output directory")
    for directory in [output, *output.parents]:
        if any((directory / marker).exists() for marker in ("FROZEN.json", "PLAN_FROZEN.json")):
            raise ValueError("output is within a frozen directory")


def protocol_spec():
    return {
        "version": VERSION, "condition": CONDITION, "budget_k": K,
        "open_loop": True, "previous_step_feedback": False,
        "planner_source": "independent human, frozen before execution",
        "max_tokens": MAX_TOKENS, "thinking_disabled": False,
        "temperature_explicitly_set": False,
        "system": static.STEP_SYSTEM_PROMPT, "user": static.STEP_USER_TEMPLATE,
        "extra_body": static.step_extra_body(),
        "source_hashes": {p.name: static.sha256_file(p) for p in (
            Path(__file__), Path(static.__file__),
            Path(static.__file__).with_name("test_H2_Multistep_Grounding_on_Charxiv.py"),
            Path(static.__file__).with_name("test_RIG_on_Charxiv.py"))},
    }


def load_records(output, config, plans):
    path = output / "experiment_config.json"
    if not path.is_file():
        raise ValueError("resume requires an existing experiment_config.json")
    old = read_json(path)
    for key, value in config.items():
        if key != "created_at" and old.get(key) != value:
            raise ValueError(f"cannot resume with changed {key}")
    if read_json(output / "plans.json") != list(plans.values()):
        raise ValueError("local plan snapshot differs from frozen source")
    records = {}
    for row in read_json(output / "step_outputs.json"):
        key = (row["index"], row["step_id"])
        if key in records or key[0] not in plans or key[1] not in {f"s{k}" for k in range(1, 7)}:
            raise ValueError("duplicate or unexpected saved step")
        need = plans[key[0]]["steps"][int(key[1][1:]) - 1]["evidence_need"]
        if row.get("evidence_need") != need or row.get("status") not in TERMINAL | {"request_failed"}:
            raise ValueError("saved step does not match the frozen plan/status protocol")
        records[key] = row
    return records


def save_progress(output, plans, rows, records, requested):
    ordered = [records[(i, f"s{k}")] for i in plans for k in range(1, 7)
               if (i, f"s{k}") in records]
    static.save_json_atomic(output / "step_outputs.json", ordered)
    prefixes, results = {}, []
    for index, plan in plans.items():
        sample = []
        for k in range(1, 7):
            if (index, f"s{k}") not in records:
                break
            sample.append(records[(index, f"s{k}")])
            result = static.build_prefix_result(
                index=index, question=plan["question"], question_source=plan["question_source"],
                category=int(rows[index].get("inst_category", 0)), steps=plan["steps"],
                records=sample, prefix_k=k, model_name="Qwen3.5-27B")
            result.update(prompt_version=VERSION, h2_mode=MODE, condition=CONDITION)
            prefixes.setdefault(index, {})[k] = result
            if k == K:
                results.append(result)
    static.save_prefix_outputs(output, list(plans), prefixes)
    static.save_json_atomic(output / "rig_results.json", results)
    completed = [i for i in plans if all(records.get((i, f"s{k}"), {}).get("status") in TERMINAL
                                        for k in range(1, 7))]
    failed = [i for i in plans if any(records.get((i, f"s{k}"), {}).get("status") == "request_failed"
                                     for k in range(1, 7))]
    status = {
        "experiment_version": VERSION, "mode": MODE, "condition": CONDITION,
        "budget_k": K, "plan_only": False, "requested_indices": list(plans),
        "last_invocation_indices": requested, "completed_indices": completed,
        "request_failed_indices": failed, "request_failed_samples": len(failed),
        "recorded_steps": len(ordered), "expected_steps": len(plans) * K,
        "step_status_counts": {s: sum(r["status"] == s for r in ordered)
                               for s in sorted(TERMINAL | {"request_failed"})},
        "model_failure_steps": sum(r["status"] in {"truncated", "parse_failed"} for r in ordered),
        "complete": len(completed) == len(plans),
        "invocation_complete": all(i in completed for i in requested),
        "updated_at": static.utc_now(),
    }
    static.save_json_atomic(output / "run_status.json", status)
    return prefixes, status


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=["Qwen3.5-27B"], default="Qwen3.5-27B")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices", nargs="+", type=int)
    selection.add_argument("--smoke", action="store_true")
    parser.add_argument("--plans-from", type=Path, default=human.DEFAULT_FROZEN_DIR / "plans.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tsv-path", type=Path, default=static.DEFAULT_TSV_PATH)
    parser.add_argument("--image-dir", type=Path, default=static.DEFAULT_IMAGE_DIR)
    parser.add_argument("--api-base")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plans, frozen_config = load_frozen_plans(args.plans_from)
    requested = list(plans) if args.smoke else args.indices
    if len(set(requested)) != len(requested) or not set(requested) <= set(plans):
        raise ValueError("indices must be a unique subset of the frozen cohort")
    if args.retries < 1 or args.timeout <= 0 or args.retry_wait < 0:
        raise ValueError("invalid request retry/timeout settings")
    output = args.output_dir.resolve()
    assert_output_safe(output, args.plans_from)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use --resume")
    frame = pd.read_csv(args.tsv_path, sep="\t")
    selected = frame[frame["index"].astype(int).isin(plans)]
    rows = {int(row["index"]): row for _, row in selected.iterrows()}
    if set(rows) != set(plans) or len(selected) != len(plans):
        raise ValueError("TSV cohort missing or duplicated")
    hashes = {}
    for index, plan in plans.items():
        if static.load_question(rows[index]) != (plan["question"], plan["question_source"]):
            raise ValueError(f"frozen question mismatch: q{index}")
        hashes[str(index)] = static.source_image_sha256(rows[index], args.image_dir)
    if hashes != frozen_config["input_image_sha256"]:
        raise ValueError("input images differ from frozen authoring images")
    config = {
        "experiment_version": VERSION, "condition": CONDITION, "mode": MODE,
        "model_name": args.model_name, "budget_k": K, "selected_indices": list(plans),
        "plan_only": False, "open_loop": True, "step_max_tokens": MAX_TOKENS,
        "thinking_disabled": False, "temperature_explicitly_set": False,
        "protocol_sha256": static.canonical_json_sha256(protocol_spec()),
        "api_base": static.resolve_api_base(args.model_name, args.api_base),
        "tsv_sha256": static.sha256_file(args.tsv_path), "input_image_sha256": hashes,
        "plans_from": str(args.plans_from.resolve()),
        "plans_from_sha256": static.sha256_file(args.plans_from),
        "plan_manifest_sha256": static.sha256_file(args.plans_from.parent / human.MARKER),
        "grounder_extra_body": static.step_extra_body(),
        "author_anonymous_id": frozen_config["author_anonymous_id"],
        "human_planning_cost": "not measured; this is a human feasibility condition",
        "created_at": static.utc_now(),
    }
    records = load_records(output, config, plans) if args.resume else {}
    if args.dry_run:
        print(json.dumps({**config, "invocation_indices": requested,
                          "output_dir": str(output)}, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        static.save_json_atomic(output / "experiment_config.json", config)
        static.save_json_atomic(output / "plans.json", list(plans.values()))
    save_progress(output, plans, rows, records, requested)
    with OpenAI(api_key=args.api_key, base_url=config["api_base"], max_retries=0,
                http_client=httpx.Client(timeout=args.timeout, trust_env=False)) as client:
        for index in requested:
            plan = plans[index]
            image = static.load_image(rows[index], args.image_dir)
            uri, size = static.encode_image(image)
            print(f"q{index}: execute six human open-loop steps", flush=True)
            for position, step in enumerate(plan["steps"], 1):
                key = (index, step["step_id"])
                if key in records and records[key]["status"] in TERMINAL:
                    print(f"  {step['step_id']}: skip={records[key]['status']}", flush=True)
                    continue
                messages = static.step_messages(plan["question"], step["evidence_need"], uri)
                # Hash the exact request including the encoded image, without storing credentials.
                request = {"model": args.model_name, "messages": messages,
                           "max_tokens": MAX_TOKENS, "extra_body": static.step_extra_body()}
                response, elapsed, error, attempts = static.call_model(
                    client, model_name=args.model_name, messages=messages,
                    max_tokens=MAX_TOKENS, extra_body=request["extra_body"],
                    retries=args.retries, retry_wait=args.retry_wait)
                record = {"index": index, **step, "round": position,
                          "attempts": attempts, "elapsed_seconds": elapsed,
                          "request_sha256": static.canonical_json_sha256(request),
                          "encoded_image_sha256": hashlib.sha256(uri.encode()).hexdigest(),
                          "source_image_sha256": hashes[str(index)],
                          "status": "request_failed", "elements": [], "token_usage": {}}
                if response is None:
                    record["error"] = error
                else:
                    metadata = static.response_metadata(response, elapsed)
                    parsed = static.parse_rig_response(metadata["raw_output"])
                    status = ("truncated" if metadata["finish_reason"] == "length"
                              else "valid" if parsed else "parse_failed")
                    raw = output / "raw" / f"q{index:04d}" / f"{step['step_id']}.txt"
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_text(metadata["raw_output"], encoding="utf-8")
                    record.update(metadata, status=status, elements=parsed if status == "valid" else [],
                                  diagnostic_parsed_elements=parsed if status == "truncated" else None,
                                  raw_output_path=str(raw), raw_output_sha256=static.sha256_file(raw))
                if key in records:
                    record["prior_request_failures"] = records[key].get("prior_request_failures", []) + [
                        {k: v for k, v in records[key].items() if k != "prior_request_failures"}]
                records[key] = record
                prefixes, status = save_progress(output, plans, rows, records, requested)
                print(f"  {step['step_id']}: {record['status']} boxes={len(record['elements'])}", flush=True)
            # Rebuild images on resume too, in case interruption happened after saving a step.
            prefixes, status = save_progress(output, plans, rows, records, requested)
            for k, result in prefixes.get(index, {}).items():
                static.draw_bboxes_on_image(image, result["elements"], size,
                    output / "visualizations" / f"q{index:04d}_k{k}.jpg", show=False)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["invocation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
