"""Build a source-first, local-only human review packet for 18 old targets.

No inference, automatic annotation, or changes to frozen screening/review files.
Run from the repository root: python -m scripts.build_training_target_review
The HTML is static: annotations must be saved separately in the blank JSON.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

from radplain.common import fingerprint, read_jsonl, write_json


DEFAULT_PACKET = "data/p0/pre-gpu-20261116/target-audit/packet.jsonl"
DEFAULT_MANIFEST = "examples/p0/real_target_audit_manifest.json"
DEFAULT_SUMMARY = "results/p0/pre-gpu-20261116/target_ai_review_summary.json"
REVIEW_PATHS = ("results/p0/pre-gpu-20261116/target_ai_review_01_12.json",
                "results/p0/pre-gpu-20261116/target_ai_review_13_18.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def indexed(rows):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate review/source IDs")
    return result


def validate_case(row, manifest_case, ai_review):
    key = row["id"]
    if fingerprint(row["report"]) != row["normalized_source_sha256"]:
        raise ValueError("Source hash mismatch: " + key)
    if text_sha(row["reference_zh"]) != row["target_sha256"]:
        raise ValueError("Target hash mismatch: " + key)
    for field in ("id", "normalized_source_sha256", "target_sha256", "bucket"):
        if not (row[field] == manifest_case[field] == ai_review[field]):
            raise ValueError("Manifest/source/AI-review binding mismatch: " + field)
    if row["flags"] != manifest_case["flags"]:
        raise ValueError("Frozen risk flags mismatch: " + key)


def prepare(root):
    manifest = json.loads((root / DEFAULT_MANIFEST).read_text())
    summary = json.loads((root / DEFAULT_SUMMARY).read_text())
    paths = (DEFAULT_PACKET, DEFAULT_MANIFEST, DEFAULT_SUMMARY, *REVIEW_PATHS)
    input_hashes = {path: sha(root / path) for path in paths}
    if input_hashes[DEFAULT_PACKET] != manifest["packet_sha256"]:
        raise ValueError("Packet changed since the original audit manifest")
    for path in (DEFAULT_PACKET, DEFAULT_MANIFEST, *REVIEW_PATHS):
        if input_hashes[path] != summary["inputs_sha256"][path]:
            raise ValueError("Input differs from the bound AI-review summary: " + path)
    rows = read_jsonl(root / DEFAULT_PACKET)
    originals = indexed(manifest["cases"])
    indexed(rows)
    ai_rows = []
    for path in REVIEW_PATHS:
        value = json.loads((root / path).read_text())
        if value["reviewer_type"] != "project_AI_not_independent_clinical" or value["human_reviews"] != 0:
            raise ValueError("Unexpected review provenance: " + path)
        ai_rows.extend(value["cases"])
    ai_reviews = indexed(ai_rows)
    if len(rows) != 18 or set(originals) != set(ai_reviews) or set(originals) != {r["id"] for r in rows}:
        raise ValueError("Expected exactly the fixed eighteen reviewed pairs")
    if [r["id"] for r in rows] != [r["id"] for r in manifest["cases"]]:
        raise ValueError("Original audit order changed")
    packet = []
    for i, row in enumerate(rows, 1):
        ai_review = ai_reviews[row["id"]]
        validate_case(row, originals[row["id"]], ai_review)
        packet.append({**row, "case_number": i, "source_sha256": text_sha(row["report"]),
                       "ai_review": ai_review})
    return packet, input_hashes


def blank_annotations(packet, input_hashes):
    return {
        "schema_version": 1,
        "purpose": "human_review_of_existing_training_targets_not_new_final_evaluation",
        "human_reviews_completed_at_build": 0,
        "inputs_sha256": input_hashes,
        "instructions_zh": [
            "先复制本模板另存为annotations.json；此HTML不会保存输入或统计完成数。",
            "先只读源文，填写must_retain_facts及证据，再展开目标，最后查看AI提示。",
            "事实条目可用fact_id、source_evidence、fact_zh、required及判断理由；不要直接复制AI意见。",
            "明确矛盾、未保留内容与无法判断可以同时存在，不能相加当作不同案例。",
            "只有审阅者实际完成后才填写reviewer及时间，并将human_review_completed改为true。",
            "事实标签初始null表示尚未判断，不等于false；不懂的医学术语或原文歧义保留为无法判断。",
            "target_review.status可填source_error、incomplete、unjudgeable或no_observed_issue；最后一项不等于临床正确。",
            "flag_assessment可填supported、false_alarm、mixed、no_flag_observed_issue、no_flag_no_observed_issue或unjudgeable。",
            "本18例为旧训练候选的固定探索样本，不能据其比例估计总体错误率；不属于独立盲评。"
        ],
        "cases": [{
            "case_number": case["case_number"], "id": case["id"],
            "source_sha256": case["source_sha256"],
            "normalized_source_sha256": case["normalized_source_sha256"],
            "target_sha256": case["target_sha256"],
            "reviewer": None, "reviewer_background": None,
            "human_review_completed": False,
            "started_at": None, "finished_at": None, "active_minutes": None,
            "source_review_completed_before_target": None,
            "must_retain_facts": [], "source_unjudgeable_items": [],
            "target_review": {"status": None, "observed_factual_error": None,
                              "required_information_missing": None, "unjudgeable": None,
                              "fact_checks": [], "unsupported_claims": [],
                              "omitted_information": [], "unjudgeable_items": [],
                              "notes": None},
            "ai_hints_read": None, "flag_assessment": None,
            "agreement_or_disagreement_with_ai": None,
            "second_review_status": "not_arranged"
        } for case in packet]
    }


def render(packet):
    esc = lambda value: html.escape(str(value if value is not None else ""), quote=True)
    parts = ['''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>P0：18条训练目标人工复核包</title><style>
body{max-width:960px;margin:32px auto;padding:0 20px;font:17px/1.75 system-ui;color:#202832;background:#fafbfc}
article{background:white;border:1px solid #d8dfe7;border-radius:10px;padding:22px;margin:26px 0;scroll-margin-top:20px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.7 ui-monospace,monospace;background:#f1f4f7;padding:16px}
summary{cursor:pointer;font-weight:600;padding:12px 0}li{margin:7px 0}.note{color:#526171}.pending{color:#875009}
a{color:#1d5e96}nav{display:flex;flex-wrap:wrap;gap:8px}nav a{padding:4px 10px;border:1px solid #c5d2e1;border-radius:5px;background:white}
code{overflow-wrap:anywhere;font-size:13px}h2{margin-top:0}.steps{border-left:4px solid #5a7b9c;padding-left:18px}
@media(max-width:540px){body{padding:0 12px;margin:18px auto}article{padding:16px}pre{padding:12px}}
</style></head><body><header id="top"><h1>先读源文，再核对训练目标</h1>
<p>这是固定的18条旧训练目标，供人工复核已发现的筛查误报和漏报。它们属于探索性开发材料，不能作为新的最终测试，也不能据此估计总体错误率。</p>
<p><b>本包生成时人工完成数：0 / 18。</b>没有运行新模型。源文直接显示；目标、AI判断与规则告警默认折叠。折叠只帮助按顺序审阅，不构成独立盲评。</p>
<ol class="steps"><li>先读源文，在模板中列出必须保留的事实、对应证据和无法判断项。</li>
<li>再展开目标，对照对象、位置、数值、时间、否定、不确定性及建议语气。把矛盾、遗漏和无法判断分别记录。</li>
<li>最后展开AI提示，记录自己的同意或不同意理由；AI意见不是标准答案。</li>
<li>将<a href="annotations.template.json" download>空白人工标注JSON模板</a>另存为annotations.json，手动填写审阅者、时间和判断。此静态页不会保存批注或自动更新完成数。</li></ol>
<p class="note">不确定的医学含义应保留为无法判断；只有实际完成审阅的人才能将对应记录标为完成。原文和完整目标仅保存在本地。</p>
<h2>按案例定位</h2><nav aria-label="案例目录">''']
    for case in packet:
        n = case["case_number"]
        parts.append(f'<a href="#case-{n:02}">{n:02}</a>')
    parts.append('</nav></header><main>')
    for case in packet:
        n = case["case_number"]
        parts.append(f'<article id="case-{n:02}"><h2>案例 {n:02} / {len(packet)}</h2>')
        parts.append('<p class="note">记录ID：<code>' + esc(case["id"]) + '</code></p>')
        parts.append('<h3>第一步：源文</h3><pre>' + esc(case["report"]) + '</pre>')
        parts.append('<p class="note">先在模板的 must_retain_facts 中记录事实与证据；源文不清楚的内容放入 source_unjudgeable_items，再继续下一步。</p>')
        parts.append('<details class="target"><summary>第二步：展开原有中文训练目标</summary><pre>' +
                     esc(case["reference_zh"]) + '</pre><p class="note">这里只展示原有合成目标，未修订文本。请先独立记录判断，再看AI提示。</p></details>')
        parts.append('<details class="ai"><summary>第三步：展开既有AI意见和规则告警</summary><p class="pending">项目内AI审查，不是人工或临床金标准。以下标签没有写入空白人工模板。</p>')
        parts.append('<p><b>AI说明：</b>' + esc(case["ai_review"]["notes_zh"]) + '</p>')
        labels = {key: case["ai_review"][key] for key in
                  ("observed_factual_error", "required_information_missing", "unjudgeable",
                   "flag_assessment", "issue_types")}
        parts.append('<pre>' + esc(json.dumps(labels, ensure_ascii=False, indent=2)) + '</pre>')
        parts.append('<p>原抽样分组：<code>' + esc(case["bucket"]) + '</code></p><p>冻结规则告警：<code>' +
                     esc(", ".join(case["flags"]) if case["flags"] else "无已知规则标记；不代表事实正确") + '</code></p></details>')
        parts.append('<details class="binding"><summary>核对本例文本绑定</summary><p>源文原始UTF-8 SHA256：<code>' +
                     esc(case["source_sha256"]) + '</code></p><p>源文归一化SHA256：<code>' +
                     esc(case["normalized_source_sha256"]) + '</code></p><p>原目标UTF-8 SHA256：<code>' +
                     esc(case["target_sha256"]) + '</code></p></details>')
        navigation = ['<a href="#top">回到目录</a>']
        if n > 1:
            navigation.append(f'<a href="#case-{n-1:02}">上一例</a>')
        if n < len(packet):
            navigation.append(f'<a href="#case-{n+1:02}">下一例</a>')
        parts.append('<p>' + ' · '.join(navigation) + '</p></article>')
    parts.append('</main></body></html>')
    return ''.join(parts)


def build(root, out, public_receipt):
    root, out, public_receipt = root.resolve(), out.resolve(), public_receipt.resolve()
    if not out.is_relative_to(root / "data" / "p0"):
        raise ValueError("Raw review packets must remain under ignored data/p0/")
    if not public_receipt.is_relative_to(root / "results" / "p0"):
        raise ValueError("Public hash receipt must remain under results/p0/")
    if out.exists() or public_receipt.exists():
        raise FileExistsError("Refusing to overwrite a review packet or receipt")
    packet, input_hashes = prepare(root)
    annotations = blank_annotations(packet, input_hashes)
    out.mkdir(parents=True, exist_ok=False)
    (out / "review.html").write_text(render(packet), encoding="utf-8")
    write_json(out / "annotations.template.json", annotations)
    builder_hash = sha(Path(__file__))
    receipt = {"schema_version": 1, "reports": len(packet), "target_outputs": len(packet),
               "new_inference_calls": 0, "human_reviews_completed_at_build": 0,
               "source_first": True, "targets_and_ai_hints_initially_collapsed": True,
               "ai_labels_copied_into_human_annotations": False,
               "inputs_sha256": input_hashes, "builder_sha256": builder_hash,
               "outputs_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "receipt.json", receipt)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o600)
    # Public content is hashes only, without report/target text, AI labels or IDs.
    write_json(public_receipt, {"inputs_sha256": input_hashes, "builder_sha256": builder_hash,
               "outputs_sha256": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/p0/target-review-20261117"))
    parser.add_argument("--public-receipt", type=Path,
                        default=Path("results/p0/pre-gpu-20261116/target_review_packet_receipt.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = build(root, root / args.out, root / args.public_receipt)
    print(json.dumps({key: receipt[key] for key in
                      ("reports", "new_inference_calls", "human_reviews_completed_at_build")}, indent=2))


if __name__ == "__main__":
    main()
