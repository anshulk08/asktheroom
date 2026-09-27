# Question-time naming latency: N object crops + a question -> JSON, via the Anthropic API (only key present).
# usage: python vlm_latency.py "<question>" crop1.jpg crop2.jpg ...   (env ANTHROPIC_API_KEY)
import sys, time, json, base64, statistics, anthropic
q, paths = sys.argv[1], sys.argv[2:]
client = anthropic.Anthropic()
content = []
for i, p in enumerate(paths):
    content.append({"type": "text", "text": f"Candidate {i}:"})
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(open(p, "rb").read()).decode()}})
content.append({"type": "text", "text": "These are overhead crops of objects on a desk. " + q +
                " Give a short name for every candidate, then the index of the best match (-1 if none)."})
schema = {"type": "object", "properties": {
    "names": {"type": "array", "items": {"type": "string"}},
    "best_index": {"type": "integer"}, "confidence": {"type": "number"}},
    "required": ["names", "best_index", "confidence"], "additionalProperties": False}
for model, extra in [("claude-haiku-4-5", {}), ("claude-sonnet-5", {"thinking": {"type": "disabled"}})]:
    ts = []
    for k in range(5):
        t = time.perf_counter()
        r = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": content}],
                                   output_config={"format": {"type": "json_schema", "schema": schema}}, **extra)
        ts.append(time.perf_counter() - t)
        txt = next(b.text for b in r.content if b.type == "text")
    print(f"RESULT {model}: {len(paths)} crops, wall median {statistics.median(ts):.2f}s min {min(ts):.2f}s max {max(ts):.2f}s;"
          f" in_tok {r.usage.input_tokens} out_tok {r.usage.output_tokens}; last answer {txt}", flush=True)
