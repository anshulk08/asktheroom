# Mini accuracy eval: for each object, ask "which candidate is the <name>?" over the same 8 crops. 1 call per (model, question, scale).
import sys, os, glob, time, json, base64, io, statistics, anthropic, cv2
client = anthropic.Anthropic()
files = sorted(glob.glob("crops_degr/*.jpg"), key=lambda p: int(os.path.basename(p).split("_")[0]))
truth = [os.path.basename(p).split("_", 1)[1][:-4] for p in files]
Q = {"wallet": "wallet", "keys": "set of house keys", "phone": "phone", "glasses": "reading glasses", "key_fob": "car key fob",
     "camera": "camera", "pen": "pen", "watch": "wristwatch"}
schema = {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}, "best_index": {"type": "integer"}},
          "required": ["names", "best_index"], "additionalProperties": False}
def content(scale):
    c = []
    for i, p in enumerate(files):
        im = cv2.imread(p)
        if scale != 1: im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        c += [{"type": "text", "text": f"Candidate {i}:"}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
              "data": base64.standard_b64encode(cv2.imencode(".jpg", im)[1].tobytes()).decode()}}]
    return c
for model, extra in [("claude-haiku-4-5", {}), ("claude-sonnet-5", {"thinking": {"type": "disabled"}})]:
    for scale in (1, 3):
        base = content(scale); ok = 0; ts = []; toks = []
        for obj, q in Q.items():
            t = time.perf_counter()
            r = client.messages.create(model=model, max_tokens=512, output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": base + [{"type": "text", "text": f"Overhead crops of objects on a desk. Which candidate is the {q}? Name every candidate briefly, then give best_index (-1 if none)."}]}], **extra)
            ts.append(time.perf_counter() - t); toks.append(r.usage.input_tokens)
            ans = json.loads(next(b.text for b in r.content if b.type == "text"))
            ok += ans["best_index"] == truth.index(obj)
        print(f"RESULT {model} crops x{scale}: {ok}/8 correct; median {statistics.median(ts):.2f}s (max {max(ts):.2f}); in_tok {toks[0]}; names(last) {ans['names']}", flush=True)
