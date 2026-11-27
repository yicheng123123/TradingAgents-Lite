"""Build and score dated, source-backed probes without exposing the answer key to the model."""
from collections import Counter
from datetime import date
import csv
from hashlib import sha256
import json
import math
import random
from pathlib import Path
import re
import statistics

from .data import HistoricalData
from .data_v2_market import ISSUERS, atomic_write, write_json
from .runner import wilson


STRATA = (
    ("2025-08-01", "2025-10-31"), ("2025-11-01", "2025-11-19"),
    ("2025-11-20", "2025-12-31"), ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-08-31"), ("2026-09-01", "2026-11-25"),
)
EVENTS = {"1.01": "material agreement", "2.02": "earnings results", "5.02": "leadership or director change",
          "8.01": "other event"}
QUARTERS = ("first", "second", "third", "fourth")
SCORING = {
    "numeric": "absolute error no larger than the stated tolerance; units as asked",
    "exact": "trim and casefold; exact numeric means equality to rounded source answer",
    "choice": "one letter A through D only", "abstention": "exact UNKNOWN is incorrect but tracked",
    "boundary": "stratified bootstrap of balanced open and MC accuracy; first of two adjacent strata "
                "at least 20 points below the mean of strata 1 and 2; require 80% same-knot stability",
}
PROMPT = ("Answer this single dated financial recall question using only your existing knowledge. "
          "Return only the requested number, category, or letter. If unsure return UNKNOWN.")


def _rank(seed, item):
    return sha256(f"{seed}:{item}".encode()).hexdigest()


def _prices(market, first, last, feature):
    spy = market.series["SPY"]
    spy_pos = {bar.day: i for i, bar in enumerate(spy)}
    bench = {bar.day: bar for bar in spy}
    candidates = []
    for symbol in sorted(ISSUERS):
        bars = {bar.day: bar for bar in market.series[symbol] if bar.volume > 0}
        for end, bar in bars.items():
            if not first <= str(end) <= last or end not in spy_pos:
                continue
            source = market.manifest["assets"][symbol]
            provenance = {"url": source["url"], "sha256": source["raw_sha256"],
                          "retrieved_at": source["retrieved_at"]}
            if feature == "price":
                factor = math.prod(ratio for day, ratio in market.split_events.get(symbol, ()) if day > end)
                candidates.append(dict(key=f"{symbol}/{end}", ticker=symbol, date=str(end), feature=feature,
                    question=f"What was {symbol}'s US regular-session closing price on {end}, in USD per share "
                             "on the share scale quoted that day? Return a number.",
                    answer=round(bar.close * factor, 2), tolerance=0.50, sources=[provenance]))
                continue
            pos = spy_pos[end]
            if pos < 5:
                continue
            window = [point.day for point in spy[pos - 5:pos + 1]]
            if any(day not in bars for day in window):
                continue
            start = window[0]
            excess = (pos % 2 == 0)
            asset = 100 * (bar.close / bars[start].close - 1)
            index = 100 * (bench[end].close / bench[start].close - 1)
            bench_source = market.manifest["assets"]["SPY"]
            formula = "minus SPY's equivalent return, in percentage points" if excess else "in percent"
            candidates.append(dict(key=f"{symbol}/{start}/{end}", ticker=symbol, date=str(end), feature=feature,
                question=f"What was {symbol}'s split-consistent close-to-close price return from {start} to {end}, "
                         f"{formula}? Exclude dividends. Return a signed number.",
                answer=round(asset - index if excess else asset, 2), tolerance=0.50,
                target="alpha" if excess else "return", sources=[provenance] + ([{
                    "url": bench_source["url"], "sha256": bench_source["raw_sha256"],
                    "retrieved_at": bench_source["retrieved_at"]}] if excess else [])))
    return candidates


def _filing_facts(sec, first, last):
    root = sec / "fundamentals"
    manifest = json.loads((root / "manifest.json").read_text())["issuers"]
    candidates = []
    for symbol, item in sorted(manifest.items()):
        if "company_facts_sha256" not in item:
            continue
        bundle = json.loads((root / f"{symbol}.json").read_text())
        filings = bundle["filings"]
        values = {}
        for fact in bundle["facts"]:
            filing = filings[fact["accession"]]
            if (filing["form"] not in {"10-Q", "10-K"} or not first <= filing["filing_date"] <= last
                    or fact["end"] != filing["report_date"]
                    or fact["tag"] not in {"Assets", "Liabilities"} or fact["unit"] != "USD"):
                continue
            values.setdefault((fact["accession"], fact["tag"]), set()).add(fact["value"])
        for (acc, tag), amounts in sorted(values.items()):
            if len(amounts) != 1:
                continue
            filing = filings[acc]
            label = "total assets" if tag == "Assets" else "total liabilities"
            answer = round(next(iter(amounts)) / 1e9, 2)
            candidates.append(dict(key=f"{symbol}/{acc}/{tag}", ticker=symbol,
                date=filing["filing_date"], feature="filing", accession=acc, answer=answer,
                tolerance=max(0.05, round(abs(answer) * .005, 2)),
                question=f"What were {symbol}'s {label} at {filing['report_date']} as stated in the "
                         f"original SEC {filing['form']} filed on {filing['filing_date']}? Return a number in USD billions.",
                sources=[{"url": filing["source_url"], "accession": acc,
                          "company_facts_sha256": item["company_facts_sha256"],
                          "company_facts_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(filing['cik']):010}.json"}]))
    return candidates


def _events(sec, first, last):
    candidates = []
    for line in (sec / "filings.jsonl").read_text().splitlines():
        filing = json.loads(line)
        if (filing["form"] != "8-K" or not first <= filing["filing_date"] <= last
                or not filing.get("acceptance_datetime")):
            continue
        items = set(filing["items"].replace(" ", "").split(","))
        # Exclude generic 8.01 when a more substantive matching item is present.
        known = (items & (EVENTS.keys() - {"8.01"})) or (items & {"8.01"})
        if len(known) != 1:
            continue
        item, = known
        symbol, day, acc = (filing[key] for key in ("ticker", "filing_date", "accession"))
        candidates.append(dict(key=f"{symbol}/{day}/{acc}", ticker=symbol, date=day, feature="event",
            accession=acc, answer=EVENTS[item], tolerance=None,
            question=f"Which event category applies to {symbol}'s SEC 8-K filed on {day}, accession {acc}? "
                     f"Use one exact category: {', '.join(EVENTS.values())}.",
            sources=[{"url": filing["source_url"], "accession": acc, "sec_item": item,
                      "accepted_at": filing["acceptance_datetime"]}]))
    return candidates


def _macro_events(root, first, last):
    path = root / "macro" / "DFEDTARU.csv"
    source = json.loads((path.parent / "receipt.json").read_text())
    if source["sha256"] != sha256(path.read_bytes()).hexdigest():
        raise ValueError("FRED receipt mismatch")
    with path.open(newline="") as stream:
        series = list(csv.DictReader(stream))
    candidates = []
    for before, after in zip(series, series[1:]):
        day = after["observation_date"]
        if not first <= day <= last or before["DFEDTARU"] == after["DFEDTARU"]:
            continue
        try:
            value = float(after["DFEDTARU"])
            float(before["DFEDTARU"])
        except ValueError:
            continue
        candidates.append(dict(key=f"FOMC/{day}", ticker="FOMC", date=day, feature="event",
            question=f"What upper bound of the Federal Reserve federal funds target range took effect on {day}? "
                     "Return a number in percent.", answer=value, tolerance=0.01, sources=[source]))
    return candidates


def _ir_events(root, first, last):
    path = root / "ir" / "headlines.jsonl"
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest["headlines_sha256"] != sha256(path.read_bytes()).hexdigest():
        raise ValueError("IR metadata hash mismatch")
    candidates = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        day = item["source_metadata"]["published_date"]
        if not first <= day <= last:
            continue
        match = re.fullmatch(r"Apple reports (first|second|third|fourth) quarter results", item["excerpt"])
        if match:
            candidates.append(dict(key=f"Apple-IR/{day}", ticker="AAPL", date=day, feature="event",
                answer=match[1], tolerance=None, category_options=list(QUARTERS),
                question=f"Which fiscal quarter's results did Apple announce on {day}? "
                         f"Use one exact category: {', '.join(QUARTERS)}.",
                sources=[{"url": item["source_url"], "sha256": item["content_hash"],
                          "retrieved_at": item["ingested_at"], "hash_scope": "publisher metadata"}]))
    return candidates


def _select(pool, count, seed):
    if not pool:
        raise ValueError(f"Source pool is empty for {seed}")
    ranked = sorted(pool, key=lambda row: (_rank(seed, row["key"]), row["key"]))
    selected = []
    for _ in range(count):
        options = [row for row in ranked if row not in selected]
        if not options:
            raise ValueError(f"Insufficient independently sourced probes for {seed}")
        choice = min(options, key=lambda row: (
            sum(old.get("accession") == row["accession"] for old in selected) if "accession" in row else 0,
            sum(old["ticker"] == row["ticker"] or old["date"] == row["date"] for old in selected),
            sum(old["answer"] == row["answer"] for old in selected) if row["feature"] == "event" else 0,
        ))
        selected.append(choice)
    return selected


def _choices(row, seed, correct):
    answer = row["answer"]
    rng = random.Random(seed)
    if isinstance(answer, str):
        distractors = [value for value in row.get("category_options", EVENTS.values()) if value != answer]
    elif row["ticker"] == "FOMC":
        distractors = [round(answer + step, 2) for step in rng.choice(
            [(-.50, -.25, .25), (-.25, .25, .50)])]
    else:
        # Locally plausible alternatives with distance beyond scoring tolerance.
        scale = max(row["tolerance"] * 1.25, abs(answer) * .035)
        signs = rng.choice(([-1, -1, 1], [-1, 1, 1]))
        distractors = [round(answer + sign * scale * rng.uniform(1 + i * .7, 1.6 + i * .7), 2)
                       for i, sign in enumerate(signs)]
        if row["feature"] in {"price", "filing"} and min(distractors) <= 0:
            distractors = [round(answer + scale * i, 2) for i in (1, 2, 3)]
    rng.shuffle(distractors)
    distractors.insert(correct, answer)
    return distractors


def freeze_probes(market_path: Path, sec: Path, out: Path):
    market = HistoricalData(market_path)
    rows, choice_index = [], 0
    positions = [i % 4 for i in range(30)]
    random.Random("v3-c frozen choice positions").shuffle(positions)
    for idx, (first, last) in enumerate(STRATA, 1):
        counts = {"filing": 3, "price": 3 if idx % 2 else 2,
                  "return": 2 if idx % 2 else 3, "event": 2}
        pools = {feature: (_filing_facts(sec, first, last) if feature == "filing" else
                          _events(sec, first, last) if feature == "event" else _prices(market, first, last, feature))
                 for feature in counts}
        supplementary = sec.parent / "probe-sec"
        if idx == 3 and len({row["accession"] for row in pools["filing"]}) < 3:
            pools["filing"] += _filing_facts(supplementary, first, last)
        if len({row["accession"] for row in pools["filing"]}) < 3:
            raise ValueError(f"Filing stratum {idx} requires three distinct original accessions")
        sequence = []
        for feature, count in counts.items():
            special = []
            if feature == "event":
                if idx in {1, 3, 5}:
                    special = _macro_events(sec.parent, first, last)
                elif idx in {4, 6}:
                    special = _ir_events(sec.parent, first, last)
            sequence.extend(_select(pools[feature], count - bool(special), f"v3c/{idx}/{feature}"))
            if special:
                sequence.extend(_select(special, 1, f"v3c/{idx}/official-extra"))
        for pos, source in enumerate(sequence, 1):
            row = {**source, "probe_id": f"v3c-{idx}-{pos:02d}", "stratum": idx}
            use_choice = (pos + idx) % 2 == 0
            if use_choice:
                correct = positions[choice_index]
                options = _choices(row, row["probe_id"], correct)
                row.update(mode="choice", options=options, letter="ABCD"[correct])
                choice_index += 1
            else:
                row["mode"] = "open"
            rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "probes.jsonl"
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode())
    manifest = {
        "probe_sha256": sha256(path.read_bytes()).hexdigest(), "strata": STRATA, "scoring": SCORING, "prompt": PROMPT,
        "count": len(rows), "features": dict(Counter(row["feature"] for row in rows)),
        "gold_positions": dict(Counter(row["letter"] for row in rows if row["mode"] == "choice")),
        "source_manifests": {name: sha256(path.read_bytes()).hexdigest() for name, path in {
            "market": market_path / "manifest.json", "sec": sec / "source_manifest.json",
            "ir": sec.parent / "ir" / "manifest.json", "macro": sec.parent / "macro" / "receipt.json",
            "probe_sec": sec.parent / "probe-sec" / "manifest.json",
        }.items()},
    }
    write_json(out / "freeze.json", manifest)
    return manifest


def question(row):
    if row["mode"] == "open":
        return row["question"]
    choices = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", row["options"], strict=True))
    stem = re.split(r" Return a (?:signed )?number| Use one exact category:", row["question"])[0]
    return f"{stem}\nSelect exactly one letter.\n{choices}"


def score(row, response):
    value = response.strip()
    unknown = value.upper() == "UNKNOWN"
    guess = None
    normalized = "UNKNOWN" if unknown else None
    if row["mode"] == "choice":
        if value.upper() in set("ABCD"):
            guess = row["options"]["ABCD".index(value.upper())]
            normalized = value.upper()
        exact = value.upper() == row["letter"]
    elif isinstance(row["answer"], str):
        exact = value.casefold() == row["answer"].casefold()
        if value.casefold() in row.get("category_options", EVENTS.values()):
            normalized = value.casefold()
    else:
        guess = float(value) if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", value) else None
        if guess is not None and not math.isfinite(guess):
            guess = None
        if guess is not None:
            normalized = guess
        exact = guess == row["answer"]
    numeric = isinstance(row["answer"], (int, float))
    accurate = abs(guess - row["answer"]) <= row["tolerance"] if numeric and guess is not None else False
    direction = row["feature"] == "return" and row["answer"] != 0
    return dict(probe_id=row["probe_id"], stratum=row["stratum"], mode=row["mode"],
                feature=row["feature"], normalized_response=normalized,
                exact=exact, correct=exact if row["mode"] == "choice" or not numeric else accurate,
                numerical_accuracy=accurate if numeric and row["mode"] == "open" else None,
                unknown=unknown, directional=(guess is not None and guess * row["answer"] > 0) if direction else None,
                alpha_pair=[guess, row["answer"]] if row.get("target") == "alpha" and guess is not None else None)


def _metric(values):
    n = len(values)
    hits = sum(values)
    return {"hits": hits, "n": n, "rate": hits / n if n else None, "wilson_95": wilson(hits, n)}


def summarize(rows):
    def metrics(group):
        pairs = [row["alpha_pair"] for row in group if row.get("alpha_pair") is not None]
        corr = None
        if len(pairs) >= 3:
            x, y = zip(*pairs)
            if len(set(x)) > 1 and len(set(y)) > 1:
                corr = statistics.correlation(x, y)
        return {**{key: _metric([row[key] for row in group if row.get(key) is not None])
                   for key in ("correct", "exact", "numerical_accuracy", "unknown", "directional")},
                "multiple_choice": _metric([row["correct"] for row in group if row["mode"] == "choice"]),
                "open_accuracy": _metric([row["correct"] for row in group if row["mode"] == "open"]),
                "alpha_pearson": corr, "alpha_n": len(pairs)}

    grouped = [[row for row in rows if row["stratum"] == i] for i in range(1, 7)]
    boundary = {"candidate": None, "bootstrap_same_candidate": None, "identified": False}
    if all(len(group) == 10 for group in grouped):
        def knot(rates):
            reference = statistics.mean(rates[:2])
            return next((i for i in range(2, 5) if max(rates[i:i+2]) <= reference - .20), None)

        rates = [statistics.mean(row["correct"] for row in group) for group in grouped]
        detected = knot(rates)
        rng = random.Random(42)
        samples = []
        for _ in range(2000):
            samples.append(knot([statistics.mean(row["correct"] for mode in ("open", "choice")
                for pool in [[item for item in group if item["mode"] == mode]]
                for row in rng.choices(pool, k=len(pool))) for group in grouped]))
        stability = samples.count(detected) / len(samples)
        boundary.update(candidate=STRATA[detected][0] if detected is not None else None,
                        bootstrap_same_candidate=stability,
                        identified=detected is not None and stability >= .8)
    return {"overall": metrics(rows), "strata": [metrics(group) for group in grouped], "boundary": boundary}
