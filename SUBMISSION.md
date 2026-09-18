# Submission manifest

Fill this in at code freeze. Never write secret values here.

| Item | Value |
|---|---|
| Public base URL | `https://gridwise-api-production.up.railway.app` (GET /health, POST /optimize-energy). Always use https: Railway answers http:// with a 301. |
| GitHub repository | `https://github.com/ahmed3142/gridwise`. It is private during the event and made public right after the deadline. |
| Submitted commit / tag | `v1.0.0` (`git rev-parse v1.0.0`) |
| Docker image (tag) | `docker.io/ahmed3142/gridwise-llm:1.0.0` (linux/amd64). It is private until the deadline and public after. |
| Docker image (digest) | `docker.io/ahmed3142/gridwise-llm@sha256:28919fa8f8ba5f9a16ed9470792e20fa1dfde0d5c5e85431dc29a7310362ffd7` |
| Exposed port | `8080` (binds `0.0.0.0:$PORT`) |
| Required env vars | `OPENAI_API_KEY` is required. Optional: `OPENAI_MODEL`, `OPENAI_FALLBACK_MODEL`, `PORT` (see README section 4). |
| LLM provider / model | OpenAI, `gpt-5.4-mini` with fallback `gpt-4.1` (both 65/65 on the paraphrase suite) |
| README | `https://github.com/ahmed3142/gridwise/blob/main/README.md` |
| Optimizer | SciPy `linprog` with the HiGHS solver, as an exact lexicographic LP |
| 3-minute video | `<link>`, confirmed viewable without login and no longer than 3:00 |

## Verified run command for the fallback image

```bash
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=<your key> docker.io/ahmed3142/gridwise-llm:1.0.0
curl http://127.0.0.1:8080/health          # {"status":"ok"}
python scripts/judge.py --url http://127.0.0.1:8080
```

## Pre-submit checklist

- [ ] `python -m pytest -q` passes.
- [ ] `scripts/judge.py --url <public URL> --repeat 2 --concurrency 4` reports 10/10 passed, 0 non-200 responses, p95 under 5 s, and no `DEGRADED` flag.
- [ ] `GET <public URL>/version` shows the pinned `primary_model` and `llm_status.llm_calls_failed` equal to 0.
- [ ] A logged-out `docker pull` of the tag works. `docker run` without a key still returns `/health` 200.
- [ ] No secret appears in the git history: `git log -p --all | grep -nE "sk-[A-Za-z0-9_-]{20,}"` prints nothing (this also matches `sk-proj-` keys).
- [ ] The repo is private now, with a reminder set to flip it to public after 23:00.
- [ ] The video link opens in an incognito window.
- [ ] Scrub the video and any screenshots for secrets: no `.env`, no `fly secrets set` command, no dashboard with the key visible.
