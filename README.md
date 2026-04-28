# modellhorizont

Virtualize the *horizon* of model layouts (railroad / gunpla / diorama / architectural models).
Take a static model scene, separate the foreground from far background, and replace the far with
an AI-generated sky / mountains / cityscape — turning a desk into a believable horizon line.

Status: PoC stage. Issues #1–#12 track the design.

## PoC scripts (`poc/`)

| script | purpose | issue |
|---|---|---|
| `stereo_farmask.py` | uncalibrated stereo from two hand-held shots → far mask | #1 |
| `mono_farmask.py` | Depth Anything v2 monocular → far mask | #4 |

### Run

```
uv sync
uv run python3 poc/stereo_farmask.py LEFT.jpg RIGHT.jpg --out out_dir
uv run python3 poc/mono_farmask.py IMG.jpg [IMG2.jpg ...] --out out_dir
```

### Network notes

`mono_farmask.py` downloads weights from Hugging Face on first run.
If you are behind a TLS-intercepting proxy, point both
`SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` at a CA bundle that includes
the proxy's root certificate before running. On a clean home network
no extra setup is needed.
