"""ACE C2PA roundtrip nodes: signer + verifier in one standalone file.

Drop into ComfyUI custom_nodes/ and restart. Requires c2patool on PATH.

End-to-end tests:
  1) Sign as us:    Load Image -> [ACE C2PA Signer].image -> signed_path -> [ACE C2PA Verifier].file_path
  2) Gemini cosign: nano banana raw_paths -> [Signer].parent_path, edited tensor -> [Signer].image
                    -> signed_path -> [Verifier].file_path
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_KEY = "/workspace/comfyui/custom_nodes/comfyui_c2pa_signer/keys/es256_private.key"
DEFAULT_CERT = "/workspace/comfyui/custom_nodes/comfyui_c2pa_signer/keys/es256_certs.pem"


def _output_dir() -> str:
    try:
        import folder_paths  # ComfyUI runtime module
        d = folder_paths.get_output_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "output")
    os.makedirs(d, exist_ok=True)
    return d


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    t = x.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _placeholder() -> torch.Tensor:
    img = Image.new("RGB", (512, 512), (100, 100, 100))
    return _pil_to_tensor(img)


def _first_path(s: str) -> str:
    """Multi-line path strings (e.g. raw_paths) -> first non-empty line."""
    for line in (s or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _c2patool_bin() -> str:
    p = shutil.which("c2patool")
    if not p:
        raise RuntimeError(
            "c2patool not found on PATH. Install it and ensure `c2patool --version` works."
        )
    return p


def _read_report(tool: str, path: str) -> Tuple[bool, dict, str]:
    """Run `c2patool <path>`. Returns (has_manifest, parsed_json_or_empty, raw_text)."""
    proc = subprocess.run([tool, path], capture_output=True, text=True, timeout=60)
    raw = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr.strip() else "")
    raw = raw.strip()
    if proc.returncode != 0:
        return False, {}, raw
    try:
        data = json.loads(proc.stdout)
        return True, data, proc.stdout.strip()
    except Exception:
        return False, {}, raw


def _summarize(data: dict) -> str:
    """Human summary with tick/cross from a c2patool manifest store report."""
    lines: List[str] = []
    state = data.get("validation_state", "")
    if state in ("Trusted",):
        lines.append(f"\u2705 Manifest found - validation_state: {state}")
    elif state in ("Valid",):
        lines.append(
            f"\u2705 Manifest found - validation_state: {state} (signature valid; cert not on trust list)"
        )
    elif state:
        lines.append(f"\u274c Manifest found but validation_state: {state}")
    else:
        lines.append("\u2705 Manifest found (no validation_state field in report)")

    active = data.get("active_manifest", "")
    manifests = data.get("manifests", {}) or {}
    if active:
        lines.append(f"Active manifest: {active}")
    lines.append(f"Manifests in store: {len(manifests)}")

    m = manifests.get(active, {}) if active else {}
    gens = m.get("claim_generator_info") or []
    if gens:
        g = gens[0]
        lines.append(f"Generator: {g.get('name', '?')} {g.get('version', '')}".strip())
    sig = m.get("signature_info") or {}
    if sig:
        lines.append(
            f"Signed by: {sig.get('issuer', sig.get('common_name', '?'))} "
            f"(alg {sig.get('alg', '?')}, time {sig.get('time', '?')})"
        )
    ingredients = m.get("ingredients") or []
    if ingredients:
        lines.append(f"Ingredients: {len(ingredients)}")
        for ing in ingredients:
            lines.append(
                f"  - {ing.get('title', ing.get('label', '?'))} "
                f"({ing.get('relationship', '?')})"
            )
    assertions = m.get("assertions") or []
    if assertions:
        lines.append(f"Assertions: {len(assertions)}")
        for a in assertions:
            lines.append(f"  - {a.get('label', '?')}")

    vs = data.get("validation_status") or []
    for v in vs:
        lines.append(f"\u26a0\ufe0f {v.get('code', '?')}: {v.get('explanation', '')}")

    return "\n".join(lines)


class AceC2PASigner:
    """Sign an IMAGE tensor or a file via c2patool; optional parent ingredient (cosign chain)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "ACE_C2PA"}),
                "manifest_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Manifest DEFINITION (what to write), e.g. {} or {\"assertions\": [...]}. NOT a verifier dump.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"forceInput": False, "tooltip": "Tensor to sign (used only if source_path is empty)"},
                ),
                "source_path": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": False,
                        "tooltip": "File to sign directly (bytes untouched before signing). Multi-line ok; first line used.",
                    },
                ),
                "parent_path": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": False,
                        "tooltip": "Parent ingredient file (cosign chain, c2patool --parent), e.g. nano banana raw_paths. Multi-line ok; first line used.",
                    },
                ),
                "private_key_path": (
                    "STRING",
                    {"default": DEFAULT_KEY, "tooltip": "PEM private key. Empty = c2patool built-in test key."},
                ),
                "cert_path": (
                    "STRING",
                    {"default": DEFAULT_CERT, "tooltip": "PEM cert chain. Empty = c2patool built-in test cert."},
                ),
                "ta_url": (
                    "STRING",
                    {"default": "", "tooltip": "Optional Time Authority URL, e.g. http://timestamp.digicert.com"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("signed_path", "report", "image")
    FUNCTION = "sign"
    CATEGORY = "Ace_C2PA"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Sign an IMAGE tensor or a file with a C2PA manifest via c2patool. "
        "parent_path chains an existing signed file in as ingredient (cosign). "
        "Empty key/cert = built-in test key."
    )

    def sign(
        self,
        filename_prefix: str = "ACE_C2PA",
        manifest_json: str = "{}",
        image: Optional[torch.Tensor] = None,
        source_path: str = "",
        parent_path: str = "",
        private_key_path: str = DEFAULT_KEY,
        cert_path: str = DEFAULT_CERT,
        ta_url: str = "",
        **kwargs,
    ) -> Tuple[str, str, torch.Tensor]:
        tool = _c2patool_bin()
        log: List[str] = []

        src = _first_path(source_path)
        parent = _first_path(parent_path)
        tmpdir = tempfile.mkdtemp(prefix="ace_c2pa_")

        try:
            # --- resolve source file ---
            if src:
                if not os.path.isfile(src):
                    raise RuntimeError(f"source_path not found: {src}")
                log.append(f"Signing file directly: {src}")
            else:
                if image is None:
                    raise RuntimeError(
                        "Nothing to sign: connect `image` or set `source_path`."
                    )
                src = os.path.join(tmpdir, "source.png")
                _tensor_to_pil(image).save(src, "PNG")
                log.append("Signing tensor (re-encoded to PNG; no prior manifest possible).")

            src_ext = os.path.splitext(src)[1].lower() or ".png"

            # --- build manifest definition ---
            try:
                manifest = json.loads(manifest_json) if manifest_json.strip() else {}
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"manifest_json is not valid JSON: {e}. "
                    "It must be a manifest DEFINITION (e.g. {}), not a verifier report."
                )
            if not isinstance(manifest, dict):
                raise RuntimeError("manifest_json must be a JSON object.")
            for bad in ("manifests", "active_manifest", "validation_status", "validation_results"):
                if bad in manifest:
                    raise RuntimeError(
                        f"manifest_json contains '{bad}' - that's a verifier REPORT, "
                        "not a manifest definition. Use {} or your own assertions."
                    )

            pk = private_key_path.strip()
            ct = cert_path.strip()
            if pk or ct:
                if not (pk and ct):
                    raise RuntimeError(
                        "Provide BOTH private_key_path and cert_path, or neither (built-in test key)."
                    )
                if not os.path.isfile(pk):
                    raise RuntimeError(f"private_key_path not found: {pk}")
                if not os.path.isfile(ct):
                    raise RuntimeError(f"cert_path not found: {ct}")
                manifest.setdefault("alg", "es256")
                manifest["private_key"] = pk
                manifest["sign_cert"] = ct
                log.append(f"Using key: {pk}")
            else:
                log.append("Using c2patool built-in TEST key (dev only).")

            if ta_url.strip():
                manifest["ta_url"] = ta_url.strip()

            manifest_file = os.path.join(tmpdir, "manifest.json")
            with open(manifest_file, "w") as f:
                json.dump(manifest, f)

            # --- output file (extension must match source) ---
            out_name = f"{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}{src_ext}"
            out_path = os.path.join(_output_dir(), out_name)

            # --- sign ---
            cmd = [tool, src, "-m", manifest_file, "-o", out_path, "-f"]
            if parent:
                if not os.path.isfile(parent):
                    raise RuntimeError(f"parent_path not found: {parent}")
                cmd += ["-p", parent]
                log.append(f"Parent ingredient: {parent}")

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"c2patool signing failed (exit {proc.returncode}).\n"
                    f"cmd: {' '.join(cmd)}\nstderr: {proc.stderr.strip()}\nstdout: {proc.stdout.strip()}"
                )
            if proc.stderr.strip():
                log.append(f"c2patool: {proc.stderr.strip()}")
            log.append(f"Signed -> {out_path}")

            # --- report on the signed file ---
            has, data, raw = _read_report(tool, out_path)
            report = raw if raw else "(no report)"

            preview = _pil_to_tensor(Image.open(out_path))
            return (out_path, "\n".join(log) + "\n\n" + report, preview)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class AceC2PAVerifier:
    """Verify a file's C2PA manifest; tick/cross summary + full manifest JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": False,
                        "tooltip": "File to verify (e.g. signed_path or raw_paths). Multi-line ok; first line used.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "forceInput": False,
                        "tooltip": "Tensor to verify (used only if file_path is empty). Tensors can never carry a manifest - expect a cross.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("summary", "manifest_json", "image")
    FUNCTION = "verify"
    CATEGORY = "Ace_C2PA"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Verify C2PA data in a file via c2patool. Summary shows \u2705/\u274c, "
        "manifest_json carries the full store report."
    )

    def verify(
        self,
        file_path: str = "",
        image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[str, str, torch.Tensor]:
        tool = _c2patool_bin()
        path = _first_path(file_path)
        tmpdir = None

        try:
            if path:
                if not os.path.isfile(path):
                    return (f"\u274c File not found: {path}", "{}", _placeholder())
                target = path
                note = f"File: {path}"
            else:
                if image is None:
                    return (
                        "\u274c Nothing to verify: set file_path or connect image.",
                        "{}",
                        _placeholder(),
                    )
                tmpdir = tempfile.mkdtemp(prefix="ace_c2pa_v_")
                target = os.path.join(tmpdir, "verify.png")
                _tensor_to_pil(image).save(target, "PNG")
                note = "Tensor input (re-encoded PNG - a manifest is impossible here by design)"

            has, data, raw = _read_report(tool, target)

            if path:
                try:
                    preview = _pil_to_tensor(Image.open(target))
                except Exception:
                    preview = _placeholder()
            else:
                preview = image if image is not None else _placeholder()

            if not has:
                return (f"\u274c No C2PA manifest found.\n{note}\n{raw}", "{}", preview)

            return (f"{_summarize(data)}\n{note}", raw, preview)

        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AceC2PASigner": AceC2PASigner,
    "AceC2PAVerifier": AceC2PAVerifier,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceC2PASigner": "ACE C2PA Signer",
    "AceC2PAVerifier": "ACE C2PA Verifier",
}
