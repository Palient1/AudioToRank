from __future__ import annotations

from pathlib import Path
import site
import nemo


def find_nemo_diar_yaml() -> Path:
    candidates: list[Path] = []

    roots: list[Path] = []
    for sp in site.getsitepackages():
        p = Path(sp) / "nemo"
        if p.exists():
            roots.append(p)

    user_sp = site.getusersitepackages()
    if user_sp:
        p = Path(user_sp) / "nemo"
        if p.exists():
            roots.append(p)

    nemo_root = Path(nemo.__file__).resolve().parent
    if nemo_root.exists():
        roots.append(nemo_root)

    local_cfg = Path(__file__).resolve().parent / "nemo_diarization.yaml"
    if local_cfg.exists():
        candidates.append(local_cfg)

    patterns = [
        "**/*diar*infer*.yaml",
        "**/*diar*inference*.yaml",
        "**/*diarization*infer*.yaml",
        "**/*clustering*diar*.yaml",
        "**/*diar*.yaml",
    ]

    for root in roots:
        for pat in patterns:
            candidates.extend(root.glob(pat))

    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "Не нашёл diarization YAML внутри установленного NeMo. "
            "Проверь, что пакет nemo_toolkit установлен полностью."
        )

    def score(p: Path) -> tuple[int, int]:
        s = str(p).lower()
        pri = 0
        if "infer" in s or "inference" in s:
            pri += 10
        if "diar" in s or "diarization" in s:
            pri += 10
        if "telephonic" in s:
            pri += 5
        if "meeting" in s:
            pri += 4
        if "general" in s:
            pri += 3
        if "conf" in s:
            pri += 1
        return (pri, -len(s))

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def load_nemo_diar_base_cfg(
    device: str,
    num_speakers: int = 2,
):
    from omegaconf import OmegaConf

    yaml_path = find_nemo_diar_yaml()
    cfg = OmegaConf.load(str(yaml_path))

    if "device" in cfg:
        cfg.device = device
    else:
        cfg["device"] = device

    cfg.diarizer.clustering.parameters.oracle_num_speakers = True
    cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers
    cfg.diarizer.clustering.parameters.min_num_speakers = num_speakers

    return cfg, yaml_path
