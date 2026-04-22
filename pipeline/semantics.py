import cv2
import numpy as np
import torch
import torch.nn.functional as F

NEUTRAL_PROMPTS = [
    "a quiet empty room",
    "a calm outdoor scene with no people",
    "a neutral establishing shot",
    "black screen transition",
]

ACTIVE_PROMPTS = [
    "person talking intensely",
    "action scene with fast movement",
    "emotional dialogue between characters",
    "person crying or shouting",
]


def load_clip_model(device: str):
    """Load OpenCLIP ViT-H-14 once and return (model, preprocess, tokenizer)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model.eval()
    return model, preprocess, tokenizer


def precompute_text_features(model, tokenizer, device: str) -> tuple:
    """Encode neutral/active prompts once — reuse across all candidates."""
    def encode(prompts):
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    return encode(NEUTRAL_PROMPTS), encode(ACTIVE_PROMPTS)


def score_clip_neutrality(
    video_path: str,
    timestamp: float,
    model,
    preprocess,
    neutral_feats: torch.Tensor,
    active_feats: torch.Tensor,
    device: str,
) -> float:
    """
    Return cosine_similarity(frame, neutral_prompts).mean()
         - cosine_similarity(frame, active_prompts).mean()

    Positive → frame looks neutral/safe. Negative → frame looks active/dramatic.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 0.0

    from PIL import Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = model.encode_image(image_tensor)
    image_feat = F.normalize(image_feat, dim=-1)

    neutral_sim = (image_feat @ neutral_feats.T).mean().item()
    active_sim  = (image_feat @ active_feats.T).mean().item()

    return neutral_sim - active_sim
