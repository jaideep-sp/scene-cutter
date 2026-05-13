import cv2
import numpy as np
import torch
import torch.nn.functional as F

NEUTRAL_PROMPTS = [
    "an empty room with no people",
    "a wide shot of a building or house exterior",
    "a quiet landscape or scenery with no movement",
    "an empty street or park",
    "a close-up of a still object like a vase, clock, or furniture",
    "a static shot of a decorative item",
    "a blurry out-of-focus background or bokeh",
    "a black screen or very dark transition",
    "a neutral wall or ceiling",
    "a simple texture with no movement",
]

ACTIVE_PROMPTS = [
    "a person speaking with their mouth open",
    "two people looking at each other intensely",
    "a person gesturing with their hands or pointing",
    "a group of people interacting or talking",
    "a close-up of a face showing strong emotion like crying, anger, or shock",
    "a person with a focused or intense expression",
    "a dramatic reaction shot of a character",
    "text on screen, subtitles, or a logo",
    "a news lower-third or graphic overlay",
    "opening or closing credits",
    "a person walking towards the camera",
    "fast movement or an action sequence",
]


def load_clip_model(device: str):
    """Load OpenCLIP ViT-L-14 (DataComp-1B) once and return (model, preprocess, tokenizer)."""
    import open_clip
    # ViT-L-14 with DataComp-1B is more accurate and smaller/faster than ViT-H-14
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="datacomp_xl_s13b_b90k",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
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
    Samples 5 frames in a ±0.5s window around the timestamp.
    Processes all 5 frames in a SINGLE batch for maximum speed.
    Returns the MINIMUM neutrality score (pessimistic approach).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    # Sample 5 points: -0.5s, -0.25s, 0.0s, +0.25s, +0.5s
    offsets = [-0.5, -0.25, 0.0, 0.25, 0.5]
    
    from PIL import Image
    
    preprocessed_images = []
    
    for offset in offsets:
        t = max(0.0, timestamp + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Preprocess and store
        preprocessed_images.append(preprocess(pil_image))

    cap.release()

    if not preprocessed_images:
        return 0.0

    # 1. Batch all images into a single 4D tensor [N, C, H, W]
    image_batch = torch.stack(preprocessed_images).to(device)

    # 2. Single pass through the model
    with torch.no_grad():
        image_feats = model.encode_image(image_batch)
    
    # Normalize features
    image_feats = F.normalize(image_feats, dim=-1)

    # 3. Calculate scores for each frame in the batch
    # (image_feats @ neutral_feats.T) gives a [5, N_PROMPTS] matrix
    neutral_sims = (image_feats @ neutral_feats.T).mean(dim=-1)
    active_sims  = (image_feats @ active_feats.T).mean(dim=-1)
    
    scores = (neutral_sims - active_sims).cpu().numpy()

    # Return the minimum score (if any frame is active, the neutrality is low)
    return float(np.min(scores))
