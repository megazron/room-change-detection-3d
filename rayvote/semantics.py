"""Zero-shot labels and appearance embeddings with CLIP."""
import numpy as np
import torch

VOCAB = [
    "chair", "office chair", "stool", "table", "desk", "cardboard box", "box", "ball", "backpack",
    "bag", "suitcase", "lamp", "potted plant", "bottle", "cup", "book", "laptop", "monitor",
    "trash can", "bucket", "shoe", "pillow", "toy", "basket", "fan", "guitar", "umbrella",
    "jacket", "towel", "speaker", "keyboard", "sofa", "bed", "shelf", "cabinet", "door",
    "window", "floor", "wall", "person",
]
STRUCTURE = {"floor", "wall", "door", "window"}


def _square(im):
    """Pad to square so CLIP's centre crop does not cut off tall or wide objects."""
    from PIL import Image
    w, h = im.size
    if w == h:
        return im
    s = max(w, h)
    out = Image.new("RGB", (s, s), (128, 128, 128))
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


class Clip:
    def __init__(self, device, vocab=None):
        import open_clip
        self.device = device
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
        self.model.eval()
        self.tok = open_clip.get_tokenizer("ViT-B-32")
        self.vocab = list(vocab or VOCAB)
        with torch.no_grad():
            feats = []
            for tmpl in ("a photo of a {}.", "a photo of a {} in a room.", "a {} on the floor."):
                t = self.tok([tmpl.format(v) for v in self.vocab]).to(device)
                f = self.model.encode_text(t).float()
                feats.append(f / f.norm(dim=-1, keepdim=True))
            f = torch.stack(feats).mean(0)
            self.text = f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed(self, pil_images):
        x = torch.stack([self.pre(_square(im)) for im in pil_images]).to(self.device)
        f = self.model.encode_image(x).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    def label(self, emb):
        """Returns (label, prob, top3) for one embedding."""
        logits = 100.0 * torch.from_numpy(emb).to(self.device) @ self.text.T
        p = logits.softmax(-1).cpu().numpy()
        order = np.argsort(-p)
        top = [(self.vocab[i], float(p[i])) for i in order[:3]]
        return top[0][0], top[0][1], top
