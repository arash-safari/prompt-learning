import os.path as osp
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torch.utils.data import DataLoader

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CenterClass',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, class_centers):
        super().__init__()
        self.class_centers = class_centers  # Fixed text embeddings
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.class_centers  # Use the centers as text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CenterClass(TrainerX):
    """Modified CenterClass Trainer using class centers as text embeddings."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CENTERCLASS.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CENTERCLASS.PREC == "fp32" or cfg.TRAINER.CENTERCLASS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Computing class centers")
        self.compute_class_centers(clip_model, classnames)

        print("Building custom CLIP with fixed class centers")
        self.model = CustomCLIP(cfg, clip_model, self.class_centers)

        self.model.to(self.device)
        # No parameters to optimize unless you have other trainable components
        self.optim = build_optimizer(self.model.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("clip_model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CENTERCLASS.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def compute_class_centers(self, clip_model, classnames):
        # Collect samples for each class
        from collections import defaultdict
        embeddings_dict = defaultdict(list)

        # Assuming self.dm.dataset.train_x contains the training data
        # and that each data point is a dict with 'img' and 'label'
        train_data = self.dm.dataset.train_x

        # Collect up to 16 samples per class
        class_samples = defaultdict(list)
        for sample in train_data:
            label = sample['label']
            if len(class_samples[label]) < 16:
                class_samples[label].append(sample)

        # Prepare data for processing
        samples_to_process = []
        for samples in class_samples.values():
            samples_to_process.extend(samples)

        # Create a data loader
        data_loader = DataLoader(samples_to_process, batch_size=64, shuffle=False, num_workers=4)

        clip_model.eval()  # Ensure the model is in evaluation mode
        clip_model.to(self.device)
        with torch.no_grad():
            for batch in data_loader:
                images = batch['img'].to(self.device)
                labels = batch['label']
                image_features = clip_model.visual(images.type(clip_model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                for img_feat, label in zip(image_features, labels):
                    embeddings_dict[label.item()].append(img_feat.cpu())

        # Compute centers
        num_classes = len(classnames)
        embedding_dim = image_features.shape[-1]
        centers_tensor = torch.zeros((num_classes, embedding_dim))
        for class_label in range(num_classes):
            embeddings = embeddings_dict[class_label]
            embeddings = torch.stack(embeddings)
            center = embeddings.mean(dim=0)
            centers_tensor[class_label] = center

        self.class_centers = centers_tensor.to(self.device).type(clip_model.dtype)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.CENTERCLASS.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def save_class_centers(self, filepath):
        torch.save(self.class_centers.cpu(), filepath)

    def load_class_centers(self, filepath):
        self.class_centers = torch.load(filepath).to(self.device).type(self.model.dtype)
