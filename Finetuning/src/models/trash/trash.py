
class DownstreamModel_CRHead(nn.Module):
    def __init__(
        self,
        model_path: str,
        checkpoint_path_relative: str,
        num_classes: int = None,  # default None (required for regression)
    ):

        super().__init__()

        # -------------------------
        # Load config & pretrained model
        # -------------------------
        config_path = f"{model_path}/used_parameters.json"
        raw_cfg = omegaconf.OmegaConf.load(config_path)
        resolved_dict = omegaconf.OmegaConf.to_container(raw_cfg, resolve=True)
        config = omegaconf.OmegaConf.create(resolved_dict)

        self.model = hydra.utils.instantiate(config.model)
        ckpt = torch.load(os.path.join(model_path, checkpoint_path_relative), map_location="cpu")
        sd = ckpt["model_state_dict"]
        if all(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        
        # -------------------------
        # Freeze backbone, keep reconstruction layer
        # -------------------------
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # -------------------------
        # Change to fine-tuning mode: replace cr_head with MicroUNet and unfreeze it
        # -------------------------
        out_channels = self.model.final_layer[-1].out_channels
        self.model.cr_head = MicroUNet( 
            n_channels=out_channels, num_classes=num_classes, bilinear=True, activation="none")
        for p in self.model.cr_head.parameters():
            p.requires_grad = True

        self.num_classes = num_classes  # kept for compatibility

    def forward(self, timestamps: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, region_idx: torch.Tensor):
        with torch.cuda.amp.autocast(True):
            reconstruction, seg = self.model(timestamps, x0, y0, region_idx, mosaic_width=10980)
        return reconstruction, seg

