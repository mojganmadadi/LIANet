import torch
from torch.profiler import profile, record_function, ProfilerActivity

from models.LIANet import LIANetLight


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # Minimal model config (match your config_fourier.yaml)
    time_mode = "fourier_learned"
    model = LIANetLight(
        timestamp_dim=43,
        time_mode=time_mode,
        num_time_frequencies=8,
        time_input_dim=3,
        time_mlp_hidden=64,
        levels=13,
        n_min=16,
        growth=2,
        table_size=524288,
        feat_dim=128,
        complete_tile_size=20976,
        resunet_backbone_size="small",
        bilinear=True,
        out_channels=12,
        preproj_channels=128,
    ).to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    model.eval()

    B = 16
    memorized_crop_size = 128
    x0 = torch.randint(0, 20976 - memorized_crop_size, (B,), device=device)
    y0 = torch.randint(0, 20976 - memorized_crop_size, (B,), device=device)
    if time_mode == "mlp":
        t = torch.randn(B, 3, device=device)
    else:
        t = torch.randn(B, device=device)

    # Warmup (avoid first-iteration overhead in profile)
    with torch.no_grad():
        for _ in range(5):
            _ = model(t, x0, y0, memorized_crop_size=memorized_crop_size)
        if use_cuda:
            torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("model_forward"):
            with torch.no_grad():
                for _ in range(10):
                    _ = model(t, x0, y0, memorized_crop_size=memorized_crop_size)
        if use_cuda:
            torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if use_cuda else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
