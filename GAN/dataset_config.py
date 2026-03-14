DATASET_CONFIG = {
    "genimagepp": {
        "hf_id": "Lunahera/genimagepp",
        "split": "train",
        "label_map": {
            "real": 0,
            "stable_diffusion": 1,
            "dalle": 1,
            "midjourney": 1,
            "stylegan": 1,
            "stylegan2": 1,
        },
        "use_for": ["layer3_gan"],
        "max_samples": 20000,
    },
    "custom_product": {
        "path": "dataset/custom/",
        "label_map": {
            "real": 0,
            "expiry_edit": 1,
            "packaging_edit": 1,
            "ai_generated": 1,
            "gan_inpainted": 1,
        },
        "use_for": ["layer3_gan"],
        "max_samples": None,
    },
}
