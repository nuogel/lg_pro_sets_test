from others.compression.Quantization.pq_quantization.pq.utils import quantize_model_, SizeTracker

from others.compression.Quantization.pq_quantization import quantization_options


def util_quantize_model(model):
    config = quantization_options.parse_config_yaml()
    # get configuration parameters
    n_centroids_config = config["n_centroids"]
    block_sizes_config = config["block_sizes"]
    layers_to_quantize = config["layers_to_quantize"]

    # size tracker for keeping track of assignments, centroids and non-compressed sizes
    size_tracker = SizeTracker(model)

    # Quantize model by stages
    for step in range(len(layers_to_quantize)):
        # quantize model in-place
        quantized_layers = quantize_model_(
            model,
            size_tracker,
            layers_to_quantize,
            block_sizes_config,
            n_centroids_config,
            step=step,
        )
        print('Finetuning stage', step, ' quantized layers:', quantized_layers)
        print(f"{size_tracker}")

        # Don't forget to re-create/update trainer/optimizer since model parameters have changed
        # optimizer = ...

        # Finetune the centroids with your usual training loop for a few epochs
        # trainer.train_epoch()

    return model
