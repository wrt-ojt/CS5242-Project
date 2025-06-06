import subprocess

experiments = [
    {
        'experiment_name': 'multimodal_all_default',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'projection_dim': 256,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'multimodal_no_cnn',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'projection_dim': 256,
        'use_cnn_layer': 'false',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'multimodal_no_projection',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'projection_dim': None,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'multimodal_no_cross-attention',
        'modality': 'multimodal',
        'use_cross_attention': 'false',
        'projection_dim': 256,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'multimodal_less_mlp',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'projection_dim': 256,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '512,256',
    },
    {
        'experiment_name': 'multimodal_more_mlp',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'projection_dim': 256,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '2048,1024',
    },
    {
        'experiment_name': 'image_only_with_cnn',
        'modality': 'image',
        'use_cross_attention': 'false',
        'projection_dim': None,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'image_only_no_cnn',
        'modality': 'image',
        'use_cross_attention': 'false',
        'projection_dim': None,
        'use_cnn_layer': 'false',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'text_only_with_cnn',
        'modality': 'text',
        'use_cross_attention': 'false',
        'projection_dim': None,
        'use_cnn_layer': 'true',
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'text_only_no_cnn',
        'modality': 'text',
        'use_cross_attention': 'false',
        'projection_dim': None,
        'use_cnn_layer': 'false',
        'classifier_hidden_layers': '1024,512',
    },
]

def run_experiment(config):
    command = [
        'python', 'run_experiment.py',
        '--experiment_name', config['experiment_name'],
        '--modality', config['modality'],
        '--use_cross_attention', config['use_cross_attention'],
        '--use_cnn_layer', config['use_cnn_layer'],
        '--classifier_hidden_layers', config['classifier_hidden_layers'],
    ]

    if config['projection_dim'] is not None:
        command += ['--projection_dim', str(config['projection_dim'])]

    print(f"Running experiment: {config['experiment_name']}")
    subprocess.run(command)

def main():
    for config in experiments:
        run_experiment(config)

if __name__ == "__main__":
    main()
