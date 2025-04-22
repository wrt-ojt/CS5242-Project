import subprocess

# 定义TODO实验配置
experiments = [
    {
        'experiment_name': 'multimodal_cross_attention_projection_cnn_linear_mlp',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'dropout_mlp': 0.5,  # 可根据需求调整
        'classifier_hidden_layers': '1024,512',  # 可根据需求调整
    },
    {
        'experiment_name': 'multimodal_cross_attention_projection_cnn_linear_mlp_optional',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '',
    },
    {
        'experiment_name': 'multimodal_cross_attention_projection_cnn_mlp',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'multimodal_cross_attention_projection_cnn_mlp_optional',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '',
    },
    {
        'experiment_name': 'multimodal_cross_attention_projection_cnn_linear_less_mlp',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'dropout_mlp': 0.3,
        'classifier_hidden_layers': '512,256',
    },
    {
        'experiment_name': 'multimodal_cross_attention_projection_cnn_linear_more_mlp',
        'modality': 'multimodal',
        'use_cross_attention': 'true',
        'dropout_mlp': 0.7,
        'classifier_hidden_layers': '2048,1024',
    },
    {
        'experiment_name': 'image_only_cnn_linear_mlp',
        'modality': 'image',
        'use_cross_attention': 'false',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'image_only_cnn_linear_mlp_optional',
        'modality': 'image',
        'use_cross_attention': 'false',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '',
    },
    {
        'experiment_name': 'text_only_cnn_linear_mlp',
        'modality': 'text',
        'use_cross_attention': 'false',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '1024,512',
    },
    {
        'experiment_name': 'text_only_cnn_linear_mlp_optional',
        'modality': 'text',
        'use_cross_attention': 'false',
        'dropout_mlp': 0.5,
        'classifier_hidden_layers': '',
    },
]

def run_experiment(experiment_config):
    """运行一个实验，调用 run_experiment.py 来执行"""
    command = [
        'python', 'run_experiment.py',
        '--experiment_name', experiment_config['experiment_name'],
        '--modality', experiment_config['modality'],
        '--use_cross_attention', experiment_config['use_cross_attention'],
        '--dropout_mlp', str(experiment_config['dropout_mlp']),
        '--classifier_hidden_layers', experiment_config['classifier_hidden_layers'],
        '--batch_size', '32',  # 使用一个固定的批次大小，可以根据需要修改
        '--num_epochs', '10',  # 设定为10个epoch
        '--learning_rate_head', '0.001',
        '--learning_rate_clip', '0.0001',
        '--weight_decay_head', '0.01',
        '--weight_decay_clip', '0.01',
        '--freeze_clip', 'false',
        '--device', 'cuda',
        '--num_workers', '4',
    ]

    print(f"Running experiment: {experiment_config['experiment_name']}")
    subprocess.run(command)

def main():
    """运行所有实验"""
    for experiment_config in experiments:
        run_experiment(experiment_config)

if __name__ == "__main__":
    main()