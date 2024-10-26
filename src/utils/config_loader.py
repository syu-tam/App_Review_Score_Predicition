import yaml
from types import SimpleNamespace
import os

def dict_to_namespace(d):
    """
    辞書を再帰的にSimpleNamespaceに変換し、ドット記法でアクセス可能にする。
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_config(file_paths):
    """
    指定されたYAMLファイルを読み込み、辞書として統合する。
    ファイルが見つからない場合は警告を表示し、読み込みエラー時にはエラーメッセージを表示。
    """
    config = {}
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config.update(yaml.safe_load(file))
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {file_path}")
            print(e)
            continue
    return config  # SimpleNamespaceに変換せず、辞書形式で返す

def load_all_configs():
    """
    複数の設定ファイルを読み込んで統合する。各設定は辞書に統合され、最終的にSimpleNamespace形式に変換される。
    """
    config_paths = [
        'config/config.yaml'
    ]
    config = load_config(config_paths)
    return dict_to_namespace(config)  # SimpleNamespaceに変換
