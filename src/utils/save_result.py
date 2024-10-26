import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

def create_unique_output_dir(base_dir, base_name):
    """
    指定した base_name をもとに一意のディレクトリを作成する。
    既存のディレクトリがある場合、連番を付加する。
    """
    output_dir = os.path.join(base_dir, base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return output_dir

    # 既存ディレクトリがある場合、連番で新しいディレクトリを作成
    counter = 1
    while True:
        new_output_dir = f"{output_dir}_{counter}"
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)
            return new_output_dir
        counter += 1

def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    モデルとトークナイザーを指定ディレクトリに保存する。
    """
    model_save_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)

    tokenizer_save_dir = os.path.join(output_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_save_dir)

    print(f"Model and tokenizer saved to '{output_dir}'")

def save_metrics_to_csv(epochs, train_losses, train_accuracies, train_qwks, val_losses, val_accuracies, val_qwks, output_dir, all_train):
    """
    損失、精度、QWKなどのメトリクスをCSV形式で保存。
    all_trainがTrueの場合、検証データのメトリクスは含めない。
    """
    if all_train:
        metrics_df = pd.DataFrame({
            'epoch': epochs, 
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'train_qwk': train_qwks
        })
    else:
        metrics_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'train_qwk': train_qwks,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies,
            'val_qwk': val_qwks
        })
    metrics_save_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_save_path, sep=',', index=False)
    print(f"Metrics saved to '{metrics_save_path}'")

def save_plots(train_losses, train_accuracies, train_qwks, val_losses, val_accuracies, val_qwks, output_dir):
    """
    各メトリクスの推移をプロットし、画像として保存する。
    """
    def plot_and_save(metric, train_data, val_data, title, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(train_data, label=f"Train {metric}")
        if val_data:
            plt.plot(val_data, label=f"Val {metric}")
        plt.legend()
        plt.title(f"{title} over Epochs")
        plt.savefig(os.path.join(output_dir, filename))

    plot_and_save("Loss", train_losses, val_losses, "Loss", "loss_plot.png")
    plot_and_save("Accuracy", train_accuracies, val_accuracies, "Accuracy", "accuracy_plot.png")
    plot_and_save("QWK", train_qwks, val_qwks, "QWK", "qwk_plot.png")

    print(f"Plots saved to '{output_dir}'")

def save_combined_config(config, output_dir):
    """
    統合された設定をYAML形式で保存。
    """
    combined_config_path = os.path.join(output_dir, "combined_config.yaml")
    config_dict = namespace_to_dict(config)
    with open(combined_config_path, 'w') as file:
        yaml.dump(config_dict, file)
    print(f"Combined config saved to '{combined_config_path}'")

def namespace_to_dict(namespace):
    """
    SimpleNamespaceを再帰的に辞書形式に変換。
    """
    if isinstance(namespace, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(namespace).items()}
    elif isinstance(namespace, list):
        return [namespace_to_dict(i) for i in namespace]
    else:
        return namespace

def save_metrics(output_dir, accuracy, qwk, class_report):
    """
    メトリクスと分類レポートをテキストファイルに保存。
    """
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Final Accuracy: {accuracy}\nFinal QWK: {qwk}\n')
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

def save_predictions(output_dir, final_test_predictions_seeds):
    """
    予測結果を提出用CSVファイルとして保存。
    """
    sample_submission = pd.read_csv('data/sample_submission.csv', header=None)
    sample_submission.iloc[:, 1] = final_test_predictions_seeds
    submission_path = os.path.join(output_dir, 'submit.csv')
    sample_submission.to_csv(submission_path, index=False, header=None)
    print(f"Final ensemble predictions saved to '{submission_path}'")
    
# 最終的な評価と結果の保存
def evaluate_and_save_results(y, predictions, test_predictions, config, output_dir):
    accuracy = accuracy_score(y, predictions)
    qwk = cohen_kappa_score(y, predictions, weights='quadratic')
    print(f'Final Accuracy: {accuracy}\nFinal QWK: {qwk}')

    class_report = classification_report(y, predictions, target_names=[str(i) for i in range(config.feature_extraction.num_classes)])
    print("Classification Report:\n", class_report)
    
    save_metrics(output_dir, accuracy, qwk, class_report)
    save_predictions(output_dir, test_predictions)
