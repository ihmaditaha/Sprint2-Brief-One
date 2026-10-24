import mlflow
from src.preprocessing import download_and_extract_dataset, get_datasets, prepare_dataset
from src.model_training import build_model, train_transfer_learning, train_fine_tuning, setup_mlflow
from src.evaluation import evaluate_both_and_register_champion

def main():
    setup_mlflow()

    data_path = download_and_extract_dataset()

    raw_train, raw_val, raw_test = get_datasets(data_path)

    train_dataset = prepare_dataset(raw_train, training=True)
    val_dataset = prepare_dataset(raw_val, training=False)
    test_dataset = prepare_dataset(raw_test, training=False)

    model, base_model = build_model()

    model, _ = train_transfer_learning(model, train_dataset, val_dataset)

    model, _ = train_fine_tuning(model, base_model, train_dataset, val_dataset)

    evaluate_both_and_register_champion(test_dataset)

if __name__ == "__main__":
    main()