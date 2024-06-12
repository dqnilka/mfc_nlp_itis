import pandas as pd


def extract_unique_labels(file_path):
    # Чтение CSV файла
    df = pd.read_csv(file_path)

    # Извлечение столбца 'labels'
    labels = df['labels']

    # Фильтрация значений
    filtered_labels = labels[labels.notnull()]  # Удаление null значений
    value_counts = filtered_labels.value_counts()
    valid_labels = value_counts[value_counts > 2].index  # Фильтрация по количеству > 2

    # Сохранение порядка уникальных значений
    seen = set()
    unique_ordered_labels = [x for x in labels if x in valid_labels and not (x in seen or seen.add(x))]

    # Формирование словаря {номер: значение}
    label_dict = {i: label for i, label in enumerate(unique_ordered_labels)}

    return label_dict


# Пример использования
file_path = 'data/data.csv'
label_dict = extract_unique_labels(file_path)
print(label_dict)
