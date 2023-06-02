# <YOUR_IMPORTS>
import dill
import os
import json
import pandas as pd

path = os.environ.get('PROJECT_PATH', '../..')


def predict():
    def newest(file_dir):
        files = os.listdir(file_dir)
        paths = [os.path.join(file_dir, basename) for basename in files]
        return max(paths, key=os.path.getctime)

    pkl_path = newest(f'{path}/data/models')
    with open(pkl_path, 'rb') as file:
        model = dill.load(file)

    df_predict = pd.DataFrame(columns=['id', 'predict'])

    for filename in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', filename), 'r') as file:
            js_file = json.load(file)
            df = pd.DataFrame.from_dict([js_file])

            y = model.predict(df)
            pred = {'id': df.id.values[0], 'predict': y[0]}
            df_predict = df_predict.append(pred, ignore_index=True)
    df_predict.to_csv(f'{path}/data/predictions/predict.csv', index=False)


if __name__ == '__main__':
    predict()
