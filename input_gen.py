import codecs
import json
import os.path

import pandas as pd


def create_band_list_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    band_list = []
    for _, row in df.iterrows():
        values = row.values
        band_name, member_names = values[0], values[1:]
        member_names = member_names[~pd.isnull(member_names)]
        band_list.append(
            dict(
                band_name=band_name,
                member_names=member_names.tolist()
            )
        )
    return band_list


def create_config_from_values(n_room, n_timespan):
    return dict(
        n_room=n_room,
        n_timespan=n_timespan
    )


def write_json(name, band_list: list[dict], config: dict):
    json_root = dict(
        band_list=band_list,
        config=config
    )
    with codecs.open(f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(json_root, f, ensure_ascii=False, sort_keys=True, indent=2)


def main():
    csv_name = 'test.csv'

    name = os.path.splitext(csv_name)[0]

    write_json(
        name,
        band_list=create_band_list_from_csv(csv_name),
        config=create_config_from_values(
            n_room=7,
            n_timespan=9
        )
    )


if __name__ == '__main__':
    main()
