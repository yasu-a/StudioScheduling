from project_data import ProjectData


def main():
    json_name = 'test.json'

    proj_data = ProjectData.from_input_json_path(json_name)
    proj_data.dump_cache()


if __name__ == '__main__':
    main()
