import time
import os
import platform
import docker 
import filecmp
import difflib


def get_default_platform() -> str:
    machine = platform.uname().machine
    machine = {"x86_64": "amd64"}.get(machine, machine)
    return f"linux/{machine}"


class SecurityException(Exception):
    pass

def verify_critical_files_were_not_altered():
    files_to_check = ['bot/DO_NOT_TOUCH/evaluate.py', 'bot/DO_NOT_TOUCH/environment.py']
    differences_found = False

    for file in files_to_check:
        original_file = file
        new_file = f'bot/{file.split("/")[-1]}'

        if not filecmp.cmp(original_file, new_file):
            differences_found = True
            print(f"Differences found in {file}:")
            with open(original_file, 'r') as orig, open(new_file, 'r') as new:
                orig_lines = orig.readlines()
                new_lines = new.readlines()
                diff = difflib.unified_diff(orig_lines, new_lines, fromfile="original", tofile="new", lineterm='')
                for line in diff:
                    print(line)

    if differences_found:
        raise SecurityException("Critical files were modified! Aborting task.")

if __name__ == '__main__':
    verify_critical_files_were_not_altered()

    # make sure docker is running on your system first
    client = docker.from_env()

    docker_image_tag = 'test-bot'
    client.images.build(path='./bot', tag=docker_image_tag, platform=get_default_platform(), quiet=False)


    current_dir = os.getcwd() # docker needs us to specify an absolute path

    current_timestamp_as_string = str(int(time.time())) 
    data_dir = os.path.join(current_dir, "bot", "results")
    output_file = os.path.join(data_dir, f"{current_timestamp_as_string}.json")

    input_file = os.path.join("bot", "data", "validation_data.csv")
    with open(input_file, 'r') as file:
        data = file.read()
    
    input_file_in_data_dir = os.path.join(data_dir, "input-data.csv")
    with open(input_file_in_data_dir, 'w') as file:
        file.write(data)

    container = client.containers.run(
        docker_image_tag, 
        command=f"python bot/evaluate.py --output_file {output_file} --data {input_file_in_data_dir} --present_index {0} --initial_soc {7.5} --initial_profit {0}", 
        volumes={data_dir: {'bind': data_dir, 'mode': 'rw'}},
        detach=True,
        network_mode="none",
    )

    for line in container.logs(stream=True):
        print(line.strip())

    container.stop()
    container.remove()

    if os.path.exists(os.path.join(data_dir, output_file)):
        print(f"Policy run successfully, output can be found at: {os.path.join(data_dir, output_file)}")
    else:
        raise Exception(f"Policy run failed, no output file found at {os.path.join(data_dir, output_file)}")