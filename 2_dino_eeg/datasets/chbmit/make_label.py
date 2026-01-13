import os
import sys
import h5py


def convert_time_to_seconds(time_str):
    # 将 HH:MM:SS 格式转换为秒
    time_str = time_str.replace(" ", "")
    hours, minutes, seconds = map(int, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def handle_part(section: str):
    result_dict = {}
    idx = 1
    for line in section.splitlines():
        # 分割冒号前后的部分并去掉空白
        key, value = map(str.strip, line.split(":", 1))

        if "Seizure" in key and "Start Time" in key:
            result_dict[f"Seizure {idx} Start Time"] = value.split(" seconds")[0]
        elif "Seizure" in key and "End Time" in key:
            result_dict[f"Seizure {idx} End Time"] = value.split(" seconds")[0]
            idx += 1
        else:
            result_dict[key] = value


    # 这些用不上
    # result_dict["File Start Time"] = convert_time_to_seconds(
    #     result_dict["File Start Time"]
    # )
    # result_dict["File End Time"] = convert_time_to_seconds(result_dict["File End Time"])
    # result_dict["Duration"] = (
    #     result_dict["File End Time"] - result_dict["File Start Time"]
    # )

    seiz_times = int(result_dict["Number of Seizures in File"])
    result_dict["Number of Seizures in File"] = seiz_times
    result_dict["Boxes"] = []

    if seiz_times > 0:
        idx = 1
        while idx <= seiz_times:
            start = int(result_dict[f"Seizure {idx} Start Time"])
            end = int(result_dict[f"Seizure {idx} End Time"])
            result_dict["Boxes"].append((start, end))
            idx += 1

    return result_dict


def handle(path, save_dir):
    with open(path, "r") as txt:
        content = txt.readlines()

    # 初始化一个列表用于存储结果
    sections = []
    current_section = []

    for line in content:
        # 检查是否是空行
        if line.strip() == "":
            if current_section:  # 如果当前部分不为空
                sections.append("".join(current_section).strip())
                current_section = []  # 重置当前部分
        else:
            current_section.append(line)  # 添加非空行到当前部分

    # 不要忘记添加最后一个部分（如果有的话）
    if current_section:
        sections.append("".join(current_section).strip())

    for section in sections:
        if section.startswith("File Name"):
            res_dict = handle_part(section)
            if res_dict:
                save_path = os.path.join(
                    save_dir, res_dict["File Name"].split(".edf")[0] + ".h5"
                )
                with h5py.File(save_path, "w") as shf:
                    shf.create_dataset("boxes", data=res_dict["Boxes"])
                    # shf.create_dataset("duration", data=res_dict["Duration"])


def main(base_dir, save_dir):
    txt_files = []
    for path, _, files in os.walk(base_dir):
        for name in files:
            if ".txt" in name:
                txt_files.append(os.path.join(path, name))

    for path in txt_files:
        handle(path, save_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
