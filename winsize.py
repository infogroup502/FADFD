import subprocess
import os
import sys


def main(win_size, fixed_product):
    script_path = os.path.abspath("main.py")
    python_executable = sys.executable

    # 遍历可能的 count 值，并计算 win_size_1
    count = 2
    while count <= fixed_product:
        win_size_1 = fixed_product // count

        # 确保 win_size_1 * count 等于固定的值
        if win_size_1 * count == fixed_product:
            print(f"处理 win_size={win_size}, win_size_1={win_size_1}, count={count}")

            result = subprocess.run(
                [python_executable, script_path,
                 "--win_size", str(win_size),
                 "--win_size_1", str(win_size_1),
                 "--count", str(count)],
                capture_output=True, text=True
            )

            print(result.stdout)
            if result.stderr:
                print("错误：", result.stderr)

        count += 2


if __name__ == "__main__":
    win_size = 30 # 示例固定的 win_size 值
    fixed_product = 120  # count * win_size_1 固定值

    main(win_size, fixed_product)