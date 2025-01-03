import sys
import os

def main():
    print("DFSMN Models Tests:")
    print("Please select an option:")

    options = ["No compression", "8_10", "8_12", "8_14", "8_16", "Quit"]
    options2 = ["yes", "No", "Quit"]

    while True:
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input("Enter your choice (1-6): "))
            if choice < 1 or choice > len(options):
                raise ValueError
        except ValueError:
            print("Invalid option, please try again.")
            continue

        selected_option = options[choice - 1]

        if selected_option == "Quit":
            print("Exiting...")
            break
        else:
            stuck=True
            print(f"You chose {selected_option}")
            while stuck:
                print("Convert to tflite?")

                for i, option in enumerate(options2, 1):
                    print(f"{i}. {option}")

                try:
                    choice = int(input("Enter your choice (1-3): "))
                    if choice < 1 or choice > len(options):
                        raise ValueError
                    else:
                        stuck=False
                except ValueError:
                    print("Invalid option, please try again.")
                    continue

            convert = options2[choice - 1]

            if convert == "Quit":
                print("Exiting...")
                break
            if convert == "yes":
                convert=True
            else:
                convert=False    

            if selected_option == "No compression" and  convert:
                command = (
                        f'python tools{os.sep}test_convert.py '
                        f'"test_models{os.sep}dfsmn{os.sep}bnn{os.sep}dfsmn_bnn2.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}bnn{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_no_compression"'
                    )
            elif selected_option == "8_10" and  convert:
                command = (
                        f'python tools{os.sep}test_convert.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_10{os.sep}dfsmn_cbnn8_10.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_10{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_10"'
                    )
            elif selected_option == "8_12" and  convert:
                command = (
                        f'python tools{os.sep}test_convert.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_12{os.sep}dfsmn_cbnn.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_12{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_12"'
                    )
            elif selected_option == "8_14" and  convert:
                command = (
                        f'python tools{os.sep}test_convert.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_14{os.sep}dfsmn_cbnn8_14.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_14{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_14"'
                    )
            elif selected_option == "8_16" and not convert:
                command = (
                        f'python tools{os.sep}test_convert.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_16{os.sep}dfsmn_cbnn8_16.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_16{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_16"'
                    )
            if selected_option == "No compression" and not convert:
                command = (
                        f'python tools{os.sep}test.py '
                        f'"test_models{os.sep}dfsmn{os.sep}bnn{os.sep}dfsmn_bnn2.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}bnn{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_no_compression"'
                    )
            elif selected_option == "8_10" and not convert:
                command = (
                        f'python tools{os.sep}test.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_10{os.sep}dfsmn_cbnn8_10.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_10{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_10"'
                    )
            elif selected_option == "8_12" and not convert:
                command = (
                        f'python tools{os.sep}test.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_12{os.sep}dfsmn_cbnn.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_12{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_12"'
                    )
            elif selected_option == "8_14" and not convert:
                command = (
                        f'python tools{os.sep}test.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_14{os.sep}dfsmn_cbnn8_14.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_14{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_14"'
                    )
            elif selected_option == "8_16" and not convert:
                command = (
                        f'python tools{os.sep}test.py '
                        f'"test_models{os.sep}dfsmn{os.sep}8_16{os.sep}dfsmn_cbnn8_16.py" '
                        f'"test_models{os.sep}dfsmn{os.sep}8_16{os.sep}checkpoint.pth" '
                        f'--work-dir "results{os.sep}result_8_16"'
                    )
        os.system(command)

if __name__ == "__main__":
    main()
