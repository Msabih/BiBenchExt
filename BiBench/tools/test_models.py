import sys
import os

def main():
    print("DFSMN Models Tests:")
    print("Please select an option:")

    options = ["No compression", "8_10", "8_12", "8_14", "8_16", "Quit"]

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
            print(f"You chose {selected_option}")
            
            if selected_option == "No compression":
                command = (
                        'python tools\\test.py '
                        '"test_models\\dfsmn\\bnn\\dfsmn_bnn2.py" '
                        '"test_models\\dfsmn\\bnn\\checkpoint.pth" '
                        '--work-dir "results\\result_no_compression"'
                    )
            elif selected_option == "8_10":
                command = (
                        'python tools\\test.py '
                        '"test_models\\dfsmn\\8_10\\dfsmn_cbnn8_10.py" '
                        '"test_models\\dfsmn\\8_10\\checkpoint.pth" '
                        '--work-dir "results\\result_8_10"'
                    )
            elif selected_option == "8_12":
                command = (
                        'python tools\\test.py '
                        '"test_models\\dfsmn\\8_12\\dfsmn_cbnn.py" '
                        '"test_models\\dfsmn\\8_12\\checkpoint.pth" '
                        '--work-dir "results\\result_8_12"'
                    )
            elif selected_option == "8_14":
                command = (
                        'python tools\\test.py '
                        '"test_models\\dfsmn\\8_14\\dfsmn_cbnn8_14.py" '
                        '"test_models\\dfsmn\\8_14\\checkpoint.pth" '
                        '--work-dir "results\\result_8_14"'
                    )
            elif selected_option == "8_16":
                command = (
                        'python tools\\test.py '
                        '"test_models\\dfsmn\\8_16\\dfsmn_cbnn8_16.py" '
                        '"test_models\\dfsmn\\8_16\\checkpoint.pth" '
                        '--work-dir "results\\result_8_16"'
                    )
        os.system(command)

if __name__ == "__main__":
    main()
