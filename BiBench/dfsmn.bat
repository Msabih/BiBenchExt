python tools\train.py "D:\study\S3\Project_thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_bnn.py" --work-dir "D:\study\S3\Project_thesis\Results\Speech detection\dfsmn_bnn" --gpus 1
echo "Finished dfsmn_bnn!"
python tools\train.py "D:\study\S3\Project_thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn8_10.py" --work-dir "D:\study\S3\Project_thesis\Results\Speech detection\dfsmn_cbnn\8_10" --gpus 1
echo "Finished dfsmn_cbnn8_10!"
python tools\train.py "D:\study\S3\Project_thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn.py" --work-dir "D:\study\S3\Project_thesis\Results\Speech detection\dfsmn_cbnn\8_12" --gpus 1
echo "Finished dfsmn_cbnn8_12!"
python tools\train.py "D:\study\S3\Project_thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn8_14.py" --work-dir "D:\study\S3\Project_thesis\Results\Speech detection\dfsmn_cbnn\8_14" --gpus 1
echo "Finished dfsmn_cbnn8_14!"
python tools\train.py "D:\study\S3\Project_thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn8_16.py" --work-dir "D:\study\S3\Project_thesis\Results\Speech detection\dfsmn_cbnn\8_16" --gpus 1
echo "Finished dfsmn_cbnn8_16!"
pause